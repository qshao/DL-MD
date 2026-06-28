"""Active learning loop for single-protein conformational exploration.

Usage
-----
python scripts/active_learning.py \\
    --pdb             input.pdb                    \\
    --checkpoint      checkpoints/v2_256h_90k.pt   \\
    --out             my_protein_loop              \\
    --rounds          10                           \\
    --proposals       100                          \\
    --batch-size      20                           \\
    --md-ns           10                           \\
    --replay-cap      5000                         \\
    --novel-threshold 1.5                          \\
    --stop            coverage                     \\
    --stop-threshold  0.10                         \\
    --bootstrap-ns    10                           \\
    --fine-tune-steps 2000                         \\
    --n-parallel      4                            \\
    --device          cuda
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from lsmd.active_loop import (
    _pdb_to_shard, _min_rmsd_kabsch, bootstrap_check,
    shard_from_md_runs, build_replay_shard, check_convergence,
)
from lsmd.cv_guidance import CVSpace
from lsmd.decoder import write_ca_pdb
from lsmd.md_validation import run_md
from lsmd.reconstruct import AllAtomReconstructor
from lsmd.transfer_eval import load_checkpoint, rollout


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_completed_rounds(out_dir: str) -> dict:
    """Return {round_num: summary_dict} for all .done-stamped rounds."""
    completed = {}
    out_path = Path(out_dir)
    if not out_path.exists():
        return completed
    for entry in sorted(out_path.iterdir()):
        if not entry.name.startswith("round_"):
            continue
        done = entry / ".done"
        summary_path = entry / "round_summary.json"
        if done.exists() and summary_path.exists():
            with open(summary_path) as fh:
                completed[int(entry.name.split("_")[1])] = json.load(fh)
    return completed


# ---------------------------------------------------------------------------
# Novel filtering
# ---------------------------------------------------------------------------

def _filter_novel(proposals: list, accumulated_t: torch.Tensor,
                  novel_threshold: float):
    """Return (novel_list, min_rmsds_all) where novel proposals have min-RMSD > threshold.

    Args:
        proposals:        list of [N, 3] Cα tensors (from rollout).
        accumulated_t:    [F, N, 3] all accumulated Cα frames.
        novel_threshold:  Å — proposals with min-RMSD > this are novel.

    Returns:
        (novel: list[Tensor], min_rmsds: list[float])
    """
    novel, min_rmsds = [], []
    for prop in proposals:
        mr = _min_rmsd_kabsch(prop, accumulated_t)
        min_rmsds.append(mr)
        if mr > novel_threshold:
            novel.append(prop)
    return novel, min_rmsds


# ---------------------------------------------------------------------------
# Main round loop
# ---------------------------------------------------------------------------

def run_round(round_num: int, args, current_ckpt: str, protein_meta: dict,
              shard_1f: dict, accumulated_pt: str, prev_total_md_ns: float,
              prev_novel_fraction: float, prev_accumulated_t,
              loop_summary: list):
    """Execute one active learning round; return updated state or None if converged."""
    round_dir  = os.path.join(args.out, f"round_{round_num}")
    done_stamp = os.path.join(round_dir, ".done")
    os.makedirs(round_dir, exist_ok=True)

    # ── 1. Load current accumulated Cα frames ────────────────────────────────
    if os.path.exists(accumulated_pt):
        acc = torch.load(accumulated_pt, map_location="cpu", weights_only=False)
        accumulated_t = acc["t"]   # [F_acc, N, 3]
    else:
        accumulated_t = shard_1f["t"]  # F=1 from input PDB

    # ── 2. Build / update CV space ──────────────────────────────────────────
    cv_space = CVSpace(n_pc=5)
    cv_space.fit(accumulated_t)
    cv_basis_path = os.path.join(round_dir, "cv_basis.pt")
    cv_space.save(cv_basis_path)

    # ── 3. Load model for this round ─────────────────────────────────────────
    ckpt = torch.load(current_ckpt, map_location="cpu", weights_only=False)
    net, schedule, update_norm = load_checkpoint(ckpt, args.device)

    # Pre-fill CV buffer from accumulated frames (up to 500 entries)
    cv_buffer = []
    F_acc = accumulated_t.shape[0]
    for i in range(min(500, F_acc)):
        cv_buffer.append(cv_space.project_single(accumulated_t[i]).detach())

    R0 = shard_1f["R"][0].to(args.device)
    t0 = shard_1f["t"][0].to(args.device)

    # ── 4. Generate proposals ────────────────────────────────────────────────
    proposals_dir = os.path.join(round_dir, "proposals")
    os.makedirs(proposals_dir, exist_ok=True)
    seq = shard_1f.get("seq", ["ALA"] * shard_1f["n_res"])

    proposals_ca = []
    for i in range(args.proposals):
        traj = rollout(
            net, schedule, update_norm, R0, t0,
            shard_1f["res_type"].to(args.device),
            shard_1f["chain_id"].to(args.device),
            shard_1f["res_index"].to(args.device),
            steps=50, tau_ps=2000, k=12,
            diff_steps=20, eta=1.0, temp_K=375.0,
            cv_space=cv_space if len(cv_buffer) >= 50 else None,
            cv_buffer=cv_buffer,
            k_guide=0.05, sigma_cv=1.0, guide_warmup=50,
            device=args.device,
        )
        x_final = traj[-1].cpu()
        proposals_ca.append(x_final)
        cv_buffer.append(cv_space.project_single(x_final).detach())

        pdb_path = os.path.join(proposals_dir, f"prop_{i:04d}.pdb")
        write_ca_pdb(x_final, seq, pdb_path)

    # ── 5. Filter novel proposals ─────────────────────────────────────────────
    novel, min_rmsds = _filter_novel(proposals_ca, accumulated_t, args.novel_threshold)
    n_novel = len(novel)

    if n_novel == 0:
        print(f"[round {round_num}] No novel proposals — landscape exhausted; terminating.",
              flush=True)
        _write_summary(round_dir, round_num, args, proposals_ca, novel,
                       md_success=0, new_frames=0, total_md_ns=prev_total_md_ns,
                       novel_fraction=0.0, fes_js=float("nan"), converged=True,
                       prev_accumulated_t=prev_accumulated_t, accumulated_t=accumulated_t)
        Path(done_stamp).touch()
        return None

    # Random selection from novel candidates
    batch_size = min(args.batch_size, n_novel)
    selected_ca = random.sample(novel, batch_size)
    print(f"[round {round_num}] generated={args.proposals} novel={n_novel} selected={batch_size}",
          flush=True)

    # ── 6. Reconstruct all-atom structures ──────────────────────────────────
    allatom_dir = os.path.join(round_dir, "allatom")
    os.makedirs(allatom_dir, exist_ok=True)
    rec = AllAtomReconstructor(args.pdb, args.pdb)  # use input PDB as template
    allatom_pdbs = []
    for j, ca_struct in enumerate(selected_ca):
        xyz = rec.reconstruct_frame_ca(ca_struct)   # numpy [N_heavy, 3]
        import mdtraj as md
        traj_tmp = md.load(args.pdb)
        top_tmp  = traj_tmp.topology
        ha_idx   = top_tmp.select("protein and not type H")
        xyz_nm   = xyz / 10.0                        # Å → nm
        t_out = md.Trajectory(xyz_nm[None], traj_tmp.atom_slice(ha_idx).topology)
        out_pdb = os.path.join(allatom_dir, f"struct_{j:04d}.pdb")
        t_out.save_pdb(out_pdb)
        allatom_pdbs.append(out_pdb)

    # ── 7. Run MD validation (parallel) ──────────────────────────────────────
    md_runs_dir = os.path.join(round_dir, "md_runs")
    os.makedirs(md_runs_dir, exist_ok=True)
    md_run_dirs = []
    def _run_one(j_pdb):
        j, pdb = j_pdb
        run_dir_j = os.path.join(md_runs_dir, f"struct_{j:04d}")
        run_md(pdb, run_dir_j, md_ns=args.md_ns, temp_K=310.0)
        return run_dir_j

    with ThreadPoolExecutor(max_workers=args.n_parallel) as pool:
        md_run_dirs = list(pool.map(_run_one, enumerate(allatom_pdbs)))

    n_md_success = sum(
        1 for d in md_run_dirs
        if os.path.exists(os.path.join(d, "metrics.json")) and
           json.load(open(os.path.join(d, "metrics.json"))).get("error") is None
    )
    print(f"[round {round_num}] MD success: {n_md_success}/{batch_size}", flush=True)

    # ── 8. Extract frames and build replay shard ───────────────────────────
    new_R, new_t = shard_from_md_runs(md_run_dirs, dt_ps=200)
    new_frames = len(new_t) if new_t.shape[0] > 0 else 0

    if new_frames > 0:
        replay_shard = build_replay_shard(
            new_R, new_t, accumulated_pt, protein_meta,
            replay_cap=args.replay_cap, dt_ps=200.0
        )
        replay_shard_path = os.path.join(round_dir, "replay_shard.pt")
        torch.save(replay_shard, replay_shard_path)

        # ── 9. Fine-tune model ────────────────────────────────────────────
        next_ckpt = os.path.join(round_dir, "checkpoint.pt")
        subprocess.run([
            sys.executable, "scripts/train_transfer.py",
            "--shard",   replay_shard_path,
            "--resume",  args.checkpoint,   # always from universal base
            "--steps",   str(args.fine_tune_steps),
            "--lr",      "1e-4",
            "--hidden",  "256",
            "--layers",  "6",
            "--lags_ps", "200", "1000", "5000",
            "--time_reversal",
            "--device",  args.device,
            "--out",     next_ckpt,
        ], check=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    else:
        # No new frames: skip fine-tuning, carry forward previous checkpoint
        next_ckpt = current_ckpt
        replay_shard_path = None
        print(f"[round {round_num}] WARNING: no MD frames extracted; skipping fine-tune",
              flush=True)

    # ── 10. Check stopping criterion ────────────────────────────────────────
    total_md_ns = prev_total_md_ns + n_md_success * args.md_ns
    novel_fraction = n_novel / len(proposals_ca)

    # Load updated accumulated_t for FES criterion
    if os.path.exists(accumulated_pt):
        acc_now = torch.load(accumulated_pt, map_location="cpu", weights_only=False)["t"]
    else:
        acc_now = accumulated_t

    state = {
        "total_md_ns":         total_md_ns,
        "last_novel_fraction": novel_fraction,
        "accumulated_t":       acc_now,
        "prev_accumulated_t":  prev_accumulated_t,
        "round":               round_num,
    }
    converged, metric = check_convergence(args.stop, args.stop_threshold, state)

    # Determine metric label
    fes_js = metric if args.stop == "fes" else float("nan")

    _write_summary(round_dir, round_num, args, proposals_ca, novel,
                   md_success=n_md_success, new_frames=new_frames,
                   total_md_ns=total_md_ns,
                   novel_fraction=novel_fraction, fes_js=fes_js,
                   converged=converged, prev_accumulated_t=prev_accumulated_t,
                   accumulated_t=acc_now)
    Path(done_stamp).touch()

    return {
        "next_ckpt":          next_ckpt,
        "total_md_ns":        total_md_ns,
        "novel_fraction":     novel_fraction,
        "accumulated_t":      acc_now,
        "converged":          converged,
    }


def _write_summary(round_dir, round_num, args, all_proposals, novel_proposals,
                   md_success, new_frames, total_md_ns, novel_fraction, fes_js,
                   converged, prev_accumulated_t, accumulated_t):
    """Write round_summary.json and append to loop_summary.json."""
    n_acc_before = prev_accumulated_t.shape[0] if prev_accumulated_t is not None else 0
    summary = {
        "round":                   round_num,
        "n_proposals_generated":   len(all_proposals),
        "n_novel_filtered":        len(novel_proposals),
        "n_md_attempted":          min(args.batch_size, len(novel_proposals)),
        "n_md_success":            md_success,
        "new_frames_this_round":   new_frames,
        "total_frames_accumulated": n_acc_before + new_frames,
        "total_md_ns":             total_md_ns,
        "last_novel_fraction":     novel_fraction,
        "fes_js":                  None if fes_js != fes_js else fes_js,  # nan → None
        "converged":               converged,
        "stop_criterion":          args.stop,
        "stop_threshold":          args.stop_threshold,
    }
    with open(os.path.join(round_dir, "round_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    loop_path = os.path.join(args.out, "loop_summary.json")
    if os.path.exists(loop_path):
        with open(loop_path) as fh:
            loop_data = json.load(fh)
    else:
        loop_data = []
    loop_data.append(summary)
    with open(loop_path, "w") as fh:
        json.dump(loop_data, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Active learning loop for single-protein conformational exploration."
    )
    ap.add_argument("--pdb",             required=True,
                    help="Input heavy-atom PDB (crystal or AlphaFold structure).")
    ap.add_argument("--checkpoint",      required=True,
                    help="Universal pretrained checkpoint (.pt), e.g. v2_256h_90k.pt.")
    ap.add_argument("--out",             required=True,
                    help="Output directory (created if needed; resume-safe).")
    ap.add_argument("--rounds",          type=int,   default=10)
    ap.add_argument("--proposals",       type=int,   default=100,
                    help="DDIM proposals generated per round.")
    ap.add_argument("--batch-size",      type=int,   default=20,
                    help="Number of novel proposals to validate with MD per round.")
    ap.add_argument("--md-ns",           type=float, default=10.0,
                    help="MD validation length per structure (nanoseconds).")
    ap.add_argument("--replay-cap",      type=int,   default=5000,
                    help="Max frames in replay shard (controls fine-tuning cost).")
    ap.add_argument("--novel-threshold", type=float, default=1.5,
                    help="Min-RMSD (Å) to count a proposal as novel.")
    ap.add_argument("--stop",            choices=["budget", "coverage", "fes"],
                    default="coverage")
    ap.add_argument("--stop-threshold",  type=float, default=0.10,
                    help="Stopping value: ns (budget), fraction (coverage), JS (fes).")
    ap.add_argument("--bootstrap-ns",    type=float, default=10.0,
                    help="Bootstrap MD length (ns) if universal model geometry is poor.")
    ap.add_argument("--fine-tune-steps", type=int,   default=2000)
    ap.add_argument("--n-parallel",      type=int,   default=4,
                    help="Parallel MD worker threads.")
    ap.add_argument("--device",          default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Copy input PDB into output dir for provenance
    shutil.copy(args.pdb, os.path.join(args.out, "input.pdb"))

    accumulated_pt = os.path.join(args.out, "accumulated_frames.pt")

    # Resume: load already-completed rounds
    completed = _load_completed_rounds(args.out)
    loop_summary = [completed[r] for r in sorted(completed)]
    print(f"[active_learning] resuming from round {len(completed)} "
          f"(completed: {sorted(completed.keys())})", flush=True)

    # ── Round 0: bootstrap check ─────────────────────────────────────────────
    bootstrap_shard_path = os.path.join(args.out, "round_0", "bootstrap_shard.pt")
    protein_meta_path    = os.path.join(args.out, "protein_meta.pt")

    if 0 not in completed:
        os.makedirs(os.path.join(args.out, "round_0"), exist_ok=True)
        shard_1f = bootstrap_check(
            pdb_path=args.pdb,
            checkpoint=args.checkpoint,
            device=args.device,
            bootstrap_ns=args.bootstrap_ns,
            out_dir=os.path.join(args.out, "round_0", "bootstrap"),
        )
        torch.save(shard_1f, bootstrap_shard_path)
        protein_meta = {k: shard_1f[k]
                        for k in ("res_type", "chain_id", "res_index", "seq", "n_res")}
        torch.save(protein_meta, protein_meta_path)
    else:
        shard_1f     = torch.load(bootstrap_shard_path, map_location="cpu", weights_only=False)
        protein_meta = torch.load(protein_meta_path,    map_location="cpu", weights_only=False)

    # State carried between rounds
    if completed:
        last = completed[max(completed)]
        total_md_ns     = last["total_md_ns"]
        novel_fraction  = last["last_novel_fraction"]
    else:
        total_md_ns    = 0.0
        novel_fraction = 1.0

    # Previous accumulated_t for FES criterion
    if os.path.exists(accumulated_pt):
        prev_t = torch.load(accumulated_pt, map_location="cpu", weights_only=False)["t"]
    else:
        prev_t = None

    # ── Round loop ────────────────────────────────────────────────────────────
    for round_num in range(args.rounds):
        if round_num in completed:
            if completed[round_num]["converged"]:
                print(f"[active_learning] converged in round {round_num} (loaded from cache)",
                      flush=True)
                break
            continue  # already done, not converged

        # Determine current checkpoint
        if round_num == 0:
            current_ckpt = args.checkpoint
        else:
            prev_ckpt = os.path.join(args.out, f"round_{round_num - 1}", "checkpoint.pt")
            current_ckpt = prev_ckpt if os.path.exists(prev_ckpt) else args.checkpoint

        result = run_round(
            round_num=round_num,
            args=args,
            current_ckpt=current_ckpt,
            protein_meta=protein_meta,
            shard_1f=shard_1f,
            accumulated_pt=accumulated_pt,
            prev_total_md_ns=total_md_ns,
            prev_novel_fraction=novel_fraction,
            prev_accumulated_t=prev_t,
            loop_summary=loop_summary,
        )

        if result is None:
            print("[active_learning] early termination: no novel proposals.", flush=True)
            break

        total_md_ns    = result["total_md_ns"]
        novel_fraction = result["novel_fraction"]
        prev_t         = result["accumulated_t"]
        current_ckpt   = result["next_ckpt"]

        if result["converged"]:
            print(f"[active_learning] stopping criterion '{args.stop}' met at round {round_num}.",
                  flush=True)
            break

    # ── Symlinks to final outputs ─────────────────────────────────────────────
    final_ckpt  = os.path.join(args.out, "final_checkpoint.pt")
    final_shard = os.path.join(args.out, "final_shard.pt")

    # Find last round's checkpoint
    for r in range(args.rounds - 1, -1, -1):
        ckpt_path = os.path.join(args.out, f"round_{r}", "checkpoint.pt")
        if os.path.exists(ckpt_path):
            if os.path.lexists(final_ckpt):
                os.remove(final_ckpt)
            os.symlink(os.path.abspath(ckpt_path), final_ckpt)
            break

    if os.path.exists(accumulated_pt):
        if os.path.lexists(final_shard):
            os.remove(final_shard)
        os.symlink(os.path.abspath(accumulated_pt), final_shard)

    print(f"[active_learning] done. Outputs in {args.out}", flush=True)


if __name__ == "__main__":
    main()
