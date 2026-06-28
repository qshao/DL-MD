"""Hybrid ML-MD pipeline: model proposals → reconstruction → OpenMM MD → analysis.

Usage
-----
python scripts/hybrid_pipeline.py \
    --checkpoint checkpoints/kras_ft.pt \
    --shard      data/kras_wt_shard.pt \
    --ref_traj   WT/WT-sol6.trr --ref_top WT/WT-sol6.gro \
    --objective  explore \
    --n_proposals 200 --n_parallel 4 \
    --device cuda --out kras_hybrid_explore
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

_MD_NS_DEFAULT = {"explore": 10, "kinetics": 50, "fes": 25}
_N_PROPOSALS_DEFAULT = {"explore": 200, "kinetics": 500, "fes": 300}


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _run_proposals_subprocess(args):
    """Call explore_conformations.py as a subprocess for Stage 1."""
    mode = "explore" if args.objective == "explore" else "sample"
    out = os.path.join(args.out, "proposals")
    cmd = [
        sys.executable, "scripts/explore_conformations.py",
        "--checkpoint", args.checkpoint,
        "--shard",      args.shard,
        "--mode",       mode,
        "--n_explore",  str(args.n_proposals),
        "--n_steps",    str(args.n_steps),
        "--tau_ps",     str(args.tau_ps),
        "--temp_K",     str(args.temp_K),
        "--device",     args.device,
        "--out",        out,
        "--seed",       str(args.seed),
    ]
    if args.objective == "explore":
        cmd += [
            "--k_guide",      str(args.k_guide),
            "--sigma_cv",     str(args.sigma_cv),
            "--guide_warmup", str(args.guide_warmup),
        ]
    print(f"[Stage 1] Running model proposals ({args.n_proposals} attempts)…")
    subprocess.run(cmd, check=True)


def run_proposals(args):
    """Stage 1: generate Cα proposals with the ML model."""
    done = os.path.join(args.out, ".stage1_done")
    if os.path.exists(done):
        print("[Stage 1] Already done — skipping.")
        return
    proposals_dir = os.path.join(args.out, "proposals")
    # Fix 3: compare against accepted candidate PDB files, not summary.json entry
    # count (summary only holds accepted structures, so accepted < n_proposals always,
    # causing the old n >= n_proposals check to never pass).
    candidates_dir = os.path.join(proposals_dir, "candidates")
    n_candidates = len(glob.glob(os.path.join(candidates_dir, "*.pdb")))
    if n_candidates > 0:
        Path(done).touch()
        print(f"[Stage 1] Found {n_candidates} existing candidates, skipping.")
        return
    _run_proposals_subprocess(args)
    Path(done).touch()
    # Fix 1: for fes objective, build cv_basis.pt if absent so Stage 4 can project frames.
    # explore_conformations.py (sample mode) never writes this file, so we build it here
    # from the shard immediately after Stage 1 completes.
    if args.objective == "fes":
        cv_basis_path = os.path.join(proposals_dir, "cv_basis.pt")
        if not os.path.exists(cv_basis_path):
            import torch
            from lsmd.cv_guidance import CVSpace
            shard = torch.load(args.shard, map_location="cpu", weights_only=False)
            ref_ca = shard["t"]  # [F, N, 3]
            cv = CVSpace(n_pc=5)
            cv.fit(ref_ca)
            torch.save(cv, cv_basis_path)
            print(f"[Stage 1] Built cv_basis.pt from shard ({ref_ca.shape[0]} frames)")


def run_reconstruction(args):
    """Stage 2: Cα → all-atom heavy-atom PDB via AllAtomReconstructor."""
    done = os.path.join(args.out, ".stage2_done")
    if os.path.exists(done):
        print("[Stage 2] Already done — skipping.")
        return

    from lsmd.reconstruct import AllAtomReconstructor
    import mdtraj as md

    print(f"[Stage 2] Loading template trajectory {args.ref_traj} …")
    rec = AllAtomReconstructor(args.ref_traj, args.ref_top)

    proposals_dir = os.path.join(args.out, "proposals", "candidates")
    allatom_dir   = os.path.join(args.out, "allatom")
    os.makedirs(allatom_dir, exist_ok=True)

    failed = []
    pdb_files = sorted(f for f in os.listdir(proposals_dir) if f.endswith(".pdb"))
    print(f"[Stage 2] Reconstructing {len(pdb_files)} structures…")

    for pdb_file in pdb_files:
        out_pdb = os.path.join(allatom_dir, pdb_file)
        if os.path.exists(out_pdb):
            continue
        ca_pdb = os.path.join(proposals_dir, pdb_file)
        try:
            ca_traj   = md.load(ca_pdb)
            ca_gen_A  = torch.tensor(ca_traj.xyz[0] * 10.0)  # nm → Å
            xyz_A     = rec.reconstruct_frame_ca(ca_gen_A)    # [N_heavy, 3] Å
            out_traj  = md.Trajectory(
                xyz_A[np.newaxis] / 10.0,
                rec._out_top,   # no public topology accessor on AllAtomReconstructor; _out_top is the protein-heavy-atom MDTraj Topology built in __init__  # noqa: SLF001
            )
            out_traj.save_pdb(out_pdb)
        except Exception as exc:
            failed.append(f"{pdb_file}: {exc}")

    if failed:
        fail_log = os.path.join(allatom_dir, "failed.txt")
        with open(fail_log, "w") as fh:
            fh.write("\n".join(failed))
        print(f"[Stage 2] {len(failed)} reconstruction failures — see {fail_log}")

    print(f"[Stage 2] Done. {len(pdb_files) - len(failed)} all-atom PDBs written.")
    Path(done).touch()


def _md_worker(pdb_path, out_dir, md_ns, temp_K):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    from lsmd.md_validation import run_md
    return run_md(pdb_path, out_dir, md_ns, temp_K=temp_K)


def run_md_validation(args):
    """Stage 3: parallel OpenMM MD runs."""
    done = os.path.join(args.out, ".stage3_done")
    if os.path.exists(done):
        print("[Stage 3] Already done — skipping.")
        return

    allatom_dir  = os.path.join(args.out, "allatom")
    md_runs_dir  = os.path.join(args.out, "md_runs")
    md_ns = args.md_ns if args.md_ns is not None else _MD_NS_DEFAULT[args.objective]

    pdb_files = sorted(f for f in os.listdir(allatom_dir) if f.endswith(".pdb"))
    tasks = []
    for pdb_file in pdb_files:
        struct_id = pdb_file[:-4]
        run_dir   = os.path.join(md_runs_dir, struct_id)
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            continue
        tasks.append((os.path.join(allatom_dir, pdb_file), run_dir, md_ns))

    print(f"[Stage 3] Running {len(tasks)} MD jobs "
          f"({md_ns} ns each, {args.n_parallel} workers)…")

    n_stable = 0
    with ProcessPoolExecutor(max_workers=args.n_parallel) as ex:
        futs = {ex.submit(_md_worker, pdb, out_d, ns, args.temp_K): pdb
                for pdb, out_d, ns in tasks}
        for fut in as_completed(futs):
            pdb = futs[fut]
            try:
                m = fut.result()
                status = "stable" if m["stable"] else "unstable"
                print(f"  {os.path.basename(pdb)}: {status}  "
                      f"rmsd_final={m['rmsd_final_A']} Å", flush=True)
                if m["stable"]:
                    n_stable += 1
            except Exception as exc:
                print(f"  {os.path.basename(pdb)}: ERROR {exc}", flush=True)

    # n_stable counts only this batch (tasks not already done); len(pdb_files) is all PDBs.
    print(f"[Stage 3] Done. {n_stable}/{len(tasks)} structures stable this batch "
          f"(of {len(pdb_files)} total PDBs).")
    Path(done).touch()


def run_analysis(args):
    """Stage 4: objective-specific analysis."""
    from lsmd import pipeline_analysis as pa

    md_runs_dir = os.path.join(args.out, "md_runs")
    results_dir = os.path.join(args.out, "results", args.objective)

    print(f"[Stage 4] Running analysis: {args.objective}")

    if args.objective == "explore":
        pa.analyze_explore(md_runs_dir, results_dir)

    elif args.objective == "kinetics":
        pa.analyze_kinetics(md_runs_dir, results_dir)

    elif args.objective == "fes":
        cv_basis = os.path.join(args.out, "proposals", "cv_basis.pt")
        if not os.path.exists(cv_basis):
            raise FileNotFoundError(
                f"cv_basis.pt not found at {cv_basis}. "
                "Run proposals stage first (it is written by explore_conformations.py)."
            )
        pa.analyze_fes(md_runs_dir, cv_basis, results_dir, temp_K=args.temp_K)


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Hybrid ML-MD pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard",      required=True)
    ap.add_argument("--ref_traj",   required=True,
                    help="All-atom MD trajectory for reconstruction template (TRR/DCD/XTC).")
    ap.add_argument("--ref_top",    required=True,
                    help="Topology matching ref_traj (GRO/PDB).")
    ap.add_argument("--objective",  required=True,
                    choices=["explore", "kinetics", "fes"])
    ap.add_argument("--n_proposals", type=int, default=None,
                    help="Model proposals to generate (default: 200/500/300 by objective).")
    ap.add_argument("--n_parallel",  type=int, default=4)
    ap.add_argument("--md_ns",       type=float, default=None,
                    help="MD length per structure in ns (default: 10/50/25 by objective).")
    ap.add_argument("--temp_K",      type=float, default=310.0)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",         default="hybrid_out")
    ap.add_argument("--seed",        type=int, default=42)
    # Proposal stage pass-throughs
    ap.add_argument("--n_steps",     type=int,   default=50)
    ap.add_argument("--tau_ps",      type=float, default=2000.0)
    ap.add_argument("--k_guide",     type=float, default=0.15)
    ap.add_argument("--sigma_cv",    type=float, default=0.8)
    ap.add_argument("--guide_warmup",type=int,   default=20)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.n_proposals is None:
        args.n_proposals = _N_PROPOSALS_DEFAULT[args.objective]
    os.makedirs(args.out, exist_ok=True)

    print(f"Hybrid pipeline — objective={args.objective}  out={args.out}")
    run_proposals(args)
    run_reconstruction(args)
    run_md_validation(args)
    run_analysis(args)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
