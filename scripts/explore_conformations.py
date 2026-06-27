"""CV-guided conformation explorer / kinetic trajectory generator.

Two primary modes selected with --mode:

  explore  (default)
      CV-guided repulsion drives the model toward unexplored regions of
      conformational space. Each attempt starts from a random training frame.
      Output: diverse PDB ensemble + summary.json.

  kinetics
      No guidance; the learned propagator runs freely to reproduce the
      kinetics of the training MD. Rollouts are chained end-to-end into a
      single long trajectory. Output: trajectory.pt + per-step RMSD log.
      Validates against all-atom MD via RMSF / FES / implied timescales.

  sample
      Unguided sampling with random starting frames. Produces an equilibrium
      ensemble without kinetic fidelity commitment. Useful for FES estimation.

Usage
-----
# Conformation exploration (find new states)
python scripts/explore_conformations.py \\
    --checkpoint checkpoints/kras_ft.pt \\
    --shard data/kras_wt_shard.pt \\
    --mode explore \\
    --n_explore 1000 --n_steps 50 --tau_ps 2000 \\
    --k_guide 0.15 --sigma_cv 0.8 --guide_warmup 20 \\
    --device cuda --out kras_exploration

# Kinetic trajectory (accelerated MD, preserves rates)
python scripts/explore_conformations.py \\
    --checkpoint checkpoints/kras_ft.pt \\
    --shard data/kras_wt_shard.pt \\
    --mode kinetics \\
    --n_explore 200 --n_steps 50 --tau_ps 2000 \\
    --device cuda --out kras_kinetics

# Equilibrium sampling (FES estimation, no kinetics)
python scripts/explore_conformations.py \\
    --checkpoint checkpoints/kras_ft.pt \\
    --shard data/kras_wt_shard.pt \\
    --mode sample \\
    --n_explore 500 --n_steps 50 --tau_ps 2000 \\
    --device cuda --out kras_sample
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from lsmd import geometry as g
from lsmd import transfer_eval as te
from lsmd import validation as val
from lsmd import decoder as dec
from lsmd.cv_guidance import CVSpace


def _plot_coverage(cv_buffer, cv_space, ref_cv, out_dir, step):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    if ref_cv is not None and ref_cv.shape[0] > 0:
        ax.scatter(ref_cv[:, 0], ref_cv[:, 1], c="lightgrey", s=10,
                   label="training", zorder=1)
    if cv_buffer:
        gen = np.stack([c.numpy() for c in cv_buffer])
        sc = ax.scatter(gen[:, 0], gen[:, 1],
                        c=range(len(cv_buffer)), cmap="plasma",
                        s=20, zorder=2, label="generated")
        plt.colorbar(sc, ax=ax, label="generation index")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"CV coverage — {len(cv_buffer)} accepted (step {step})")
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_coverage.png"), dpi=120)
    plt.close(fig)


def _plot_kinetics(rmsd_log, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rmsd_log, lw=0.8)
    ax.set_xlabel("Rollout chunk"); ax.set_ylabel("RMSD from native (Å)")
    ax.set_title("Kinetic trajectory — RMSD vs native")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kinetics_rmsd.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def run_explore(args, net, sched, norm, shard, k_eff):
    """CV-guided exploration: diverse ensemble with repulsion in CV space."""
    ca_ref = shard["t"].float()        # [F, N, 3]
    mean_ca = ca_ref.mean(dim=0)       # [N, 3]

    cv_basis_path = os.path.join(args.out, "cv_basis.pt")
    summary_path  = os.path.join(args.out, "summary.json")
    if args.resume and not os.path.exists(cv_basis_path) and os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Cannot resume: {cv_basis_path} missing but {summary_path} exists. "
            "Restore cv_basis.pt or delete summary.json to start fresh."
        )
    if args.resume and os.path.exists(cv_basis_path):
        cv_space = CVSpace.load(cv_basis_path)
    else:
        cv_space = CVSpace(n_pc=args.n_pc)
        cv_space.fit(ca_ref)
        cv_space.save(cv_basis_path)

    cv_space.to(args.device)
    ref_cv = cv_space.project_batch(ca_ref.to(args.device)).cpu().numpy()

    results = []
    cv_buffer = []
    start_id = 0
    if args.resume and os.path.exists(summary_path):
        with open(summary_path) as fh:
            results = json.load(fh)
        for r in results:
            cv_buffer.append(torch.tensor(r["cv"], dtype=torch.float32))
        start_id = max((r["id"] for r in results), default=-1) + 1
        print(f"Resuming from {len(results)} accepted structures (next id={start_id})")

    if "R_aa" in shard:
        R0_all = g.so3_exp(shard["R_aa"].float())
    else:
        R0_all = shard["R"].float()

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    cand_dir = os.path.join(args.out, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    res_type_names = shard.get("seq", ["ALA"] * shard["n_res"])

    all_coords = []
    if args.resume and os.path.exists(os.path.join(args.out, "structures.pt")):
        saved = torch.load(os.path.join(args.out, "structures.pt"),
                           map_location="cpu", weights_only=False)
        all_coords = list(saved)

    # Use per-frame mean bond length, not mean-structure bonds (which are compressed
    # by conformational averaging and would reject all geometrically valid frames).
    ref_bond = (ca_ref[:, 1:] - ca_ref[:, :-1]).norm(dim=-1).mean().item()

    for attempt in range(start_id, args.n_explore):
        f_idx = torch.randint(ca_ref.shape[0], (1,), generator=rng).item()
        R0 = R0_all[f_idx].to(args.device)
        t0 = ca_ref[f_idx].to(args.device)

        traj = te.rollout(
            net, sched, norm, R0, t0,
            shard["res_type"].to(args.device),
            shard["chain_id"].to(args.device),
            shard["res_index"].to(args.device),
            steps=args.n_steps, tau_ps=args.tau_ps, k=k_eff,
            diff_steps=args.diff_steps, eta=args.eta, temp_K=args.temp_K,
            wca_sigma=args.wca_sigma, wca_eps=args.wca_eps, wca_lam=args.wca_lam,
            cv_space=cv_space, cv_buffer=cv_buffer,
            k_guide=args.k_guide, sigma_cv=args.sigma_cv,
            guide_warmup=args.guide_warmup,
            graph_rebuild_interval=args.graph_rebuild_interval,
            device=args.device,
        )
        x_final = traj[-1].cpu()

        geo = val.ca_geometry(x_final)
        clashes   = geo["clash_count"]
        bond_rmsd = abs(geo["ca_bond_mean"] - ref_bond)
        print(f"[{attempt+1}/{args.n_explore}] accepted={len(results)}"
              f"  clashes={clashes:.2f}  bond_rmsd={bond_rmsd:.4f}", flush=True)
        if clashes >= 0.5 or bond_rmsd >= 0.1:
            print(f"  -> REJECTED (geometry)", flush=True)
            continue

        cv_i = cv_space.project_single(x_final.to(cv_space.mean.device)).detach().cpu()
        cv_buffer.append(cv_i)
        all_coords.append(x_final)

        rmsd_native = ((x_final - mean_ca) ** 2).sum(-1).mean().sqrt().item()
        results.append({
            "id": attempt,
            "cv": cv_i.tolist(),
            "rmsd_native": round(rmsd_native, 4),
            "clashes": clashes,
            "bond_rmsd": round(bond_rmsd, 4),
            "md_pass": None,
            "md_rmsd_final": None,
            "md_rg_final": None,
        })

        dec.write_ca_pdb(x_final, res_type_names,
                         os.path.join(cand_dir, f"{attempt:05d}.pdb"))

        if len(results) % 10 == 0:
            with open(summary_path, "w") as fh:
                json.dump(results, fh, indent=2)
            np.save(os.path.join(args.out, "cv_coords.npy"),
                    np.stack([r["cv"] for r in results]))
            torch.save(torch.stack(all_coords),
                       os.path.join(args.out, "structures.pt"))
            _plot_coverage(cv_buffer, cv_space, ref_cv, args.out, attempt)
            print(f"[{attempt+1}/{args.n_explore}] accepted={len(results)}")

    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    if results:
        torch.save(torch.stack(all_coords), os.path.join(args.out, "structures.pt"))
        np.save(os.path.join(args.out, "cv_coords.npy"),
                np.stack([r["cv"] for r in results]))
        _plot_coverage(cv_buffer, cv_space, ref_cv, args.out, args.n_explore)

    print(f"Done. {len(results)} structures accepted out of {args.n_explore} attempts.")
    print(f"PDB candidates: {cand_dir}/")
    print(f"Summary: {summary_path}")


def run_kinetics(args, net, sched, norm, shard, k_eff):
    """Unguided chained trajectory for kinetic fidelity validation."""
    ca_ref   = shard["t"].float()       # [F, N, 3]
    mean_ca  = ca_ref.mean(dim=0)       # [N, 3]

    traj_path    = os.path.join(args.out, "trajectory.pt")
    summary_path = os.path.join(args.out, "kinetics_summary.json")

    if "R_aa" in shard:
        R0_all = g.so3_exp(shard["R_aa"].float())
    else:
        R0_all = shard["R"].float()

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    # Pick one random starting frame for the whole chained trajectory
    f_idx = torch.randint(ca_ref.shape[0], (1,), generator=rng).item()
    R_cur = R0_all[f_idx].to(args.device)
    t_cur = ca_ref[f_idx].to(args.device)

    res_type_names = shard.get("seq", ["ALA"] * shard["n_res"])
    cand_dir = os.path.join(args.out, "snapshots")
    os.makedirs(cand_dir, exist_ok=True)

    all_frames = [t_cur.cpu()]   # accumulate every frame across all chunks
    rmsd_log   = []
    records    = []

    print(f"Kinetics mode: {args.n_explore} chunks × {args.n_steps} steps × "
          f"{args.tau_ps} ps = "
          f"{args.n_explore * args.n_steps * args.tau_ps / 1e6:.2f} µs total",
          flush=True)

    for chunk in range(args.n_explore):
        traj, R_cur, t_cur = te.rollout(
            net, sched, norm, R_cur, t_cur,
            shard["res_type"].to(args.device),
            shard["chain_id"].to(args.device),
            shard["res_index"].to(args.device),
            steps=args.n_steps, tau_ps=args.tau_ps, k=k_eff,
            diff_steps=args.diff_steps, eta=args.eta, temp_K=args.temp_K,
            wca_sigma=args.wca_sigma, wca_eps=args.wca_eps, wca_lam=args.wca_lam,
            cv_space=None, cv_buffer=None, k_guide=0.0,
            graph_rebuild_interval=args.graph_rebuild_interval,
            return_state=True,
            device=args.device,
        )
        # traj: [n_steps+1, N, 3] — skip frame 0 (== last frame of previous chunk)
        chunk_frames = traj[1:].cpu()
        all_frames.extend(list(chunk_frames))

        x_final   = chunk_frames[-1]
        rmsd_nat  = ((x_final - mean_ca) ** 2).sum(-1).mean().sqrt().item()
        rmsd_log.append(rmsd_nat)

        geo = val.ca_geometry(x_final)
        records.append({
            "chunk": chunk,
            "rmsd_native": round(rmsd_nat, 4),
            "ca_bond_mean": round(geo["ca_bond_mean"], 4),
            "clash_count":  round(geo["clash_count"],  4),
        })

        print(f"[chunk {chunk+1}/{args.n_explore}]  RMSD={rmsd_nat:.2f} Å"
              f"  bond={geo['ca_bond_mean']:.3f} Å"
              f"  clashes={geo['clash_count']:.2f}", flush=True)

        # Snapshot PDB every 10 chunks
        if (chunk + 1) % 10 == 0:
            dec.write_ca_pdb(x_final, res_type_names,
                             os.path.join(cand_dir, f"chunk{chunk+1:05d}.pdb"))

        # Checkpoint trajectory every 50 chunks
        if (chunk + 1) % 50 == 0:
            torch.save(torch.stack(all_frames), traj_path)
            with open(summary_path, "w") as fh:
                json.dump(records, fh, indent=2)
            _plot_kinetics(rmsd_log, args.out)

    # Final save
    full_traj = torch.stack(all_frames)   # [total_frames, N, 3]
    torch.save(full_traj, traj_path)
    with open(summary_path, "w") as fh:
        json.dump(records, fh, indent=2)
    _plot_kinetics(rmsd_log, args.out)

    total_ns = args.n_explore * args.n_steps * args.tau_ps / 1000
    print(f"\nDone. {len(all_frames)} frames saved ({total_ns:.0f} ns total).")
    print(f"Trajectory: {traj_path}")
    print(f"Summary:    {summary_path}")
    print(f"Snapshots:  {cand_dir}/")
    print(f"Mean RMSD from native: {np.mean(rmsd_log):.2f} ± {np.std(rmsd_log):.2f} Å")


def run_sample(args, net, sched, norm, shard, k_eff):
    """Unguided ensemble sampling from random starting frames (equilibrium)."""
    ca_ref   = shard["t"].float()
    mean_ca  = ca_ref.mean(dim=0)

    summary_path = os.path.join(args.out, "summary.json")
    cand_dir     = os.path.join(args.out, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    res_type_names = shard.get("seq", ["ALA"] * shard["n_res"])

    if "R_aa" in shard:
        R0_all = g.so3_exp(shard["R_aa"].float())
    else:
        R0_all = shard["R"].float()

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    ref_bond = (ca_ref[:, 1:] - ca_ref[:, :-1]).norm(dim=-1).mean().item()

    results    = []
    all_coords = []

    for attempt in range(args.n_explore):
        f_idx = torch.randint(ca_ref.shape[0], (1,), generator=rng).item()
        R0 = R0_all[f_idx].to(args.device)
        t0 = ca_ref[f_idx].to(args.device)

        traj = te.rollout(
            net, sched, norm, R0, t0,
            shard["res_type"].to(args.device),
            shard["chain_id"].to(args.device),
            shard["res_index"].to(args.device),
            steps=args.n_steps, tau_ps=args.tau_ps, k=k_eff,
            diff_steps=args.diff_steps, eta=args.eta, temp_K=args.temp_K,
            wca_sigma=args.wca_sigma, wca_eps=args.wca_eps, wca_lam=args.wca_lam,
            cv_space=None, cv_buffer=None, k_guide=0.0,
            graph_rebuild_interval=args.graph_rebuild_interval,
            device=args.device,
        )
        x_final = traj[-1].cpu()

        geo       = val.ca_geometry(x_final)
        clashes   = geo["clash_count"]
        bond_rmsd = abs(geo["ca_bond_mean"] - ref_bond)
        print(f"[{attempt+1}/{args.n_explore}] accepted={len(results)}"
              f"  clashes={clashes:.2f}  bond_rmsd={bond_rmsd:.4f}", flush=True)
        if clashes >= 0.5 or bond_rmsd >= 0.1:
            print(f"  -> REJECTED (geometry)", flush=True)
            continue

        rmsd_native = ((x_final - mean_ca) ** 2).sum(-1).mean().sqrt().item()
        results.append({
            "id": attempt,
            "rmsd_native": round(rmsd_native, 4),
            "clashes": clashes,
            "bond_rmsd": round(bond_rmsd, 4),
        })
        all_coords.append(x_final)
        dec.write_ca_pdb(x_final, res_type_names,
                         os.path.join(cand_dir, f"{attempt:05d}.pdb"))

        if len(results) % 10 == 0:
            with open(summary_path, "w") as fh:
                json.dump(results, fh, indent=2)
            torch.save(torch.stack(all_coords),
                       os.path.join(args.out, "structures.pt"))

    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    if all_coords:
        torch.save(torch.stack(all_coords), os.path.join(args.out, "structures.pt"))

    print(f"Done. {len(results)} structures accepted out of {args.n_explore} attempts.")
    print(f"PDB candidates: {cand_dir}/")
    print(f"Summary: {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Conformation explorer / kinetic trajectory generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard",      required=True)
    ap.add_argument("--mode", default="explore",
                    choices=["explore", "kinetics", "sample"],
                    help=(
                        "explore: CV-guided diverse ensemble (default). "
                        "kinetics: chained unguided trajectory for kinetic validation. "
                        "sample: unguided equilibrium ensemble."
                    ))
    ap.add_argument("--n_explore", type=int, default=500,
                    help="explore/sample: max attempts. kinetics: number of rollout chunks.")
    ap.add_argument("--n_steps",   type=int, default=50)
    ap.add_argument("--tau_ps",    type=float, default=2000.0)
    ap.add_argument("--temp_K",    type=float, default=375.0)
    # Guidance (explore only)
    ap.add_argument("--k_guide",      type=float, default=0.05,
                    help="CV repulsion strength (explore mode only).")
    ap.add_argument("--sigma_cv",     type=float, default=1.0,
                    help="Gaussian width in CV space (explore mode only).")
    ap.add_argument("--guide_warmup", type=int,   default=50,
                    help="Min buffer size before repulsion activates (explore mode only).")
    ap.add_argument("--n_pc",         type=int,   default=3,
                    help="Number of PCA dims for CVSpace (explore mode only).")
    # Diffusion
    ap.add_argument("--ddim", action="store_true",
                    help="Fast deterministic DDIM: sets eta=0.0, diff_steps=10.")
    ap.add_argument("--diff_steps", type=int,   default=None)
    ap.add_argument("--eta",        type=float, default=None)
    # WCA
    ap.add_argument("--wca_sigma", type=float, default=4.5)
    ap.add_argument("--wca_eps",   type=float, default=0.3)
    ap.add_argument("--wca_lam",   type=float, default=0.05)
    # Performance
    ap.add_argument("--graph_rebuild_interval", type=int, default=1)
    # I/O
    ap.add_argument("--out",    default="explore_out")
    ap.add_argument("--resume", action="store_true",
                    help="(explore mode) resume from existing summary.json.")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.ddim:
        if args.diff_steps is None:
            args.diff_steps = 10
        if args.eta is None:
            args.eta = 0.0
    if args.diff_steps is None:
        args.diff_steps = 20
    if args.eta is None:
        args.eta = 1.0

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shard = torch.load(args.shard,      map_location="cpu", weights_only=False)
    net, sched, norm = te.load_checkpoint(ckpt, device=args.device)
    k_eff = ckpt["hparams"].get("k", 16)

    print(f"Mode: {args.mode} | device: {args.device} | "
          f"n_explore={args.n_explore} | n_steps={args.n_steps} | "
          f"tau_ps={args.tau_ps}", flush=True)

    if args.mode == "explore":
        run_explore(args, net, sched, norm, shard, k_eff)
    elif args.mode == "kinetics":
        run_kinetics(args, net, sched, norm, shard, k_eff)
    elif args.mode == "sample":
        run_sample(args, net, sched, norm, shard, k_eff)


if __name__ == "__main__":
    main()
