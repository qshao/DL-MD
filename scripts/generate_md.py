"""Generative MD: use a trained CA-DDPM checkpoint to extend protein dynamics
autoregressively, replacing classical MD integration with neural-network sampling.

Each step samples one displacement Δ ~ p(Δ|τ) and advances the CA conformation
by one lag time (τ × 200 ps/frame).  Running N steps covers N×τ×200 ps of
conformational dynamics without solving Newton's equations.

Timescale reference
-------------------
  Classical MD step : 2 fs
  This model (τ=1)  : 200 ps/step  → 10⁵× speedup
  This model (τ=5)  : 1 ns/step    → 5×10⁵× speedup
  100 steps, τ=5    : 100 ns  in  ~100 neural-network evaluations

Usage
-----
  # Single trajectory (50 ns at 1 ns/step)
  python scripts/generate_md.py \\
      --checkpoint checkpoints/wt_200ep.pt \\
      --frames     data/wt_frames.pt \\
      --tau        5 \\
      --steps      50 \\
      --out        genmd_50steps

  # Ensemble of 4 trajectories from the same starting frame
  python scripts/generate_md.py \\
      --checkpoint checkpoints/wt_200ep.pt \\
      --frames     data/wt_frames.pt \\
      --tau        5 \\
      --steps      50 \\
      --n_chains   4 \\
      --out        genmd_ensemble

Outputs
-------
  <out>/trajectory.pdb        Multi-MODEL PDB (one MODEL per step, all chains)
  <out>/chain_<k>.pdb         Per-chain multi-MODEL PDB (if n_chains > 1)
  <out>/trajectory.pt         [n_chains, N+1, P, 3] float32 CA coordinates
  <out>/metrics.json          Per-step RMSD, displacement, final RMSF
"""
import argparse
import json
import math
import os
import time
import torch
from lsmd import data, model as m, decoder as dec, validation as val
from lsmd import geometry as g


# ---------------------------------------------------------------------------
# PDB writer for multi-MODEL trajectory
# ---------------------------------------------------------------------------

def write_trajectory_pdb(frames_list, res_names, path):
    """Write a list of CA coordinate tensors [P,3] as a multi-MODEL PDB.

    frames_list : list of [P,3] tensors
    res_names   : list of P residue name strings
    path        : output file path
    """
    lines = []
    for model_idx, ca in enumerate(frames_list, start=1):
        lines.append(f"MODEL     {model_idx:4d}")
        for ri in range(ca.shape[0]):
            x, y, z = ca[ri].tolist()
            lines.append(
                f"ATOM  {ri + 1:5d}  CA  {res_names[ri]:>3s} A{ri + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
        lines.append("ENDMDL")
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Bond-length correction
# ---------------------------------------------------------------------------

def correct_bond_lengths(ca, target=3.8, n_iter=10):
    """Project CA positions so all consecutive CA-CA distances equal target Å.

    Uses symmetric SHAKE-style constraint projection: each bond's two atoms
    are each moved by half the correction, preserving their shared midpoint.
    Repeating n_iter times converges to within ~0.01 Å of target for typical
    MD-generated conformations.

    Args:
        ca      : [P, 3] CA coordinates (modified in-place clone)
        target  : ideal CA-CA bond length in Å (default 3.8)
        n_iter  : number of projection passes (default 10)

    Returns:
        [P, 3] corrected CA coordinates
    """
    ca = ca.clone()
    for _ in range(n_iter):
        vecs  = ca[1:] - ca[:-1]                                   # [P-1, 3]
        dists = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-6)    # [P-1, 1]
        corr  = vecs * (1.0 - target / dists) * 0.5               # [P-1, 3]
        ca[:-1] = ca[:-1] + corr
        ca[1:]  = ca[1:]  - corr
    return ca


# ---------------------------------------------------------------------------
# Generative MD core
# ---------------------------------------------------------------------------

def run_chain(net, schedule, node_feats, edge_index, edge_feats,
              x_start, x_ref, tau, n_steps, diff_steps, eta, sigma_init,
              device, bond_correction=False, bond_target=3.8, bond_iters=10):
    """Run a single autoregressive trajectory chain.

    At each step:
      1. Sample displacement Δ ~ DDPM(τ)
      2. Advance: x ← x + Δ
      3. (optional) Project bond lengths back to bond_target Å
      4. Kabsch-realign x onto x_ref to prevent orientation drift
      5. Check physical validity with val.check_conformation

    Returns
    -------
    trajectory    : list of n_steps+1 tensors [P,3] on CPU (step 0 = start)
    displacements : list of n_steps floats — mean per-atom ‖Δ‖ per step
    validity      : list of n_steps dicts from val.check_conformation
    step_times    : list of n_steps floats — wall-clock seconds per step
    """
    x = x_start.to(device).clone()
    x_ref_dev = x_ref.to(device)
    trajectory = [x.cpu()]
    displacements = []
    validity = []
    step_times = []

    net.eval()
    with torch.no_grad():
        for step in range(n_steps):
            t0 = time.perf_counter()

            # Sample one displacement
            delta = m.sample_ddpm(
                net, node_feats, edge_index, edge_feats,
                K=1, tau=tau, schedule=schedule,
                steps=diff_steps, eta=eta, sigma_init=sigma_init,
            )[0]                                           # [P,3]

            # Advance
            x = x + delta

            # Bond-length correction: project each CA-CA bond back to target length
            if bond_correction:
                x = correct_bond_lengths(x, target=bond_target, n_iter=bond_iters)

            # Kabsch re-align onto reference frame (frame-0 canonical orientation)
            # This prevents slow orientation drift over many steps while preserving
            # all internal conformational change in delta.
            R, t = g.kabsch(x_ref_dev, x)                # aligns x onto x_ref
            x = x @ R.transpose(-1, -2) + t.unsqueeze(-2).squeeze(0)

            step_times.append(time.perf_counter() - t0)

            x_cpu = x.cpu()
            trajectory.append(x_cpu)
            disp = delta.norm(dim=-1).pow(2).mean().sqrt().item()
            displacements.append(disp)

            # Physical validity check on the generated conformation
            check = val.check_conformation(x_cpu)
            validity.append(check)

    return trajectory, displacements, validity, step_times


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trajectories, validity_all, tau, ps_per_frame=200):
    """Compute per-step and aggregate metrics over one or more chains.

    trajectories  : list of chains, each a list of [P,3] tensors (step 0 = start)
    validity_all  : list of chains, each a list of check_conformation dicts (n_steps each)
    """
    n_chains = len(trajectories)
    n_steps  = len(trajectories[0]) - 1
    ps_step  = tau * ps_per_frame

    # RMSD from start per step, per chain
    rmsd_chains = []
    for traj in trajectories:
        x0 = traj[0]
        rmsd_chains.append([
            round((traj[s] - x0).norm(dim=-1).pow(2).mean().sqrt().item(), 4)
            for s in range(1, len(traj))
        ])

    mean_rmsd = [
        round(sum(c[s] for c in rmsd_chains) / n_chains, 4)
        for s in range(n_steps)
    ]

    # RMSF over the generated trajectory (all chains pooled)
    all_frames = torch.stack([
        frame for traj in trajectories for frame in traj[1:]
    ], dim=0)
    mu   = all_frames.mean(0)
    rmsf = (all_frames - mu).norm(dim=-1).pow(2).mean(0).sqrt()

    # Physical validity summary across all chains × steps
    all_checks = [c for chain_v in validity_all for c in chain_v]
    n_total    = len(all_checks)
    n_valid    = sum(1 for c in all_checks if c["valid"])
    n_bond_vio = sum(1 for c in all_checks if not c["bond_ok"])
    n_clash    = sum(1 for c in all_checks if not c["clash_free"])
    n_rg_vio   = sum(1 for c in all_checks if not c["rg_ok"])

    return {
        "n_chains":               n_chains,
        "n_steps":                n_steps,
        "tau":                    tau,
        "time_per_step_ps":       ps_step,
        "total_time_ns":          round(n_steps * ps_step / 1000, 2),
        "timescale_vs_2fs_MD":    f"{tau * ps_per_frame * 1000 // 2}x per step",
        "mean_rmsd_from_start_A": mean_rmsd,
        "final_mean_rmsd_A":      mean_rmsd[-1] if mean_rmsd else 0.0,
        "rmsf_mean_A":            round(rmsf.mean().item(), 4),
        "rmsf_max_A":             round(rmsf.max().item(), 4),
        "rmsf_max_residue":       int(rmsf.argmax().item()),
        "rmsf_per_residue_A":     [round(v, 4) for v in rmsf.tolist()],
        "validity": {
            "n_steps_checked":    n_total,
            "n_valid":            n_valid,
            "valid_fraction":     round(n_valid / max(n_total, 1), 4),
            "n_bond_violations":  n_bond_vio,
            "n_clashes":          n_clash,
            "n_rg_violations":    n_rg_vio,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generative MD: autoregressive CA trajectory via DDPM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[0].strip(),
    )
    ap.add_argument("--checkpoint",   required=True, help="Checkpoint from train.py")
    ap.add_argument("--frames",       required=True, help="Preprocessed frames from preprocess.py")
    ap.add_argument("--tau",          type=int,   default=5,
                    help="Lag per step (frames, 200 ps/frame). τ=5 → 1 ns/step.")
    ap.add_argument("--steps",        type=int,   default=50,
                    help="Number of generative steps (total time = steps × tau × 200 ps)")
    ap.add_argument("--n_chains",     type=int,   default=1,
                    help="Number of independent trajectory chains")
    ap.add_argument("--source_frame", type=int,   default=None,
                    help="Starting CA frame index (default: first val frame at --tau)")
    ap.add_argument("--out",          default="genmd_out", help="Output directory")
    ap.add_argument("--diff_steps",   type=int,   default=50)
    ap.add_argument("--eta",          type=float, default=1.0,
                    help="DDPM stochasticity (1.0 = full stochastic, 0.0 = DDIM deterministic)")
    ap.add_argument("--sigma_init",   type=float, default=1.0)
    ap.add_argument("--save_pt",      action="store_true",
                    help="Also save trajectory as .pt tensor [chains, steps+1, P, 3]")
    ap.add_argument("--correct_bonds", action="store_true",
                    help="Project CA-CA bonds back to 3.8 Å after each step")
    ap.add_argument("--bond_target",  type=float, default=3.8,
                    help="Target CA-CA bond length for --correct_bonds (default 3.8 Å)")
    ap.add_argument("--bond_iters",   type=int,   default=10,
                    help="Projection passes per step for --correct_bonds (default 10)")
    ap.add_argument("--device",       default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device \
             else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    hp   = ckpt["hparams"]
    net  = m.FlowNet(
        node_dim=hp["node_dim"], edge_dim=hp["edge_dim"],
        hidden=hp["hidden"], layers=hp["layers"], point_dim=hp["point_dim"],
    ).to(device)
    net.load_state_dict(ckpt["net_state"])

    schedule   = m.NoiseSchedule(T=hp["T_diff"]).to(device)
    schedule.load_state_dict(ckpt["schedule_state"])
    node_feats = ckpt["node_feats"].to(device)
    edge_index = ckpt["edge_index"].to(device)
    edge_feats = ckpt["edge_feats"].to(device)
    taus_trained = hp["taus"]

    if args.tau not in taus_trained:
        print(f"Warning: --tau {args.tau} not in training taus {taus_trained}. "
              f"Model may generalise poorly.")

    # Load frames, pick starting point
    frames = torch.load(args.frames, map_location="cpu")
    X_all  = frames["t"]
    F, P   = X_all.shape[:2]

    pairs = data.make_multi_lag_pairs(F, taus_trained)
    _, val_pairs = data.time_split(pairs, val_frac=0.2)
    matching = val_pairs[val_pairs[:, 2] == args.tau]
    if args.source_frame is not None:
        src = args.source_frame
    elif matching.shape[0] > 0:
        src = int(matching[0, 0])
    else:
        src = int(val_pairs[0, 0])

    x_start = X_all[src]      # [P,3] — source frame in canonical (frame-0) orientation
    x_ref   = X_all[0]        # [P,3] — canonical reference for re-alignment

    ps_step    = args.tau * 200
    total_ns   = args.steps * ps_step / 1000
    print(f"Generative MD")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Starting frame : {src}  ({F} total, {P} residues)")
    print(f"  Tau per step   : {args.tau} frames = {ps_step} ps")
    print(f"  Steps          : {args.steps}")
    print(f"  Chains         : {args.n_chains}")
    print(f"  Total time     : {total_ns:.1f} ns  "
          f"({args.steps * args.tau * 200 * 1000 // 2}× faster than 2 fs MD)")
    if args.correct_bonds:
        print(f"  Bond correction: ON  (target={args.bond_target} Å, {args.bond_iters} iters/step)")
    print(f"  Device         : {device}")
    print()

    os.makedirs(args.out, exist_ok=True)
    res_names = ["ALA"] * P

    # Run chains
    all_trajs     = []
    all_disps     = []
    all_validity  = []
    all_times     = []
    time_labels   = [f"{i * ps_step / 1000:.2f} ns" for i in range(args.steps + 1)]

    for chain_idx in range(args.n_chains):
        print(f"Chain {chain_idx + 1}/{args.n_chains} ...", flush=True)
        traj, disps, validity, step_times = run_chain(
            net, schedule, node_feats, edge_index, edge_feats,
            x_start, x_ref, args.tau, args.steps,
            args.diff_steps, args.eta, args.sigma_init, device,
            bond_correction=args.correct_bonds,
            bond_target=args.bond_target,
            bond_iters=args.bond_iters,
        )
        all_trajs.append(traj)
        all_disps.append(disps)
        all_validity.append(validity)
        all_times.extend(step_times)

        n_valid = sum(1 for c in validity if c["valid"])
        print(f"  Done. Mean disp/step: {sum(disps)/len(disps):.3f} Å  "
              f"Final RMSD: {(traj[-1]-traj[0]).norm(dim=-1).pow(2).mean().sqrt().item():.3f} Å  "
              f"Valid steps: {n_valid}/{len(validity)}")

        # Per-chain multi-model PDB
        if args.n_chains > 1:
            chain_pdb = os.path.join(args.out, f"chain_{chain_idx}.pdb")
            write_trajectory_pdb(traj, res_names, chain_pdb)

    # Combined trajectory PDB (all chains interleaved as separate models)
    combined = [frame for traj in all_trajs for frame in traj]
    write_trajectory_pdb(combined, res_names, os.path.join(args.out, "trajectory.pdb"))

    # Save tensor
    if args.save_pt:
        traj_tensor = torch.stack([
            torch.stack(traj) for traj in all_trajs
        ])                                                  # [chains, steps+1, P, 3]
        torch.save(traj_tensor, os.path.join(args.out, "trajectory.pt"))
        print(f"Trajectory tensor saved: shape {list(traj_tensor.shape)}")

    # Metrics
    metrics = compute_metrics(all_trajs, all_validity, args.tau)
    metrics["source_frame"]    = src
    metrics["checkpoint"]      = args.checkpoint
    metrics["time_labels"]     = time_labels
    metrics["bond_correction"] = args.correct_bonds

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Timing report
    mean_step_s    = sum(all_times) / len(all_times)
    timing_path    = os.path.join(args.out, "timing_report.txt")
    timing_info    = val.timing_report(
        tau=args.tau,
        time_per_step_s=mean_step_s,
        out_path=timing_path,
        target_ns=1000,
    )

    # Summary printout
    v = metrics["validity"]
    print()
    print("=" * 60)
    print(f"Generated {total_ns:.1f} ns of CA dynamics")
    print(f"  Total steps     : {args.steps}  (τ={args.tau} → {ps_step} ps/step)")
    print(f"  Chains          : {args.n_chains}")
    print(f"  Final RMSD      : {metrics['final_mean_rmsd_A']:.3f} Å from start")
    print(f"  RMSF (mean/max) : {metrics['rmsf_mean_A']:.3f} / {metrics['rmsf_max_A']:.3f} Å")
    print(f"  Most flexible   : residue {metrics['rmsf_max_residue']}")
    print(f"  Valid steps     : {v['n_valid']}/{v['n_steps_checked']} "
          f"({v['valid_fraction']*100:.1f}%)  "
          f"[bond viol={v['n_bond_violations']}, "
          f"clashes={v['n_clashes']}, "
          f"Rg viol={v['n_rg_violations']}]")
    print(f"  Time/step       : {mean_step_s:.3f} s  "
          f"→ {timing_info['total_min']:.1f} min for 1000 ns "
          f"({timing_info['speedup_vs_classical_md']:.0f}× vs classical MD)")
    print(f"Output → {args.out}/")
    print(f"  trajectory.pdb  ({args.steps + 1} MODEL records per chain)")
    print(f"  metrics.json")
    print(f"  timing_report.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
