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
import sys
import time
import torch
from lsmd import data, model as m, decoder as dec, validation as val
from lsmd import geometry as g


# ---------------------------------------------------------------------------
# PDB writer for multi-MODEL trajectory
# ---------------------------------------------------------------------------

def write_trajectory_pdb(frames_list, res_names, path, gly_mask=None):
    """Write a list of conformation tensors as a multi-MODEL PDB.

    frames_list : list of tensors — [P,3] (CA), [P,2,3] (2-bead), or [P,4,3] (4-bead)
    res_names   : list of P residue name strings
    gly_mask    : optional bool [P] — skip CB for Gly residues in multi-bead modes
    path        : output file path
    """
    from lsmd import decoder as dec_mod
    lines = []
    for model_idx, frame in enumerate(frames_list, start=1):
        lines.append(f"MODEL     {model_idx:4d}")
        if frame.ndim == 2:              # CA-only [P, 3]
            for ri in range(frame.shape[0]):
                x, y, z = frame[ri].tolist()
                lines.append(
                    f"ATOM  {ri + 1:5d}  CA  {res_names[ri]:>3s} A{ri + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
                )
        elif frame.shape[1] == 2:        # 2-bead [P, 2, 3]
            serial = 1
            for ri in range(frame.shape[0]):
                for ai, (aname, elem) in enumerate(
                        zip(dec_mod._2BEAD_ATOM_NAMES, dec_mod._2BEAD_ELEMENTS)):
                    if ai == 1 and gly_mask is not None and gly_mask[ri]:
                        continue
                    x, y, z = frame[ri, ai].tolist()
                    lines.append(
                        f"ATOM  {serial:5d} {aname} {res_names[ri]:>3s} A{ri + 1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}"
                    )
                    serial += 1
        else:                            # 4-bead [P, 4, 3]
            serial = 1
            for ri in range(frame.shape[0]):
                for ai, (aname, elem) in enumerate(
                        zip(dec_mod._4BEAD_ATOM_NAMES, dec_mod._4BEAD_ELEMENTS)):
                    if ai == 3 and gly_mask is not None and gly_mask[ri]:
                        continue
                    x, y, z = frame[ri, ai].tolist()
                    lines.append(
                        f"ATOM  {serial:5d} {aname} {res_names[ri]:>3s} A{ri + 1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}"
                    )
                    serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Generative MD core
# ---------------------------------------------------------------------------

def _ca_from_frame(x, mode):
    """Extract CA positions [P, 3] from a bead frame regardless of mode."""
    if x.ndim == 2:      return x               # CA-only [P, 3]
    if x.shape[1] == 2:  return x[:, 0, :]      # 2-bead  CA at index 0
    return x[:, 1, :]                            # 4-bead  CA at index 1


def _nearest_md_frame(x, frames_t, mode, device):
    """Return the training frame closest to x by mean-centred CA-RMSD."""
    ca_cur = _ca_from_frame(x, mode)                        # [P, 3]
    if frames_t.ndim == 3:
        ca_ref = frames_t                                    # [F, P, 3]
    elif frames_t.shape[2] == 2:
        ca_ref = frames_t[:, :, 0, :]                       # 2-bead
    else:
        ca_ref = frames_t[:, :, 1, :]                       # 4-bead

    # Mean-centre to make comparison rotation-invariant
    ca_q = ca_cur - ca_cur.mean(0)
    ca_r = ca_ref.to(device) - ca_ref.to(device).mean(1, keepdim=True)
    sq   = ((ca_r - ca_q.unsqueeze(0)) ** 2).sum(-1).mean(-1)   # [F]
    return frames_t[int(sq.argmin().item())].to(device)


def run_chain(net, schedule, node_feats, edge_index, edge_feats,
              x_start, x_ref, tau, n_steps, diff_steps, eta, sigma_init,
              device, bond_correction=False, bond_target=3.8, bond_iters=10,
              min_energy=False, k_bond=10.0, k_clash=1.0, min_steps=100,
              mode="ca", gly_mask=None, rama_potential=None, k_rama=1.0,
              sample_mode="explore", anchor_every=50, frames_t=None):
    """Run a single autoregressive trajectory chain.

    sample_mode controls how errors are managed between steps:

      "explore"  (default) — after each step, if the conformation is invalid
          (bond / clash / Rg violation), the next step starts from the last
          valid frame rather than the bad one.  This prevents runaway unfolding
          while still allowing wide conformational exploration.

      "mimic"  — every ``anchor_every`` steps the current position is replaced
          by the nearest real MD frame (by CA-RMSD).  This keeps the trajectory
          inside the training distribution and reproduces the MD ensemble rather
          than exploring beyond it.

    Returns
    -------
    trajectory    : list of n_steps+1 tensors [P,…] on CPU (step 0 = start)
    displacements : list of n_steps floats — mean per-atom ‖Δ‖ per step
    validity      : list of n_steps dicts from val.check_conformation
    step_times    : list of n_steps floats — wall-clock seconds per step
    n_events      : int — revert count (explore) or re-anchor count (mimic)
    """
    x = x_start.to(device).clone()
    x_ref_dev   = x_ref.to(device)
    trajectory  = [x.cpu()]
    displacements, validity, step_times = [], [], []

    # explore mode: fall back to last valid frame on failure
    x_last_valid = x.clone()
    n_events = 0          # reverts (explore) or re-anchors (mimic)

    net.eval()
    with torch.no_grad():
        for step in range(n_steps):
            t0 = time.perf_counter()

            # ── 1. Sample displacement ──────────────────────────────────
            raw = m.sample_ddpm(
                net, node_feats, edge_index, edge_feats,
                K=1, tau=tau, schedule=schedule,
                steps=diff_steps, eta=eta, sigma_init=sigma_init,
            )[0]

            n_beads = {"4bead": 4, "2bead": 2}.get(mode, 1)
            delta   = raw.reshape(x.shape[0], n_beads, 3) if n_beads > 1 else raw

            # ── 2. Advance ──────────────────────────────────────────────
            x = x + delta

            # ── 3. Geometry correction ──────────────────────────────────
            if min_energy:
                if mode == "4bead":
                    x = val.minimize_energy_4bead(x, gly_mask=gly_mask,
                                                  k_bond=k_bond, k_clash=k_clash,
                                                  n_steps=min_steps,
                                                  rama_potential=rama_potential,
                                                  k_rama=k_rama)
                elif mode == "2bead":
                    x = val.minimize_energy_2bead(x, gly_mask=gly_mask,
                                                  k_bond=k_bond, k_clash=k_clash,
                                                  n_steps=min_steps)
                else:
                    x = val.minimize_energy(x, bond_target=bond_target,
                                            k_bond=k_bond, k_clash=k_clash,
                                            n_steps=min_steps)
            elif bond_correction and mode == "ca":
                ca_x = x.clone()
                for _ in range(bond_iters):
                    vecs  = ca_x[1:] - ca_x[:-1]
                    dists = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    corr  = vecs * (1.0 - bond_target / dists) * 0.5
                    ca_x[:-1] = ca_x[:-1] + corr
                    ca_x[1:]  = ca_x[1:]  - corr
                x = ca_x

            # ── 4. Kabsch re-align onto reference frame ─────────────────
            ca_bead_idx = {"4bead": 1, "2bead": 0}.get(mode, None)
            P_res       = x.shape[0]
            if ca_bead_idx is not None:
                ca_x     = x[:, ca_bead_idx, :]
                ca_ref   = x_ref_dev[:, ca_bead_idx, :]
                R, t_vec = g.kabsch(ca_ref, ca_x)
                x_flat   = x.reshape(P_res * n_beads, 3)
                x_flat   = x_flat @ R.transpose(-1, -2) + t_vec.unsqueeze(0)
                x        = x_flat.reshape(P_res, n_beads, 3)
            else:
                R, t_vec = g.kabsch(x_ref_dev, x)
                x = x @ R.transpose(-1, -2) + t_vec.unsqueeze(-2).squeeze(0)

            step_times.append(time.perf_counter() - t0)

            x_cpu = x.cpu()
            trajectory.append(x_cpu)
            disp = delta.norm(dim=-1).pow(2).mean().sqrt().item()
            displacements.append(disp)

            # ── 5. Physical validity check ──────────────────────────────
            if mode == "4bead":
                check = val.check_4bead_conformation(x_cpu, gly_mask=gly_mask,
                                                     rama_potential=rama_potential)
            elif mode == "2bead":
                check = val.check_2bead_conformation(x_cpu, gly_mask=gly_mask)
            else:
                check = val.check_conformation(x_cpu)
            validity.append(check)

            # ── 6. Sample-mode corrections ──────────────────────────────
            if sample_mode == "explore":
                # Revert starting point to last valid frame when the step
                # fails, so errors don't compound into runaway unfolding.
                if check["valid"]:
                    x_last_valid = x.clone()
                else:
                    x = x_last_valid
                    n_events += 1

            elif sample_mode == "mimic" and frames_t is not None:
                # Periodically snap back to the nearest real MD frame so
                # the trajectory stays inside the training distribution.
                if (step + 1) % anchor_every == 0:
                    x = _nearest_md_frame(x, frames_t, mode, device)
                    n_events += 1

    return trajectory, displacements, validity, step_times, n_events


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

    # Extract CA regardless of bead mode: 2bead→index 0, 4bead→index 1, ca→as-is
    def _ca(frame):
        if frame.ndim == 2:    return frame           # CA-only [P, 3]
        if frame.shape[1] == 2: return frame[:, 0, :] # 2-bead [P, 2, 3]
        return frame[:, 1, :]                         # 4-bead [P, 4, 3]

    # RMSD from start per step, per chain (CA-based)
    rmsd_chains = []
    for traj in trajectories:
        x0_ca = _ca(traj[0])
        rmsd_chains.append([
            round((_ca(traj[s]) - x0_ca).norm(dim=-1).pow(2).mean().sqrt().item(), 4)
            for s in range(1, len(traj))
        ])

    mean_rmsd = [
        round(sum(c[s] for c in rmsd_chains) / n_chains, 4)
        for s in range(n_steps)
    ]

    # RMSF over the generated trajectory, CA-based
    ca_frames = torch.stack([
        _ca(frame) for traj in trajectories for frame in traj[1:]
    ], dim=0)                                          # [N, P, 3]
    mu   = ca_frames.mean(0)
    rmsf = (ca_frames - mu).norm(dim=-1).pow(2).mean(0).sqrt()

    # Physical validity summary across all chains × steps
    all_checks = [c for chain_v in validity_all for c in chain_v]
    n_total    = len(all_checks)
    n_valid    = sum(1 for c in all_checks if c["valid"])
    n_bond_vio = sum(1 for c in all_checks if not c["bond_ok"])
    n_clash    = sum(1 for c in all_checks if not c["clash_free"])
    n_rg_vio   = sum(1 for c in all_checks if not c["rg_ok"])
    n_rama_vio = sum(1 for c in all_checks if not c.get("rama_ok", True))
    # Mean Ramachandran outlier % per step (only meaningful for 4-bead)
    rama_pcts  = [c.get("rama_outlier_pct", 0.0) for c in all_checks]
    mean_rama_pct = round(sum(rama_pcts) / max(len(rama_pcts), 1), 1)

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
            "n_steps_checked":      n_total,
            "n_valid":              n_valid,
            "valid_fraction":       round(n_valid / max(n_total, 1), 4),
            "n_bond_violations":    n_bond_vio,
            "n_clashes":            n_clash,
            "n_rg_violations":      n_rg_vio,
            "n_rama_violations":    n_rama_vio,
            "mean_rama_outlier_pct": mean_rama_pct,
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
    ap.add_argument("--ddim", action="store_true",
                    help="Fast deterministic DDIM sampling: sets eta=0.0, diff_steps=10. "
                         "Override with explicit --eta / --diff_steps.")
    ap.add_argument("--diff_steps",   type=int,   default=None)
    ap.add_argument("--eta",          type=float, default=None,
                    help="DDPM stochasticity (1.0 = full stochastic, 0.0 = DDIM deterministic)")
    ap.add_argument("--sigma_init",   type=float, default=1.0)
    ap.add_argument("--save_pt",      action="store_true",
                    help="Also save trajectory as .pt tensor [chains, steps+1, P, 3]")
    ap.add_argument("--correct_bonds", action="store_true",
                    help="SHAKE-style bond projection after each step (bonds only)")
    ap.add_argument("--min_energy",   action="store_true",
                    help="L-BFGS energy minimization after each step (bonds + clashes)")
    ap.add_argument("--bond_target",  type=float, default=3.8,
                    help="Target CA-CA bond length in Å (default 3.8)")
    ap.add_argument("--bond_iters",   type=int,   default=10,
                    help="SHAKE projection passes per step (default 10)")
    ap.add_argument("--k_bond",       type=float, default=10.0,
                    help="Bond spring constant for --min_energy (default 10.0)")
    ap.add_argument("--k_clash",      type=float, default=1.0,
                    help="Clash penalty weight for --min_energy (default 1.0)")
    ap.add_argument("--k_rama",       type=float, default=0.0,
                    help="Ramachandran penalty weight for 4-bead --min_energy (default 0 = disabled; "
                         "coupling to bond/clash terms is problematic; check validation output instead)")
    ap.add_argument("--min_steps",    type=int,   default=100,
                    help="Max L-BFGS iterations per step for --min_energy (default 100)")
    ap.add_argument("--sample_mode",  default="explore",
                    choices=["explore", "mimic"],
                    help="'explore' (default): revert to last valid frame on failure, "
                         "enabling wide conformational search without runaway unfolding. "
                         "'mimic': re-anchor to nearest MD frame every --anchor_every steps "
                         "to reproduce the training-data ensemble.")
    ap.add_argument("--anchor_every", type=int,   default=50,
                    help="Re-anchor interval for --sample_mode mimic (default 50 steps)")
    ap.add_argument("--device",       default=None)
    args = ap.parse_args()
    if args.ddim:
        if args.diff_steps is None:
            args.diff_steps = 10
        if args.eta is None:
            args.eta = 0.0
    if args.diff_steps is None:
        args.diff_steps = 50
    if args.eta is None:
        args.eta = 1.0

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
    mode         = hp.get("mode", "ca")
    gly_mask_raw = ckpt.get("gly_mask")
    gly_mask     = gly_mask_raw.cpu() if gly_mask_raw is not None else None
    rama_sd      = ckpt.get("rama_potential")
    rama_pot     = (val.RamachandranPotential.from_state_dict(rama_sd).to(device)
                    if rama_sd is not None else None)

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
    if args.min_energy:
        print(f"  Energy min     : ON  (L-BFGS, {args.min_steps} steps, "
              f"k_bond={args.k_bond}, k_clash={args.k_clash})")
    elif args.correct_bonds:
        print(f"  Bond correction: ON  (SHAKE, target={args.bond_target} Å, "
              f"{args.bond_iters} iters/step)")
    if args.sample_mode == "explore":
        print(f"  Sample mode    : explore  (revert to last valid frame on failure)")
    else:
        print(f"  Sample mode    : mimic  (re-anchor to nearest MD frame every "
              f"{args.anchor_every} steps)")
    print(f"  Device         : {device}")
    print()

    os.makedirs(args.out, exist_ok=True)
    res_names = ["ALA"] * P
    mode_labels = {"4bead": "4-bead (N, CA, C, CB)  point_dim=12",
                   "2bead": "2-bead (CA, CB)         point_dim=6",
                   "ca":    "1-bead (CA only)        point_dim=3"}
    print(f"  Mode           : {mode_labels.get(mode, mode)}")

    # Run chains
    all_trajs       = []
    all_disps       = []
    all_validity    = []
    all_times       = []
    total_events    = 0
    time_labels     = [f"{i * ps_step / 1000:.2f} ns" for i in range(args.steps + 1)]

    for chain_idx in range(args.n_chains):
        print(f"Chain {chain_idx + 1}/{args.n_chains} ...", flush=True)
        traj, disps, validity, step_times, n_events = run_chain(
            net, schedule, node_feats, edge_index, edge_feats,
            x_start, x_ref, args.tau, args.steps,
            args.diff_steps, args.eta, args.sigma_init, device,
            bond_correction=args.correct_bonds,
            bond_target=args.bond_target,
            bond_iters=args.bond_iters,
            min_energy=args.min_energy,
            k_bond=args.k_bond,
            k_clash=args.k_clash,
            min_steps=args.min_steps,
            mode=mode,
            gly_mask=gly_mask,
            rama_potential=rama_pot,
            k_rama=args.k_rama,
            sample_mode=args.sample_mode,
            anchor_every=args.anchor_every,
            frames_t=X_all,
        )
        all_trajs.append(traj)
        all_disps.append(disps)
        all_validity.append(validity)
        all_times.extend(step_times)
        total_events += n_events

        def _ca_f(f):
            if f.ndim == 2:     return f
            if f.shape[1] == 2: return f[:, 0, :]
            return f[:, 1, :]
        final_rmsd = (_ca_f(traj[-1]) - _ca_f(traj[0])).norm(dim=-1).pow(2).mean().sqrt().item()
        n_valid = sum(1 for c in validity if c["valid"])
        event_label = ("reverts" if args.sample_mode == "explore" else "re-anchors")
        print(f"  Done. Mean disp/step: {sum(disps)/len(disps):.3f} Å  "
              f"Final RMSD: {final_rmsd:.3f} Å  "
              f"Valid steps: {n_valid}/{len(validity)}  "
              f"{event_label}: {n_events}")

        # Per-chain multi-model PDB
        if args.n_chains > 1:
            chain_pdb = os.path.join(args.out, f"chain_{chain_idx}.pdb")
            write_trajectory_pdb(traj, res_names, chain_pdb)

    # Combined trajectory PDB (all chains interleaved as separate models)
    combined = [frame for traj in all_trajs for frame in traj]
    write_trajectory_pdb(combined, res_names,
                         os.path.join(args.out, "trajectory.pdb"),
                         gly_mask=gly_mask)
    if args.n_chains > 1:
        for ci, traj in enumerate(all_trajs):
            write_trajectory_pdb(traj, res_names,
                                 os.path.join(args.out, f"chain_{ci}.pdb"),
                                 gly_mask=gly_mask)

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
    metrics["energy_minimization"] = args.min_energy

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
    rama_str = (f", rama viol={v['n_rama_violations']} "
                f"(mean {v['mean_rama_outlier_pct']}% outliers/step)"
                if v.get("n_rama_violations", 0) > 0 or mode == "4bead" else "")
    print(f"  Valid steps     : {v['n_valid']}/{v['n_steps_checked']} "
          f"({v['valid_fraction']*100:.1f}%)  "
          f"[bond viol={v['n_bond_violations']}, "
          f"clashes={v['n_clashes']}, "
          f"Rg viol={v['n_rg_violations']}"
          f"{rama_str}]")
    event_label = "reverts" if args.sample_mode == "explore" else "re-anchors"
    print(f"  Sample mode     : {args.sample_mode}"
          + (f"  (anchor every {args.anchor_every} steps)" if args.sample_mode == "mimic" else "")
          + f"  {event_label}: {total_events}")
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
