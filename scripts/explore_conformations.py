"""CV-guided conformation explorer.

Generates diverse protein Cα conformations by adding a history-dependent
repulsion in collective-variable (CV) space to the DDPM denoising guidance.
Outputs PDB candidates for external MD relaxation validation.

Usage
-----
python scripts/explore_conformations.py \
    --checkpoint checkpoints/v4_3u7t_A.pt \
    --shard data/atlas/3u7t_A.pt \
    --n_explore 500 --n_steps 50 --tau_ps 2000 \
    --k_guide 0.05 --sigma_cv 1.0 --guide_warmup 50 \
    --out explore_out/3u7t_A
"""
import argparse
import json
import os

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n_explore", type=int, default=500)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--tau_ps", type=float, default=2000.0)
    ap.add_argument("--temp_K", type=float, default=375.0)
    ap.add_argument("--k_guide", type=float, default=0.05)
    ap.add_argument("--sigma_cv", type=float, default=1.0)
    ap.add_argument("--guide_warmup", type=int, default=50)
    ap.add_argument("--n_pc", type=int, default=3)
    ap.add_argument("--diff_steps", type=int, default=20)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--out", default="explore_out")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-accepted structures in summary.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cand_dir = os.path.join(args.out, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

    # Load checkpoint and shard
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shard = torch.load(args.shard, map_location="cpu", weights_only=False)
    net, sched, norm = te.load_checkpoint(ckpt, device=args.device)
    k_eff = ckpt["hparams"].get("k", 16)
    res_type_names = shard.get("seq", ["ALA"] * shard["n_res"])

    # Get Cα coordinates from shard
    ca_ref = shard["t"].float()                      # [F, N, 3]
    mean_ca = ca_ref.mean(dim=0)                     # [N, 3]

    # Fit CVSpace on training frames
    cv_space = CVSpace(n_pc=args.n_pc)
    cv_space.fit(ca_ref)
    cv_space.save(os.path.join(args.out, "cv_basis.pt"))

    # Project training frames to CV for the coverage plot
    ref_cv = np.stack([
        cv_space.project_single(ca_ref[i]).numpy()
        for i in range(ca_ref.shape[0])
    ])  # [F, n_pc+2]

    # Resume: load existing summary and CV buffer
    summary_path = os.path.join(args.out, "summary.json")
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

    # Get initial SE(3) frame
    if "R_aa" in shard:
        R0_all = g.so3_exp(shard["R_aa"].float())
    else:
        R0_all = shard["R"].float()  # [F, N, 3, 3]

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    all_coords = []  # accumulate accepted Cα tensors for structures.pt
    if args.resume and os.path.exists(os.path.join(args.out, "structures.pt")):
        saved = torch.load(os.path.join(args.out, "structures.pt"),
                           map_location="cpu", weights_only=False)
        all_coords = list(saved)  # list of [N,3] tensors

    ref_bond = (mean_ca[1:] - mean_ca[:-1]).norm(dim=-1).mean().item()

    for attempt in range(start_id, args.n_explore):
        # Pick random training frame as starting point
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
            cv_space=cv_space, cv_buffer=cv_buffer,
            k_guide=args.k_guide, sigma_cv=args.sigma_cv,
            guide_warmup=args.guide_warmup,
            device=args.device,
        )
        x_final = traj[-1].cpu()    # [N, 3]

        # Geometry filter
        geo = val.ca_geometry(x_final)
        clashes = geo["clash_count"]
        bond_rmsd = abs(geo["ca_bond_mean"] - ref_bond)
        if clashes >= 0.5 or bond_rmsd >= 0.1:
            continue

        # Compute CV and add to buffer
        cv_i = cv_space.project_single(x_final).detach()
        cv_buffer.append(cv_i)
        all_coords.append(x_final)

        # RMSD from native mean structure
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

        # Save PDB
        dec.write_ca_pdb(x_final, res_type_names,
                         os.path.join(cand_dir, f"{attempt:05d}.pdb"))

        # Flush summary every 10 accepts
        if len(results) % 10 == 0:
            with open(summary_path, "w") as fh:
                json.dump(results, fh, indent=2)
            np.save(os.path.join(args.out, "cv_coords.npy"),
                    np.stack([r["cv"] for r in results]))
            torch.save(torch.stack(all_coords),
                       os.path.join(args.out, "structures.pt"))
            _plot_coverage(cv_buffer, cv_space, ref_cv, args.out, attempt)
            print(f"[{attempt+1}/{args.n_explore}] accepted={len(results)}")

    # Final save
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


if __name__ == "__main__":
    main()
