"""Stage 1 of Phase 3: fit the learnable conservative energy to MD CA frames.

Fits LearnedCGEnergy by denoising score-matching on corpus-pooled frames with
inverse-density weighting, then (optionally) gates on whether short Langevin
sampling from the fitted energy reproduces the MD free-energy surface.

Usage
-----
python scripts/fit_energy.py --shard data/atlas/3u7t_A.pt \\
    --steps 5000 --sigma 0.5 --kT 0.593 --out checkpoints/energy_theta.pt --gate
"""
import argparse
import torch

from lsmd.learned_energy import (LearnedCGEnergy, score_matching_loss,
                                 inverse_density_weights, langevin_sample)
from lsmd import transfer_validate as tv


def _load_frames(shard_paths):
    """Return a list of (t[F,N,3], res_type[N], chain_id[N]) per shard."""
    proteins = []
    for p in shard_paths:
        s = torch.load(p, map_location="cpu", weights_only=False)
        proteins.append((s["t"].float(), s["res_type"].long(), s["chain_id"].long()))
    return proteins


def fit(proteins, *, steps, sigma, kT, lr, bins=30, clip=10.0, seed=0):
    torch.manual_seed(seed)
    energy = LearnedCGEnergy()
    opt = torch.optim.Adam(energy.parameters(), lr=lr)
    # Precompute per-protein inverse-density weights over shared-PCA CV space.
    # tv.shared_pca takes [F, N, 3]; we pass the full trajectory as the reference.
    weights = []
    for t, _rt, _cid in proteins:
        mean, comps = tv.shared_pca(t, n_components=2)
        cv = tv.project_cv(t, mean, comps)
        weights.append(inverse_density_weights(cv, bins=bins, clip=clip))
    rng = torch.Generator().manual_seed(seed)
    for step in range(steps):
        pi = torch.randint(0, len(proteins), (), generator=rng).item()
        t, rt, cid = proteins[pi]
        fi = torch.randint(0, t.shape[0], (), generator=rng).item()
        opt.zero_grad()
        loss = weights[pi][fi] * score_matching_loss(
            energy, t[fi], rt, cid, sigma=sigma, kT=kT)
        loss.backward()
        opt.step()
        if step % max(1, steps // 10) == 0:
            print(f"step {step}  loss={float(loss):.4f}")
    return energy


def gate(energy, proteins, *, kT, threshold, n_steps=4000):
    """Return (passed, fes_js, rho). Uses the first protein as the reference."""
    from scipy.stats import spearmanr  # optional; fallback below if absent
    t, rt, cid = proteins[0]
    # tv.shared_pca takes [F, N, 3]; use the full reference trajectory
    mean, comps = tv.shared_pca(t, n_components=2)
    samples = langevin_sample(energy, t[0].clone(), rt, cid,
                              n_steps=n_steps, dt=5e-3, kT=kT, stride=5)
    cv_model = tv.project_cv(samples, mean, comps)
    cv_md = tv.project_cv(t, mean, comps)
    fes = tv.fes_comparison(cv_model, cv_md)["fes_js"]
    # energy–population correlation: per-MD-frame energy vs that frame's basin count
    with torch.no_grad():
        e_per = torch.tensor([float(energy(t[i], rt, cid)) for i in
                              range(0, t.shape[0], max(1, t.shape[0] // 200))])
    rho = 0.0
    try:
        cv_sub = cv_md[::max(1, t.shape[0] // 200)][: e_per.shape[0]]
        # population proxy: negative distance density (denser = more populated)
        from lsmd.learned_energy import inverse_density_weights as idw
        pop = 1.0 / idw(cv_sub, bins=20, clip=1e6)
        rho = float(spearmanr(e_per.numpy(), pop.numpy()).correlation)
    except Exception:
        rho = float("nan")
    passed = fes < threshold
    return passed, fes, rho


def main():
    ap = argparse.ArgumentParser(description="Phase 3 Stage 1: fit conservative energy")
    ap.add_argument("--shard", action="append", required=True, dest="shards")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--kT", type=float, default=0.593)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="checkpoints/energy_theta.pt")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    args = ap.parse_args()

    proteins = _load_frames(args.shards)
    energy = fit(proteins, steps=args.steps, sigma=args.sigma, kT=args.kT, lr=args.lr)
    energy.save(args.out)
    print(f"saved energy to {args.out}")

    if args.gate:
        passed, fes, rho = gate(energy, proteins, kT=args.kT,
                                threshold=args.gate_threshold)
        print(f"GATE: {'PASS' if passed else 'FAIL'}  fes_js={fes:.3f}  rho={rho:.3f}")


if __name__ == "__main__":
    main()
