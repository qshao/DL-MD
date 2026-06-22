"""Baseline physics validation of a transferable checkpoint against MD shards.

Rolls the model out per shard, computes the kinetic + thermodynamic + structural
metric suite (lsmd.transfer_validate.validate), and writes a JSON report.

NOTE: these baselines measure fit quality, not generalization — all ATLAS shards
were seen in training and there is no held-out split yet. The report records
"heldout": false accordingly.

Usage
-----
python scripts/validate_physics.py \\
    --checkpoint checkpoints/v2_256h_90k.pt \\
    --shard data/atlas/3u7t_A.pt --shard data/atlas/1z0b_A.pt \\
    --steps 200 --tau_ps 2000 --diff_steps 20 --eta 1.0 \\
    --out validation_baseline.json
"""
import argparse
import json
import os

import torch

from lsmd import geometry as g
from lsmd import transfer_eval as te
from lsmd import transfer_validate as tv


def _protein_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def build_report(ckpt, shard_paths, settings, device):
    """Run rollout + validate for each shard. Returns the proteins dict."""
    net, sched, norm = te.load_checkpoint(ckpt, device=device)
    k_eff = ckpt["hparams"].get("k", settings["k"])
    proteins = {}
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if "R_aa" in shard:
            R0 = g.so3_exp(shard["R_aa"][0].float())
        else:
            R0 = shard["R"][0]
        t0 = shard["t"][0].float()
        traj = te.rollout(
            net, sched, norm, R0, t0,
            shard["res_type"], shard["chain_id"], shard["res_index"],
            steps=settings["steps"], tau_ps=settings["tau_ps"], k=k_eff,
            diff_steps=settings["diff_steps"], eta=settings["eta"],
            temp_K=settings["temp_K"],
            bond_constraint_iters=settings["bond_constraint_iters"],
            max_update_norm=settings["max_update_norm"],
            wca_sigma=settings["wca_sigma"], wca_eps=settings["wca_eps"],
            wca_lam=settings["wca_lam"],
            noether=settings.get("noether", False),
            device=device).cpu()
        rep = tv.validate(traj, shard["t"].float(),
                          tau_ps=settings["tau_ps"], dt_md_ps=float(shard["dt"]),
                          kT=settings["kT"], n_states=settings["n_states"])
        rep["n_res"] = int(shard["n_res"])
        rep["reweight"] = None
        proteins[_protein_id(path)] = rep
    return proteins


def summarize(proteins):
    """Mean headline metrics across proteins."""
    def mean(getter):
        vals = [getter(p) for p in proteins.values()]
        return float(sum(vals) / len(vals)) if vals else float("nan")
    return {
        "mean_rmsf_corr": mean(lambda p: p["structural"]["rmsf_corr"]),
        "mean_dist_js": mean(lambda p: p["structural"]["dist_js"]),
        "mean_fes_js": mean(lambda p: p["thermodynamic"]["fes_js"]),
        "mean_relax_ratio": mean(lambda p: p["kinetic"]["relax_ratio"]),
    }


def main():
    ap = argparse.ArgumentParser(description="Physics validation baseline")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", action="append", required=True, dest="shards",
                    help="MD shard .pt (repeatable)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tau_ps", type=float, default=2000.0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--diff_steps", type=int, default=20)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--temp_K", type=float, default=300.0)
    ap.add_argument("--wca_sigma", type=float, default=4.5)
    ap.add_argument("--wca_eps", type=float, default=0.3)
    ap.add_argument("--wca_lam", type=float, default=0.05)
    ap.add_argument("--bond_constraint_iters", type=int, default=5)
    ap.add_argument("--max_update_norm", type=float, default=3.0)
    ap.add_argument("--noether", action="store_true", default=False,
                    help="Apply Noether momentum projection after each step (Mode A).")
    ap.add_argument("--n_states", type=int, default=6)
    ap.add_argument("--kT", type=float, default=1.0)
    ap.add_argument("--out", default="validation_baseline.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    settings = {
        "steps": args.steps, "tau_ps": args.tau_ps, "k": args.k,
        "diff_steps": args.diff_steps, "eta": args.eta, "temp_K": args.temp_K,
        "wca_sigma": args.wca_sigma, "wca_eps": args.wca_eps,
        "wca_lam": args.wca_lam, "bond_constraint_iters": args.bond_constraint_iters,
        "max_update_norm": args.max_update_norm, "n_states": args.n_states,
        "kT": args.kT, "noether": args.noether,
    }
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    proteins = build_report(ckpt, args.shards, settings, device)
    report = {
        "heldout": False,
        "checkpoint": args.checkpoint,
        "settings": settings,
        "proteins": proteins,
        "summary": summarize(proteins),
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
