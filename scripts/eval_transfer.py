"""Zero-shot evaluation of a transferable checkpoint on a held-out shard.

Rolls the model out from the held-out reference and scores it against the
shard's own MD frames. If --oracle / --lower checkpoints are given, the same
held-out protein is scored under those models too, bracketing the result.

Usage
-----
python scripts/eval_transfer.py \\
    --checkpoint checkpoints/transfer.pt --shard data/atlas/1abc.pt \\
    --steps 200 --tau_ps 1000 --out eval_1abc.json
"""
import argparse
import json
import sys
import torch
from lsmd import geometry as g  # for so3_exp on compact R_aa shards
from lsmd import transfer_eval as te


def _run(ckpt, shard, steps, tau_ps, k, diff_steps, eta, temp_K,
         bond_constraint_iters, max_update_norm,
         wca_sigma, wca_eps, wca_lam, graph_rebuild_interval, device):
    net, sched, norm = te.load_checkpoint(ckpt, device=device)
    k_eff = ckpt["hparams"].get("k", k)
    # support both compact (R_aa float16) and legacy (R float32) shard formats
    if "R_aa" in shard:
        R0 = g.so3_exp(shard["R_aa"][0].float())
    else:
        R0 = shard["R"][0]
    t0 = shard["t"][0].float()
    t_md = shard["t"].float()
    traj = te.rollout(net, sched, norm, R0, t0,
                      shard["res_type"], shard["chain_id"], shard["res_index"],
                      steps=steps, tau_ps=tau_ps, k=k_eff,
                      diff_steps=diff_steps, eta=eta, temp_K=temp_K,
                      bond_constraint_iters=bond_constraint_iters,
                      max_update_norm=max_update_norm,
                      wca_sigma=wca_sigma, wca_eps=wca_eps, wca_lam=wca_lam,
                      graph_rebuild_interval=graph_rebuild_interval,
                      device=device)
    return te.evaluate(traj, t_md)


def main():
    ap = argparse.ArgumentParser(description="Zero-shot eval of transferable model")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", required=True, help="held-out shard .pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tau_ps", type=float, default=1000.0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--ddim", action="store_true",
                    help="Fast deterministic DDIM sampling: sets eta=0.0, diff_steps=10. "
                         "Override with explicit --eta / --diff_steps.")
    ap.add_argument("--diff_steps", type=int, default=None,
                    help="Denoising steps (default 50 DDPM; 10 recommended with --ddim/--eta 0)")
    ap.add_argument("--eta", type=float, default=None,
                    help="Reverse-process stochasticity: 1.0=DDPM (default), 0.0=DDIM")
    ap.add_argument("--temp_K", type=float, default=300.0,
                    help="Simulation temperature in Kelvin (default 300; used when model has temp_emb_dim > 0)")
    ap.add_argument("--max_update_norm", type=float, default=3.0,
                    help="Clip per-residue normalized update norm before de-normalization "
                         "(default 3.0). Prevents rotation drift explosion.")
    ap.add_argument("--wca_sigma", type=float, default=4.5,
                    help="WCA CA–CA diameter (Å, default 4.5). Set 0 to disable guidance.")
    ap.add_argument("--wca_eps", type=float, default=0.3,
                    help="WCA well depth (kcal/mol, default 0.3 ≈ 0.5 kT at 300 K).")
    ap.add_argument("--wca_lam", type=float, default=0.05,
                    help="WCA guidance step size in normalized update space (default 0.05).")
    ap.add_argument("--graph_rebuild_interval", type=int, default=1,
                    help="Rebuild kNN graph topology every N rollout steps (default 1 = every step).")
    ap.add_argument("--bond_constraint_iters", type=int, default=5,
                    help="SHAKE iterations enforcing CA–CA pseudo-bond lengths after each "
                         "diffusion step (default 5). Set 0 to disable. Prevents systematic "
                         "bond-length expansion that causes autoregressive explosion.")
    ap.add_argument("--oracle", default=None, help="per-protein checkpoint (upper bracket)")
    ap.add_argument("--lower", default=None, help="marginal-prior checkpoint (lower bracket)")
    ap.add_argument("--out", default="eval.json")
    ap.add_argument("--device", default=None)
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    shard = torch.load(args.shard, map_location="cpu")

    run_kwargs = dict(
        steps=args.steps, tau_ps=args.tau_ps, k=args.k,
        diff_steps=args.diff_steps, eta=args.eta, temp_K=args.temp_K,
        bond_constraint_iters=args.bond_constraint_iters,
        max_update_norm=args.max_update_norm,
        wca_sigma=args.wca_sigma, wca_eps=args.wca_eps, wca_lam=args.wca_lam,
        graph_rebuild_interval=args.graph_rebuild_interval,
        device=device,
    )
    report = {"model": _run(torch.load(args.checkpoint, map_location="cpu"),
                            shard, **run_kwargs)}
    if args.oracle:
        report["oracle"] = _run(torch.load(args.oracle, map_location="cpu"),
                                shard, **run_kwargs)
    if args.lower:
        report["lower"] = _run(torch.load(args.lower, map_location="cpu"),
                               shard, **run_kwargs)

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
