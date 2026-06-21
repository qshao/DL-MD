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
import torch
from lsmd import geometry as g
from lsmd import transfer_eval as te


def _run(ckpt, shard, steps, tau_ps, k, diff_steps, device):
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
                      diff_steps=diff_steps, device=device)
    return te.evaluate(traj, t_md)


def main():
    ap = argparse.ArgumentParser(description="Zero-shot eval of transferable model")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", required=True, help="held-out shard .pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tau_ps", type=float, default=1000.0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--diff_steps", type=int, default=50)
    ap.add_argument("--oracle", default=None, help="per-protein checkpoint (upper bracket)")
    ap.add_argument("--lower", default=None, help="marginal-prior checkpoint (lower bracket)")
    ap.add_argument("--out", default="eval.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    shard = torch.load(args.shard, map_location="cpu")

    report = {"model": _run(torch.load(args.checkpoint, map_location="cpu"),
                            shard, args.steps, args.tau_ps, args.k,
                            args.diff_steps, device)}
    if args.oracle:
        report["oracle"] = _run(torch.load(args.oracle, map_location="cpu"),
                                shard, args.steps, args.tau_ps, args.k,
                                args.diff_steps, device)
    if args.lower:
        report["lower"] = _run(torch.load(args.lower, map_location="cpu"),
                               shard, args.steps, args.tau_ps, args.k,
                               args.diff_steps, device)

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
