"""Train the transferable cross-protein propagator from a directory of shards.

Usage
-----
python scripts/train_transfer.py \\
    --shards_dir data/atlas --split data/atlas/split.json \\
    --lags_ps 200 1000 --steps 20000 --out checkpoints/transfer.pt
"""
import argparse
import glob
import json
import os
import torch
from lsmd import transfer_train as tt


def main():
    ap = argparse.ArgumentParser(description="Train transferable propagator")
    ap.add_argument("--shards_dir", required=True, help="dir of *.pt shards")
    ap.add_argument("--split", default=None,
                    help="split.json with a 'train' id list (optional)")
    ap.add_argument("--lags_ps", type=float, nargs="+", default=[200.0, 1000.0])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_union_nodes", type=int, default=2000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--T_diff", type=int, default=200)
    ap.add_argument("--norm_samples", type=int, default=256)
    ap.add_argument("--out", default="checkpoints/transfer.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    paths = sorted(glob.glob(os.path.join(args.shards_dir, "*.pt")))
    if args.split:
        with open(args.split) as fh:
            train_ids = set(json.load(fh)["train"])
        paths = [p for p in paths
                 if os.path.splitext(os.path.basename(p))[0] in train_ids]
    shards = [torch.load(p, map_location="cpu") for p in paths]
    print(f"Loaded {len(shards)} shards from {args.shards_dir}")

    ckpt = tt.train(shards, lags_ps=args.lags_ps, k=args.k, hidden=args.hidden,
                    layers=args.layers, lr=args.lr,
                    max_union_nodes=args.max_union_nodes, accum=args.accum,
                    steps=args.steps, T_diff=args.T_diff,
                    norm_samples=args.norm_samples, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"Checkpoint saved -> {args.out}")


if __name__ == "__main__":
    main()
