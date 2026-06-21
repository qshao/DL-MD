"""Train the transferable cross-protein propagator from one or more shard dirs.

Usage
-----
# single dataset
python scripts/train_transfer.py \\
    --shards_dir data/atlas \\
    --lags_ps 200 1000 --steps 20000 --out checkpoints/transfer.pt

# combined ATLAS + mdCATH
python scripts/train_transfer.py \\
    --shards_dir data/atlas data/mdcath \\
    --lags_ps 2000 5000 10000 --steps 20000 --out checkpoints/combined.pt
"""
import argparse
import glob
import json
import os
import time
import torch
from lsmd import transfer_train as tt


def main():
    ap = argparse.ArgumentParser(description="Train transferable propagator")
    ap.add_argument("--shards_dir", nargs="+", required=True,
                    help="One or more directories of *.pt shards")
    ap.add_argument("--split", default=None,
                    help="split.json with a 'train' id list (applied to first dir only)")
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
    ap.add_argument("--lam", type=float, default=0.0,
                    help="Max physics-penalty weight (C1 soft loss; 0=disabled)")
    ap.add_argument("--lam_warmup", type=int, default=500,
                    help="Gradient steps to ramp lam from 0 to --lam")
    ap.add_argument("--log_every", type=int, default=100,
                    help="Print loss + speed every N gradient steps")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="Gradient norm clip (0 = disabled)")
    ap.add_argument("--norm_dir", default=None,
                    help="Directory to sample for UpdateNorm (default: first --shards_dir). "
                         "Use an ATLAS-only dir to avoid high-T mdCATH scale inflation.")
    ap.add_argument("--no_frame_weighted", action="store_true",
                    help="Sample shards uniformly (default: proportional to frame count)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model for ~20%% GPU speedup (requires PyTorch 2.0+)")
    ap.add_argument("--temp_schedule", nargs="*", default=None, metavar="STEP:TEMP",
                    help="Temperature curriculum for mdCATH trajectories. "
                         "Space-separated 'step:temp_K' pairs, e.g.: "
                         "0:320 2000:348 5000:379 10000:413 15000:450  "
                         "At each listed gradient step, the max allowed mdCATH "
                         "temperature increases to temp_K. "
                         "ATLAS shards are unaffected (always included). "
                         "Default: no curriculum (all temperatures from step 0).")
    ap.add_argument("--out", default="checkpoints/transfer.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load shards from all requested directories ---
    shards = []
    dir_shard_lists = {}
    for i, shard_dir in enumerate(args.shards_dir):
        paths = sorted(glob.glob(os.path.join(shard_dir, "*.pt")))
        # split filter only applies to the first directory
        if i == 0 and args.split:
            with open(args.split) as fh:
                train_ids = set(json.load(fh)["train"])
            paths = [p for p in paths
                     if os.path.splitext(os.path.basename(p))[0] in train_ids]
        t0 = time.perf_counter()
        dir_shards = [torch.load(p, map_location="cpu", weights_only=False)
                      for p in paths]
        elapsed = time.perf_counter() - t0
        n_frames = sum(s["t"].shape[0] for s in dir_shards)
        print(f"  {shard_dir}: {len(dir_shards)} shards, "
              f"{n_frames:,} total frames  ({elapsed:.1f}s)", flush=True)
        shards.extend(dir_shards)
        dir_shard_lists[shard_dir] = dir_shards

    print(f"Total: {len(shards)} shards from {len(args.shards_dir)} dataset(s)",
          flush=True)

    # Resolve norm shards: default to first dir (usually ATLAS)
    norm_shards = None
    if args.norm_dir is not None:
        if args.norm_dir in dir_shard_lists:
            norm_shards = dir_shard_lists[args.norm_dir]
        else:
            norm_paths = sorted(glob.glob(os.path.join(args.norm_dir, "*.pt")))
            norm_shards = [torch.load(p, map_location="cpu", weights_only=False)
                           for p in norm_paths]
            print(f"  norm_dir {args.norm_dir}: {len(norm_shards)} shards for UpdateNorm",
                  flush=True)
    elif len(args.shards_dir) > 1:
        # Multiple datasets: default norm pool = first dir (most physiologically stable)
        norm_shards = dir_shard_lists[args.shards_dir[0]]
        print(f"  UpdateNorm fitted on: {args.shards_dir[0]} ({len(norm_shards)} shards)",
              flush=True)

    # Parse temperature schedule: "0:320 2000:348 ..." -> [(0,320),(2000,348),...]
    temp_schedule = None
    if args.temp_schedule:
        temp_schedule = []
        for token in args.temp_schedule:
            step_str, temp_str = token.split(":")
            temp_schedule.append((int(step_str), int(temp_str)))
        temp_schedule.sort()

    ckpt = tt.train(shards, lags_ps=args.lags_ps, k=args.k, hidden=args.hidden,
                    layers=args.layers, lr=args.lr,
                    max_union_nodes=args.max_union_nodes, accum=args.accum,
                    steps=args.steps, T_diff=args.T_diff,
                    norm_samples=args.norm_samples, device=device,
                    lam=args.lam, lam_warmup=args.lam_warmup,
                    log_every=args.log_every, grad_clip=args.grad_clip,
                    norm_shards=norm_shards,
                    frame_weighted=not args.no_frame_weighted,
                    compile_model=args.compile,
                    temp_schedule=temp_schedule)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"\nCheckpoint saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
