"""Download N random ATLAS proteins, build shards, and run initial training.

Usage
-----
python scripts/download_and_train_atlas.py \
    --n 10 --out data/atlas --ckpt checkpoints/atlas_init.pt \
    --steps 500 --seed 42

The script caches shards as data/atlas/<pdb_chain>.pt so re-runs skip
already-downloaded proteins. Raw trajectory files are deleted after the
.pt shard is saved. Pass --steps for a quick smoke-test run;
increase for real training.
"""
import argparse
import os
import random
import shutil
import sys

import torch

from lsmd import atlas as atl
from lsmd import transfer_train as tt


def main():
    ap = argparse.ArgumentParser(description="Download ATLAS proteins and train")
    ap.add_argument("--n", type=int, default=10, help="Number of proteins to download")
    ap.add_argument("--out", default="data/atlas", help="Directory for shards")
    ap.add_argument("--ckpt", default="checkpoints/atlas_init.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=500,
                    help="Training gradient steps (use 500 for quick smoke-test)")
    ap.add_argument("--lags_ps", type=float, nargs="+", default=[500.0, 2000.0],
                    help="Physical lag windows in ps for training examples")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Pick N random proteins from ATLAS ---
    print("Fetching ATLAS protein list ...", flush=True)
    all_ids = atl.fetch_atlas_ids()
    print(f"  {len(all_ids)} entries available", flush=True)
    rng = random.Random(args.seed)
    chosen = rng.sample(all_ids, min(args.n, len(all_ids)))
    print(f"Selected: {chosen}\n", flush=True)

    # --- Download + build shards ---
    shards = []
    failed = []
    for pdb_chain in chosen:
        shard_path = os.path.join(args.out, f"{pdb_chain}.pt")
        if os.path.exists(shard_path):
            print(f"  {pdb_chain}: loading cached shard", flush=True)
            shards.append(torch.load(shard_path, map_location="cpu",
                                     weights_only=False))
            continue

        print(f"  {pdb_chain}: downloading ...", flush=True)
        raw_dir = os.path.join(args.out, pdb_chain)
        try:
            traj_path, top_path, dt_ps = atl.download_atlas_entry(pdb_chain, raw_dir)
            print(f"    building shard (dt={dt_ps} ps) ...", flush=True)
            shard = atl.build_shard(traj_path, top_path, dt=dt_ps)
            torch.save(shard, shard_path)
            shutil.rmtree(raw_dir)
            shards.append(shard)
            F, N = shard["R_aa"].shape[:2]
            print(f"    -> {N} residues, {F} frames, seq[:8]={shard['seq'][:8]}",
                  flush=True)
        except Exception as exc:
            print(f"  {pdb_chain}: FAILED — {exc}", file=sys.stderr, flush=True)
            failed.append(pdb_chain)

    print(f"\nDownloaded {len(shards)} shards"
          + (f", {len(failed)} failed: {failed}" if failed else "") + "\n",
          flush=True)

    if len(shards) < 2:
        sys.exit("ERROR: fewer than 2 valid shards — cannot train.")

    # --- Initial training ---
    print(f"Training on {len(shards)} proteins  "
          f"(steps={args.steps}, hidden={args.hidden}, layers={args.layers}, "
          f"device={device})", flush=True)

    ckpt = tt.train(
        shards,
        lags_ps=args.lags_ps,
        k=12,
        hidden=args.hidden,
        layers=args.layers,
        lr=1e-3,
        max_union_nodes=2000,
        accum=4,
        steps=args.steps,
        T_diff=100,
        norm_samples=min(64, len(shards) * 8),
        device=device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)), exist_ok=True)
    torch.save(ckpt, args.ckpt)
    print(f"\nCheckpoint saved -> {args.ckpt}", flush=True)
    hp = ckpt["hparams"]
    print(f"  hidden={hp['hidden']}, layers={hp['layers']}, "
          f"point_dim={hp['point_dim']}", flush=True)


if __name__ == "__main__":
    main()
