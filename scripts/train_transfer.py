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
    ap.add_argument("--shards_dir", nargs="+", default=None,
                    help="One or more directories of *.pt shards")
    ap.add_argument("--shard", action="append", default=None, dest="extra_shards",
                    metavar="PATH",
                    help="Individual .pt shard file(s). Repeatable. "
                         "Use instead of --shards_dir for per-protein fine-tuning.")
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
    ap.add_argument("--lam_fdt", type=float, default=0.0,
                    help="Max FDT step-variance loss weight (0 = disabled).")
    ap.add_argument("--phys_warmup", type=int, default=500,
                    help="Gradient steps to ramp lam_fdt from 0.")
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
    ap.add_argument("--temp_emb_dim", type=int, default=8,
                    help="Temperature embedding size for PropagatorNet (0 to disable). "
                         "When > 0, the model is conditioned on simulation temperature "
                         "so it predicts correctly-scaled fluctuations at each T. "
                         "Default 8. Old checkpoints without temp conditioning used 0.")
    ap.add_argument("--time_reversal", action="store_true",
                    help="Enable time-reversal augmentation (reverse_prob=0.5). "
                         "Doubles effective training data via microscopic reversibility: "
                         "the model also learns backward transitions x_{t+τ}→x_t.")
    ap.add_argument("--temp_schedule", nargs="*", default=None, metavar="STEP:TEMP",
                    help="Temperature curriculum for mdCATH trajectories. "
                         "Space-separated 'step:temp_K' pairs, e.g.: "
                         "0:320 2000:348 5000:379 10000:413 15000:450  "
                         "At each listed gradient step, the max allowed mdCATH "
                         "temperature increases to temp_K. "
                         "ATLAS shards are unaffected (always included). "
                         "Default: no curriculum (all temperatures from step 0).")
    ap.add_argument("--resume", default=None,
                    help="Path to a checkpoint .pt file to resume from. "
                         "Model weights and optimizer state are loaded; "
                         "--steps then means additional steps beyond the checkpoint.")
    ap.add_argument("--out", default="checkpoints/transfer.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not args.shards_dir and not args.extra_shards:
        ap.error("At least one of --shards_dir or --shard must be provided.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load shards ---
    shards = []
    dir_shard_lists = {}

    if args.shards_dir:
        for i, shard_dir in enumerate(args.shards_dir):
            paths = sorted(glob.glob(os.path.join(shard_dir, "*.pt")))
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

    if args.extra_shards:
        t0 = time.perf_counter()
        extra = [torch.load(p, map_location="cpu", weights_only=False)
                 for p in args.extra_shards]
        elapsed = time.perf_counter() - t0
        n_frames = sum(s["t"].shape[0] for s in extra)
        print(f"  --shard: {len(extra)} file(s), "
              f"{n_frames:,} total frames  ({elapsed:.1f}s)", flush=True)
        shards.extend(extra)

    n_sources = len(args.shards_dir or []) + (1 if args.extra_shards else 0)
    print(f"Total: {len(shards)} shards from {n_sources} source(s)", flush=True)

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
    elif args.shards_dir and len(args.shards_dir) > 1:
        norm_shards = dir_shard_lists[args.shards_dir[0]]
        print(f"  UpdateNorm fitted on: {args.shards_dir[0]} ({len(norm_shards)} shards)",
              flush=True)
    # If only --shard provided, norm_shards stays None → UpdateNorm uses all shards

    # Parse temperature schedule: "0:320 2000:348 ..." -> [(0,320),(2000,348),...]
    temp_schedule = None
    if args.temp_schedule:
        temp_schedule = []
        for token in args.temp_schedule:
            step_str, temp_str = token.split(":")
            temp_schedule.append((int(step_str), int(temp_str)))
        temp_schedule.sort()

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        resumed_step = resume_ckpt.get("step",
                       resume_ckpt.get("hparams", {}).get("steps", "?"))
        print(f"  Resuming from {args.resume} (step {resumed_step})", flush=True)

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
                    temp_schedule=temp_schedule,
                    temp_emb_dim=args.temp_emb_dim,
                    reverse_prob=0.5 if args.time_reversal else 0.0,
                    resume_from=resume_ckpt,
                    lam_fdt=args.lam_fdt, phys_warmup=args.phys_warmup)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"\nCheckpoint saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
