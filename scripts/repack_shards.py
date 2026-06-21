"""Convert existing float32 R/t shards to compact axis-angle float16 format.

Idempotent: shards that already contain 'R_aa' are skipped.  Safe to
interrupt and re-run — each file is written atomically (temp file → rename).

Usage
-----
python scripts/repack_shards.py --dir data/atlas [--workers 4]
"""
import argparse
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from lsmd import geometry as g


def repack_file(path):
    """Convert one shard .pt in-place. Returns True if converted, False if skipped."""
    shard = torch.load(path, map_location="cpu", weights_only=False)
    if "R_aa" in shard:
        return False  # already compact

    R = shard.pop("R")                       # [F, N, 3, 3] float32
    shard["R_aa"] = g.so3_log(R).half()      # [F, N, 3]   float16
    shard["t"]    = shard["t"].half()        # [F, N, 3]   float16

    # atomic write: temp file in same dir, then rename
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(dir=dir_, suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(shard, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise
    return True


def main():
    ap = argparse.ArgumentParser(description="Repack ATLAS shards to compact float16 format")
    ap.add_argument("--dir", default="data/atlas")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel worker threads (CPU-bound so keep modest)")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.dir) if f.endswith(".pt"))
    if not files:
        print(f"No .pt files found in {args.dir}")
        return

    print(f"Repacking {len(files)} shards in {args.dir} with {args.workers} workers ...")
    converted = skipped = errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(repack_file, os.path.join(args.dir, f)): f
                   for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                changed = fut.result()
                if changed:
                    converted += 1
                else:
                    skipped += 1
            except Exception as exc:
                errors += 1
                print(f"  ERROR {fname}: {exc}", file=sys.stderr)
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  converted={converted}  "
                      f"skipped={skipped}  errors={errors}")

    print(f"\nDone: {converted} converted, {skipped} already compact, {errors} errors")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
