"""Add traj_temps to existing mdCATH shards via structure inference.

Existing shards were built with the standard MDCATH_TEMPS × MDCATH_REPS
ordering, so trajectory i belongs to temperature MDCATH_TEMPS[i // N_REPS].
This script adds the traj_temps tensor to each shard that lacks it so that
the temperature curriculum in transfer_train.py can filter by temperature.

Usage
-----
python scripts/repack_mdcath_temps.py --shards_dir data/mdcath
"""
import argparse
import glob
import os
import tempfile

import torch

from lsmd.mdcath import infer_traj_temps


def repack_file(path):
    shard = torch.load(path, map_location="cpu", weights_only=False)
    if "traj_temps" in shard:
        return False  # already has temperature metadata
    if "traj_breaks" not in shard:
        return False  # ATLAS or other non-mdCATH shard; skip

    n_trajs = int(shard["traj_breaks"].shape[0]) + 1
    shard["traj_temps"] = infer_traj_temps(n_trajs)

    # Atomic write: temp file in same directory, then os.replace
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".pt")
    os.close(fd)
    try:
        torch.save(shard, tmp)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Add traj_temps metadata to existing mdCATH shards")
    ap.add_argument("--shards_dir", default="data/mdcath",
                    help="Directory of *.pt mdCATH shards")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.shards_dir, "*.pt")))
    print(f"Found {len(paths)} shards in {args.shards_dir}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    updated = skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(repack_file, p): p for p in paths}
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            if result:
                updated += 1
            else:
                skipped += 1
            if i % 100 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)}  updated={updated}  skipped={skipped}",
                      flush=True)

    print(f"\nDone: {updated} updated, {skipped} already had traj_temps or skipped.")


if __name__ == "__main__":
    main()
