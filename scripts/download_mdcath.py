"""Bulk download of the mdCATH dataset from HuggingFace into data/mdcath/.

Downloads 1000 domain H5 files (~700 GB total) in parallel, builds compact
float16 shards (.pt), and deletes the raw H5 after each shard is saved.
Already-present *.pt shards are skipped so the script is fully resumable.

All 5 temperatures and all 5 replicas are included per domain (25 trajectories
per shard).  Trajectory boundaries are recorded in the ``traj_breaks`` key so
that lag-pair sampling never crosses replica/temperature boundaries.

Usage
-----
# background (recommended for 700 GB):
nohup python scripts/download_mdcath.py --workers 4 \
    --out data/mdcath --log data/mdcath/download.log &

# foreground smoke-test (first 5 domains only):
python scripts/download_mdcath.py --n 5
"""
import argparse
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from lsmd import mdcath as mc


def process_one(domain, out_dir, dt_ps):
    """Download H5, build shard, save .pt, delete H5. Returns (n_res, n_frames)."""
    h5_path = mc.download_mdcath_entry(domain, out_dir)
    try:
        shard = mc.build_shard_from_h5(h5_path, dt_ps=dt_ps)
        shard_path = os.path.join(out_dir, f"{domain}.pt")
        torch.save(shard, shard_path)
    finally:
        if os.path.exists(h5_path):
            os.remove(h5_path)
    n_trajs = len(shard["traj_breaks"]) + 1
    return shard["n_res"], shard["R_aa"].shape[0], n_trajs


def main():
    ap = argparse.ArgumentParser(description="Bulk-download the mdCATH corpus from HuggingFace")
    ap.add_argument("--out", default="data/mdcath")
    ap.add_argument("--log", default=None,
                    help="Log file (default: <out>/download.log)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel download threads (H5 files are large; keep modest)")
    ap.add_argument("--retry", type=int, default=2,
                    help="Per-domain retry attempts on transient errors")
    ap.add_argument("--n", type=int, default=None,
                    help="Download only the first N domains (for smoke-testing)")
    ap.add_argument("--dt_ps", type=float, default=mc.MDCATH_DT_PS,
                    help="Picoseconds per saved frame (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for --n random selection")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = args.log or os.path.join(args.out, "download.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger()

    log.info("Fetching mdCATH domain list from HuggingFace ...")
    all_ids = mc.fetch_mdcath_ids()
    log.info(f"  {len(all_ids)} domains available")

    if args.n is not None:
        rng = random.Random(args.seed)
        all_ids = rng.sample(all_ids, min(args.n, len(all_ids)))
        log.info(f"  Sampling {len(all_ids)} domains (--n {args.n})")

    todo = [d for d in all_ids
            if not os.path.exists(os.path.join(args.out, f"{d}.pt"))]
    already = len(all_ids) - len(todo)
    log.info(f"  {already} already cached, {len(todo)} to download")

    if not todo:
        log.info("Nothing to do.")
        return

    ok = already
    fail = 0
    start = time.time()

    def worker(domain):
        for attempt in range(1, args.retry + 2):
            try:
                n_res, n_frames, n_trajs = process_one(domain, args.out, args.dt_ps)
                return domain, True, n_res, n_frames, n_trajs, None
            except Exception as exc:
                if attempt <= args.retry:
                    time.sleep(10 * (2 ** (attempt - 1)) + random.uniform(0, 5))
                else:
                    return domain, False, 0, 0, 0, str(exc)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, d): d for d in todo}
        for fut in as_completed(futures):
            domain, success, n_res, n_frames, n_trajs, err = fut.result()
            elapsed = time.time() - start
            if success:
                ok += 1
                session_done = ok + fail - already
                rate = session_done / elapsed * 60 if elapsed > 0 else 0
                eta_min = (len(todo) - session_done) / rate if rate > 0 else 0
                log.info(
                    f"OK  {domain:12s}  {n_res:4d} res  {n_frames:6d} frames"
                    f"  {n_trajs:2d} trajs"
                    f"  [{ok}/{len(all_ids)}]  {rate:.1f}/min  ETA {eta_min:.0f}min"
                )
            else:
                fail += 1
                log.warning(f"FAIL {domain:12s}  {err}")

    elapsed_h = (time.time() - start) / 3600
    log.info(
        f"\nDone in {elapsed_h:.2f}h — "
        f"{ok} ok, {fail} failed / {len(all_ids)} total"
    )
    if fail:
        log.info("Re-run the script to retry failed entries.")


if __name__ == "__main__":
    main()
