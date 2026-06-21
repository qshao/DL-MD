"""Bulk download of the full ATLAS repository into data/atlas/.

Parallelises across `--workers` threads (I/O bound). Already-present
*.pt shards are skipped so the script is fully resumable. Raw trajectory
files are deleted after the .pt shard is saved.

Progress is written to --log (default data/atlas/download.log) with one line
per protein: timestamp, pdb_chain, status (ok/fail), n_res, n_frames, error.

Usage
-----
# background:
nohup python scripts/download_atlas_full.py --workers 8 \
    --out data/atlas --log data/atlas/download.log &

# or just foreground:
python scripts/download_atlas_full.py --workers 8
"""
import argparse
import logging
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from lsmd import atlas as atl


def download_one(pdb_chain, out_dir):
    """Download, build shard, save .pt, remove raw files. Returns (n_res, n_frames)."""
    raw_dir = os.path.join(out_dir, pdb_chain)
    shard_path = os.path.join(out_dir, f"{pdb_chain}.pt")
    traj_path, top_path, dt_ps = atl.download_atlas_entry(pdb_chain, raw_dir)
    shard = atl.build_shard(traj_path, top_path, dt=dt_ps)
    torch.save(shard, shard_path)
    shutil.rmtree(raw_dir)
    return shard["n_res"], shard["R_aa"].shape[0]


def main():
    ap = argparse.ArgumentParser(description="Bulk-download the full ATLAS corpus")
    ap.add_argument("--out", default="data/atlas")
    ap.add_argument("--log", default=None,
                    help="Log file path (default: <out>/download.log)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel download threads")
    ap.add_argument("--retry", type=int, default=2,
                    help="Per-protein retry attempts on transient errors")
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

    log.info("Fetching ATLAS protein list ...")
    all_ids = atl.fetch_atlas_ids()
    log.info(f"  {len(all_ids)} entries total")

    # Filter out already-downloaded proteins
    todo = [p for p in all_ids
            if not os.path.exists(os.path.join(args.out, f"{p}.pt"))]
    already = len(all_ids) - len(todo)
    log.info(f"  {already} already cached, {len(todo)} to download")

    if not todo:
        log.info("Nothing to do.")
        return

    ok = already
    fail = 0
    start = time.time()

    def worker(pdb_chain):
        for attempt in range(1, args.retry + 2):
            try:
                n_res, n_frames = download_one(pdb_chain, args.out)
                return pdb_chain, True, n_res, n_frames, None
            except Exception as exc:
                if attempt <= args.retry:
                    # exponential backoff + jitter to avoid synchronized retries
                    time.sleep(5 * (2 ** (attempt - 1)) + random.uniform(0, 3))
                else:
                    return pdb_chain, False, 0, 0, str(exc)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, p): p for p in todo}
        for fut in as_completed(futures):
            pdb_chain, success, n_res, n_frames, err = fut.result()
            elapsed = time.time() - start
            if success:
                ok += 1
                session_done = ok + fail - already
                rate = session_done / elapsed * 60 if elapsed > 0 else 0
                eta_min = (len(todo) - session_done) / rate if rate > 0 else 0
                log.info(
                    f"OK  {pdb_chain:12s}  {n_res:4d} res  {n_frames:5d} frames"
                    f"  [{ok}/{len(all_ids)}]  {rate:.1f}/min  ETA {eta_min:.0f}min"
                )
            else:
                fail += 1
                log.warning(f"FAIL {pdb_chain:12s}  {err}")

    elapsed_h = (time.time() - start) / 3600
    log.info(
        f"\nDone in {elapsed_h:.2f}h — "
        f"{ok} ok, {fail} failed / {len(all_ids)} total"
    )
    if fail:
        log.info("Re-run the script to retry failed entries.")


if __name__ == "__main__":
    main()
