"""Convert data/wt_frames.pt from legacy KRAS-WT format to atlas-compatible shard.

Legacy wt_frames.pt layout:
  R        [F, N, 3, 3] float32  per-residue rotation matrices
  t        [F, N, 3]   float32  Cα positions (Å)
  res_type [N]         int64    local 0-18 vocabulary (protein-specific enumeration)
  chain_id [N]         int64
  res_index[N]         int64
  n_types  int                  number of distinct residue types in this protein

Atlas-compatible output layout (accepted by train_transfer.py and explore_conformations.py):
  R        [F, N, 3, 3] float32  (kept; both scripts accept full-rotation format)
  t        [F, N, 3]   float32
  res_type [N]         int64    canonical 21-type vocabulary (vocab.residue_indices)
  chain_id [N]         int64
  res_index[N]         int64
  seq      list[str]            3-letter residue names in chain order
  n_res    int
  dt       float                ps per frame (200 ps for this 2-fs/step, 100k-step output)

The canonical res_type is derived from the PDB file (WT/WT_fixed.pdb), whose CA-atom
sequence must have the same length and ordering as the trajectory frames.

Usage
-----
python scripts/prepare_kras_shard.py \\
    --wt_frames data/wt_frames.pt \\
    --pdb WT/WT_fixed.pdb \\
    --dt 200.0 \\
    --out data/kras_wt_shard.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsmd.vocab import residue_indices


def read_pdb_ca_sequence(pdb_path):
    """Return list of 3-letter residue names for each unique CA atom in PDB order."""
    seq, seen = [], set()
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            resname = line[17:20].strip()
            chain_id = line[21]
            resnum = int(line[22:26])
            key = (chain_id, resnum)
            if key not in seen:
                seen.add(key)
                seq.append(resname)
    return seq


def main():
    ap = argparse.ArgumentParser(
        description="Convert KRAS-WT legacy shard to atlas-compatible training format"
    )
    ap.add_argument("--wt_frames", default="data/wt_frames.pt",
                    help="Legacy KRAS-WT trajectory file (default: data/wt_frames.pt)")
    ap.add_argument("--pdb", default="WT/WT_fixed.pdb",
                    help="PDB with canonical residue sequence (default: WT/WT_fixed.pdb)")
    ap.add_argument("--dt", type=float, default=200.0,
                    help="ps per saved frame.  prod.mdp: dt=0.002 ps, nstxout=100000 "
                         "→ 200 ps/frame (default: 200.0)")
    ap.add_argument("--out", default="data/kras_wt_shard.pt",
                    help="Output shard path (default: data/kras_wt_shard.pt)")
    args = ap.parse_args()

    print(f"Loading {args.wt_frames} ...", flush=True)
    src = torch.load(args.wt_frames, map_location="cpu", weights_only=False)
    F, N = src["t"].shape[:2]
    print(f"  {F} frames × {N} residues  (local n_types={src.get('n_types', '?')})",
          flush=True)

    print(f"Reading sequence from {args.pdb} ...", flush=True)
    seq = read_pdb_ca_sequence(args.pdb)
    if len(seq) != N:
        raise ValueError(
            f"PDB CA count ({len(seq)}) != trajectory residue count ({N}). "
            "Check that the PDB and wt_frames.pt come from the same protein."
        )
    res_type_canon = residue_indices(seq)
    print(f"  {len(set(seq))} unique residue types  "
          f"canonical range: {res_type_canon.min().item()}–{res_type_canon.max().item()}",
          flush=True)

    shard = {
        "t":         src["t"].float(),       # [F, N, 3] float32
        "R":         src["R"].float(),       # [F, N, 3, 3] float32
        "res_type":  res_type_canon,         # [N] canonical 21-type vocab
        "chain_id":  src["chain_id"],        # [N]
        "res_index": src["res_index"],       # [N]
        "seq":       seq,                    # list[str]
        "n_res":     N,
        "dt":        float(args.dt),
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(shard, args.out)
    print(f"Saved → {args.out}  "
          f"({F} frames, dt={args.dt} ps, "
          f"total={F * args.dt / 1000:.0f} ns)",
          flush=True)


if __name__ == "__main__":
    main()
