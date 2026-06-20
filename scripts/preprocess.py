"""Preprocess a GROMACS (or any MDtraj-readable) trajectory into a CA point
cloud saved as a PyTorch .pt file, ready for training.

Usage
-----
python scripts/preprocess.py \\
    --traj WT/WT-sol6.trr \\
    --top  WT/WT-sol6.gro \\
    --out  data/wt_frames.pt

The saved file is a dict with the same keys returned by lsmd.data.load_frames:
    t         [F, P, 3]  float32  CA coordinates (Å), PBC-fixed and superposed
    R         [F, P, 3, 3]       SE(3) rotation matrices (retained for compatibility)
    res_type  [P]        long     residue type index (0 … n_types-1)
    chain_id  [P]        long     chain index
    res_index [P]        long     sequential residue index
    n_types   int                 number of unique residue types
"""
import argparse
import os
import torch
from lsmd import data


def main():
    ap = argparse.ArgumentParser(description="Preprocess MD trajectory to bead point cloud")
    ap.add_argument("--traj",  required=True, help="Trajectory file (TRR, DCD, XTC, …)")
    ap.add_argument("--top",   required=True, help="Topology file (GRO, PDB, …)")
    ap.add_argument("--out",   default="data/frames.pt", help="Output .pt file path")
    ap.add_argument("--atoms", choices=["ca", "2bead", "4bead"], default="4bead",
                    help="Bead representation: 'ca' (1 bead), '2bead' (CA+CB), '4bead' (N,CA,C,CB)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"Loading  {args.traj}")
    print(f"Topology {args.top}")
    print(f"Mode     {args.atoms}")

    if args.atoms == "4bead":
        frames = data.load_frames_4bead(args.traj, args.top)
        F, P = frames["t"].shape[:2]
        n_gly = int(frames["gly_mask"].sum())
        print(f"Frames: {F}   Residues: {P}   Gly (no CB): {n_gly}   "
              f"Residue types: {frames['n_types']}")
        ca = frames["t"][:, :, 1, :]   # CA at index 1 for 4-bead
    elif args.atoms == "2bead":
        frames = data.load_frames_2bead(args.traj, args.top)
        F, P = frames["t"].shape[:2]
        n_gly = int(frames["gly_mask"].sum())
        print(f"Frames: {F}   Residues: {P}   Gly (no CB): {n_gly}   "
              f"Residue types: {frames['n_types']}")
        ca = frames["t"][:, :, 0, :]   # CA at index 0 for 2-bead
    else:
        frames = data.load_frames(args.traj, args.top)
        F, P = frames["t"].shape[:2]
        print(f"Frames: {F}   CA atoms (residues): {P}   Residue types: {frames['n_types']}")
        ca = frames["t"]

    print(f"CA coordinate range  min={ca.min():.2f} Å  max={ca.max():.2f} Å")

    torch.save(frames, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved → {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
