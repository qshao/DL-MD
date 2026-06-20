"""Reconstruct all-atom heavy-atom structures from a generated bead trajectory.

The script reads the multi-MODEL PDB file written by generate_md.py, finds the
nearest frame in the original MD trajectory for each generated frame, grafts the
generated backbone onto the template sidechains, and writes an all-atom PDB.

Usage
-----
# 4-bead trajectory
python scripts/reconstruct.py \\
    --beads       gen_4bead_tau1/trajectory.pdb \\
    --traj        WT/WT-sol6.trr \\
    --top         WT/WT-sol6.gro \\
    --checkpoint  checkpoints/wt_4bead_200ep.pt \\
    --out         gen_4bead_tau1/allatom.pdb

# CA-only trajectory (no checkpoint needed)
python scripts/reconstruct.py \\
    --beads       gen_ca_tau1/trajectory.pdb \\
    --traj        WT/WT-sol6.trr \\
    --top         WT/WT-sol6.gro \\
    --mode        ca \\
    --out         gen_ca_tau1/allatom.pdb
"""

import argparse
import os
import sys
import numpy as np
import torch
import mdtraj as md

from lsmd.reconstruct import AllAtomReconstructor


def load_bead_trajectory(pdb_path, mode):
    """Load a multi-MODEL bead PDB produced by generate_md.py.

    Returns:
        frames: [F, P, n_beads, 3] or [F, P, 3] tensor in Å
        res_names: list of P residue name strings
    """
    traj = md.load(pdb_path)
    top  = traj.topology

    # Atom names present in the bead PDB
    bead_atom_names = {
        "4bead": ["N", "CA", "C", "CB"],
        "2bead": ["CA", "CB"],
        "ca":    ["CA"],
    }[mode]

    n_beads = len(bead_atom_names)
    res_names = []
    bead_indices = []   # [P, n_beads] atom index into MDtraj trajectory

    for r in top.residues:
        ai_map = {a.name: a.index for a in r.atoms}
        indices = [ai_map.get(nm) for nm in bead_atom_names]
        if any(v is None for v in indices[:2]):   # need at least CA (index 1 or 0)
            continue
        # Glycine may lack CB — fall back to CA index for the CB slot
        if n_beads > 1 and indices[-1] is None:
            indices[-1] = ai_map.get("CA")
        if None in indices:
            continue
        bead_indices.append(indices)
        res_names.append(r.name)

    bead_idx = np.array(bead_indices, dtype=int)   # [P, n_beads]
    P = bead_idx.shape[0]
    F = traj.n_frames

    xyz_A = torch.tensor(traj.xyz * 10.0, dtype=torch.float32)   # [F, N_atoms, 3]
    # Gather bead coordinates [F, P, n_beads, 3]
    frames = xyz_A[:, bead_idx, :]    # [F, P, n_beads, 3]

    if n_beads == 1:
        frames = frames.squeeze(2)    # [F, P, 3] for CA-only

    print(f"Loaded {F} bead frames  ({P} residues, mode={mode})")
    return frames, res_names


def main():
    ap = argparse.ArgumentParser(
        description="Reconstruct all-atom structures from coarse-grained bead trajectories"
    )
    ap.add_argument("--beads",      required=True,
                    help="Bead-model trajectory PDB (from generate_md.py)")
    ap.add_argument("--traj",       required=True,
                    help="Original MD trajectory (TRR, DCD, XTC, …)")
    ap.add_argument("--top",        required=True,
                    help="Topology file for the MD trajectory (GRO, PDB, …)")
    ap.add_argument("--out",        required=True,
                    help="Output all-atom PDB path")
    ap.add_argument("--mode",       default=None,
                    choices=["4bead", "2bead", "ca"],
                    help="Bead mode (auto-detected from --checkpoint if omitted)")
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint .pt file — used to read mode and gly_mask")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Reconstruct only the first N bead frames (useful for testing)")
    args = ap.parse_args()

    # Determine mode and gly_mask
    gly_mask = None
    mode = args.mode

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        hp   = ckpt["hparams"]
        if mode is None:
            mode = hp.get("mode", "ca")
        raw_gly = ckpt.get("gly_mask")
        if raw_gly is not None:
            gly_mask = raw_gly.cpu()

    if mode is None:
        print("Error: --mode must be provided if --checkpoint is not given.", file=sys.stderr)
        sys.exit(1)

    # Load bead trajectory
    bead_frames, res_names = load_bead_trajectory(args.beads, mode)

    if args.max_frames is not None:
        bead_frames = bead_frames[:args.max_frames]

    # Build reconstructor (loads and caches the full MD trajectory)
    rec = AllAtomReconstructor(args.traj, args.top)

    # Reconstruct
    print(f"Reconstructing {bead_frames.shape[0]} frames …")
    traj_out = rec.reconstruct_trajectory(bead_frames, mode=mode, gly_mask=gly_mask)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    traj_out.save_pdb(args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved {traj_out.n_frames} frames → {args.out}  ({size_mb:.1f} MB)")
    print(f"  Heavy atoms per frame: {traj_out.n_atoms}")


if __name__ == "__main__":
    main()
