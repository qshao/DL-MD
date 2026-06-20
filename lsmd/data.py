import numpy as np
import mdtraj as md
import torch
from lsmd import geometry as g


def load_frames(traj_path, top_path):
    """Load trajectory and extract backbone frames.

    Args:
        traj_path: path to trajectory file
        top_path: path to topology file

    Returns:
        dict with keys:
            - R [F, N, 3, 3]: rotation matrices (frames)
            - t [F, N, 3]: translation vectors (CA positions)
            - res_type [N]: residue type indices (0..n_types-1), dtype long
            - chain_id [N]: chain indices (0-based), dtype long
            - res_index [N]: sequential residue indices (0-based), dtype long
            - n_types: number of unique residue types
    """
    traj = md.load(traj_path, top=top_path)
    top = traj.topology
    residues = [r for r in top.residues if r.name != "HOH"]

    def atom_index(res, name):
        for a in res.atoms:
            if a.name == name:
                return a.index
        return None

    keep, res_names, chain_ids = [], [], []
    for r in residues:
        idx = {nm: atom_index(r, nm) for nm in ("N", "CA", "C")}
        if any(v is None for v in idx.values()):
            continue  # skip residues lacking backbone (e.g. caps)
        keep.append(idx)
        res_names.append(r.name)
        chain_ids.append(r.chain.index)

    xyz = torch.tensor(traj.xyz, dtype=torch.float32) * 10.0  # nm -> Angstrom
    N = xyz[:, [k["N"] for k in keep], :]
    CA = xyz[:, [k["CA"] for k in keep], :]
    C = xyz[:, [k["C"] for k in keep], :]
    R, t = g.build_frames(N, CA, C)

    uniq = sorted(set(res_names))
    type_map = {nm: i for i, nm in enumerate(uniq)}
    res_type = torch.tensor([type_map[nm] for nm in res_names], dtype=torch.long)
    chain_id = torch.tensor(chain_ids, dtype=torch.long)
    res_index = torch.arange(len(keep), dtype=torch.long)

    return {"R": R, "t": t, "res_type": res_type, "chain_id": chain_id,
            "res_index": res_index, "n_types": len(uniq)}


def make_pairs(num_frames, tau):
    """Generate frame pairs with fixed time gap tau.

    Args:
        num_frames: total number of frames
        tau: time gap between paired frames

    Returns:
        LongTensor [P, 2] where P = num_frames - tau
            - pairs[:, 0] are starting frames (0..num_frames-tau-1)
            - pairs[:, 1] are ending frames (tau..num_frames-1)
    """
    starts = torch.arange(0, num_frames - tau, dtype=torch.long)
    return torch.stack([starts, starts + tau], dim=1)


def time_split(pairs, val_frac):
    """Split pairs into train and validation sets with time-ordered guarantee.

    Args:
        pairs: LongTensor [P, 2] of frame pairs
        val_frac: fraction of pairs for validation (0..1)

    Returns:
        (train_pairs, val_pairs) where train_pairs[:, 0].max() < val_pairs[:, 0].min()
        This guarantees no temporal leakage.
    """
    n = pairs.shape[0]
    cut = int(n * (1 - val_frac))
    return pairs[:cut], pairs[cut:]
