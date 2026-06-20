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
    if traj.unitcell_lengths is not None:
        traj.make_molecules_whole(inplace=True)   # undo PBC wrapping (protein split across box)
    ca_idx = traj.topology.select("protein and name CA")
    if len(ca_idx) > 0:
        traj.superpose(traj, 0, atom_indices=ca_idx)
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


def make_multi_lag_pairs(num_frames, taus):
    """Generate frame pairs at multiple lag times to maximise trajectory use.

    For each tau in taus, generates all valid (i, i+tau) pairs.  The result is
    sorted by start frame so time_split remains leakage-free.

    Args:
        num_frames: total number of frames
        taus: sequence of integer lag values (frames)

    Returns:
        LongTensor [P, 3] — columns: (start_frame, end_frame, tau)
    """
    segments = []
    for tau in taus:
        tau = int(tau)
        starts = torch.arange(0, num_frames - tau, dtype=torch.long)
        tau_col = torch.full((len(starts),), tau, dtype=torch.long)
        segments.append(torch.stack([starts, starts + tau, tau_col], dim=1))
    pairs = torch.cat(segments, dim=0)
    order = torch.argsort(pairs[:, 0], stable=True)  # sort by start frame
    return pairs[order]


def time_split(pairs, val_frac):
    """Split pairs into train and validation sets with time-ordered guarantee.

    Args:
        pairs: LongTensor [P, 2] or [P, 3] of frame pairs (sorted by start frame)
        val_frac: fraction of pairs for validation (0..1)

    Returns:
        (train_pairs, val_pairs) where train_pairs[:, 0].max() <= val_pairs[:, 0].min()
        This guarantees no temporal leakage.
    """
    n = pairs.shape[0]
    cut = int(n * (1 - val_frac))
    return pairs[:cut], pairs[cut:]


def compute_frame_weights(frames, n_pca=3, bins=30, density_clip=10.0):
    """Inverse-density weights for training pairs (source-frame correction).

    Projects all CA frames to a 2D PCA space, bins into a density histogram,
    and weights each frame by 1/count so rare conformations are upweighted.
    Corrects for the over-representation of dominant MD basins.

    Args:
        frames:       dict from load_frames — uses frames["t"] [F, N, 3].
        n_pca:        Number of PCA components to compute (only PC1-PC2 used).
        bins:         Grid resolution for the 2D density histogram.
        density_clip: Max weight relative to mean (prevents extreme upweighting).

    Returns:
        weights: [F] float32 tensor, mean = 1.0.
    """
    ca = frames["t"].float()          # [F, N, 3]
    F, N, _ = ca.shape
    ca_flat = ca.reshape(F, -1)       # [F, N*3]
    ca_flat = ca_flat - ca_flat.mean(0, keepdim=True)

    _, _, Vt = torch.linalg.svd(ca_flat, full_matrices=False)   # Vt: [min(F,N*3), N*3]
    n_comp = min(n_pca, Vt.shape[0])
    pc = ca_flat @ Vt[:n_comp].T      # [F, n_comp]

    # 2D histogram in PC1-PC2
    lo = pc[:, :2].min(0).values      # [2]
    hi = pc[:, :2].max(0).values      # [2]
    span = (hi - lo).clamp_min(1e-8)

    x_bin = ((pc[:, 0] - lo[0]) / span[0] * bins).long().clamp(0, bins - 1)
    y_bin = ((pc[:, 1] - lo[1]) / span[1] * bins).long().clamp(0, bins - 1)
    bin_idx = x_bin * bins + y_bin    # [F]

    counts = torch.zeros(bins * bins)
    counts.scatter_add_(0, bin_idx, torch.ones(F))

    frame_counts = counts[bin_idx].clamp_min(1.0)   # [F]
    weights = 1.0 / frame_counts
    mean_w = weights.mean()
    weights = weights.clamp(max=mean_w * density_clip)
    weights = weights / weights.mean()
    return weights
