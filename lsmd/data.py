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
            "res_index": res_index, "n_types": len(uniq), "mode": "ca",
            "res_names": res_names}


def load_frames_2bead(traj_path, top_path):
    """Load trajectory and extract CA + CB per residue (2-bead model).

    For Glycine (no CB), CA is used as a CB placeholder and ``gly_mask``
    records those residues so downstream code can skip the CA-CB bond.

    Returns:
        dict with keys:
            t        [F, P, 2, 3] float32 — bead coords in Å, order (CA, CB)
            res_type [P] long
            chain_id [P] long
            res_index[P] long
            n_types  int
            gly_mask [P] bool  — True for Glycine residues
            mode     str       — "2bead"
    """
    traj = md.load(traj_path, top=top_path)
    if traj.unitcell_lengths is not None:
        traj.make_molecules_whole(inplace=True)
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

    bead_indices = []
    res_names, chain_ids, is_gly = [], [], []
    for r in residues:
        ca_i = atom_index(r, "CA")
        if ca_i is None:
            continue
        cb_idx = atom_index(r, "CB")
        gly = cb_idx is None
        if gly:
            cb_idx = ca_i
        bead_indices.append((ca_i, cb_idx))
        res_names.append(r.name)
        chain_ids.append(r.chain.index)
        is_gly.append(gly)

    xyz    = np.array(traj.xyz) * 10.0
    idx    = np.array(bead_indices, dtype=int)          # [P, 2]
    coords = torch.tensor(xyz[:, idx, :], dtype=torch.float32)  # [F, P, 2, 3]

    uniq     = sorted(set(res_names))
    type_map = {nm: i for i, nm in enumerate(uniq)}
    res_type  = torch.tensor([type_map[nm] for nm in res_names], dtype=torch.long)
    chain_id  = torch.tensor(chain_ids, dtype=torch.long)
    res_index = torch.arange(len(bead_indices), dtype=torch.long)
    gly_mask  = torch.tensor(is_gly, dtype=torch.bool)

    return {"t": coords, "res_type": res_type, "chain_id": chain_id,
            "res_index": res_index, "n_types": len(uniq),
            "gly_mask": gly_mask, "mode": "2bead"}


def load_frames_4bead(traj_path, top_path):
    """Load trajectory and extract N, CA, C, CB per residue.

    For Glycine (no CB atom), the CB position is taken from CA so the tensor
    shape stays [F, P, 4, 3] throughout.  A boolean ``gly_mask`` [P] records
    which residues are Glycine so downstream code can skip the CA-CB bond.

    Args:
        traj_path: path to trajectory file
        top_path:  path to topology file

    Returns:
        dict with keys:
            t        [F, P, 4, 3] float32 — bead coords in Å, order (N,CA,C,CB)
            res_type [P] long      — residue type index 0..n_types-1
            chain_id [P] long      — chain index
            res_index[P] long      — sequential residue index
            n_types  int           — number of unique residue types
            gly_mask [P] bool      — True for Glycine residues (no real CB)
            mode     str           — "4bead"
    """
    traj = md.load(traj_path, top=top_path)
    if traj.unitcell_lengths is not None:
        traj.make_molecules_whole(inplace=True)
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

    bead_indices = []   # (N_idx, CA_idx, C_idx, CB_idx) per residue
    res_names, chain_ids, is_gly = [], [], []
    for r in residues:
        n_idx  = atom_index(r, "N")
        ca_i   = atom_index(r, "CA")
        c_idx  = atom_index(r, "C")
        if any(v is None for v in (n_idx, ca_i, c_idx)):
            continue
        cb_idx = atom_index(r, "CB")
        gly    = cb_idx is None
        if gly:
            cb_idx = ca_i   # placeholder: CB = CA for Gly
        bead_indices.append((n_idx, ca_i, c_idx, cb_idx))
        res_names.append(r.name)
        chain_ids.append(r.chain.index)
        is_gly.append(gly)

    xyz  = np.array(traj.xyz) * 10.0          # nm → Å, [F, N_atoms, 3]
    idx  = np.array(bead_indices, dtype=int)   # [P, 4]
    coords = torch.tensor(xyz[:, idx, :], dtype=torch.float32)  # [F, P, 4, 3]

    uniq    = sorted(set(res_names))
    type_map = {nm: i for i, nm in enumerate(uniq)}
    res_type = torch.tensor([type_map[nm] for nm in res_names], dtype=torch.long)
    chain_id = torch.tensor(chain_ids, dtype=torch.long)
    res_index = torch.arange(len(bead_indices), dtype=torch.long)
    gly_mask  = torch.tensor(is_gly, dtype=torch.bool)

    return {"t": coords, "res_type": res_type, "chain_id": chain_id,
            "res_index": res_index, "n_types": len(uniq),
            "gly_mask": gly_mask, "mode": "4bead"}


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
    ca = frames["t"].float()
    if ca.ndim == 4:
        mode = frames.get("mode", "4bead")
        ca_idx = 0 if mode == "2bead" else 1   # 2bead: CA=0; 4bead: CA=1
        ca = ca[:, :, ca_idx, :]
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


from lsmd import featurize as _feat
from lsmd import geometry as _g


def _frame_R(frames, i):
    """Return rotation matrix [N,3,3] for frame i, supporting both shard formats."""
    if "R_aa" in frames:
        return _g.so3_exp(frames["R_aa"][i].float())
    return frames["R"][i]


def _frame_t(frames, i):
    """Return CA positions [N,3] float32 for frame i."""
    return frames["t"][i].float()


def physical_lag_pairs(num_frames, dt, lags_ps, traj_breaks=None):
    """Frame pairs at physical lag times (picoseconds).

    Args:
        num_frames:   total frames in the shard.
        dt:           ps per frame.
        lags_ps:      iterable of physical lags in ps.
        traj_breaks:  optional LongTensor of frame indices where new
                      trajectories begin (e.g. [440, 880, ...] for a shard
                      that concatenates three trajectories of 440 frames each).
                      Pairs that would cross a boundary are excluded.
                      None (default) treats the entire shard as one trajectory.

    Returns:
        LongTensor [P, 3] — columns (start_frame, end_frame, tau_frames).
        Lags requiring >= segment_length frames for a given segment are skipped
        for that segment only.
    """
    if traj_breaks is None or len(traj_breaks) == 0:
        segments = [(0, num_frames)]
    else:
        walls = [0] + traj_breaks.tolist() + [num_frames]
        segments = list(zip(walls[:-1], walls[1:]))

    parts = []
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        for lag in lags_ps:
            tau_frames = max(1, int(round(float(lag) / dt)))
            if tau_frames >= seg_len:
                continue
            starts = torch.arange(seg_start, seg_end - tau_frames, dtype=torch.long)
            tau_col = torch.full((len(starts),), tau_frames, dtype=torch.long)
            parts.append(torch.stack([starts, starts + tau_frames, tau_col], dim=1))
    if not parts:
        return torch.zeros((0, 3), dtype=torch.long)
    return torch.cat(parts, dim=0)


def build_training_example(frames, i, tau_frames, k):
    """State-conditional training example from frame i to frame i+tau_frames.

    The graph is built from the CURRENT frame i; the target is the per-residue
    SE(3) update from frame i to frame i+tau_frames.

    Args:
        frames: dict with R [F,N,3,3], t [F,N,3], res_type [N], chain_id [N],
                res_index [N], dt (ps/frame).
        i:          source frame index.
        tau_frames: lag in frames.
        k:          kNN neighbours.

    Returns:
        dict with node_feats [N,24], edge_index [2,E], edge_feats [E,13],
        u_target [N,6], tau (float, ps) — consumable by union_collate.
    """
    j = i + tau_frames
    R_i, t_i = _frame_R(frames, i), _frame_t(frames, i)
    R_j, t_j = _frame_R(frames, j), _frame_t(frames, j)
    edge_index, edge_feats = _feat.frame_graph(R_i, t_i, k)
    node_feats = _feat.frame_node_features(
        frames["res_type"], frames["chain_id"], frames["res_index"])
    u_target = _feat.relative_update(R_i, t_i, R_j, t_j)
    if not u_target.isfinite().all():
        return None  # degenerate backbone frame; caller should resample
    return {
        "node_feats": node_feats,
        "edge_index": edge_index,
        "edge_feats": edge_feats,
        "u_target": u_target,
        "tau": float(tau_frames) * float(frames["dt"]),
        "R_cur": R_i,
        "t_cur": t_i,
        "chain_id": frames["chain_id"],
    }
