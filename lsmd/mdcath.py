"""mdCATH dataset loader: H5 → compact shard.

Each H5 file holds one CATH domain with 5 temperatures × 5 replicas = 25
all-atom trajectories.  ``build_shard_from_h5`` extracts backbone frames
(N / CA / C) for every (temp, rep) pair, converts to axis-angle float16,
and concatenates all trajectories into one shard.  A ``traj_breaks`` tensor
marks where trajectory boundaries fall so downstream lag-pair sampling stays
within each trajectory.

References
----------
Mirarchi et al., *Sci. Data* 11, 1299 (2024). https://doi.org/10.1038/s41597-024-04140-z
HuggingFace: compsciencelab/mdCATH (CC-BY 4.0)
"""
import io
import json
import os
import shutil
import tempfile
import urllib.request

import h5py
import numpy as np
import mdtraj as md
import torch

from lsmd import geometry as g
from lsmd import vocab

# Estimated dt: ~500 ns simulation / ~500 saved frames ≈ 1 ns per frame.
# Not stored in H5; override with dt_ps if you know the exact value.
MDCATH_DT_PS = 1000.0

MDCATH_TEMPS = ["320", "348", "379", "413", "450"]
MDCATH_REPS  = ["0", "1", "2", "3", "4"]

_HF_TREE_URL = (
    "https://huggingface.co/api/datasets/compsciencelab/mdCATH"
    "/tree/main/data"
)
_HF_FILE_URL = (
    "https://huggingface.co/datasets/compsciencelab/mdCATH"
    "/resolve/main/data/mdcath_dataset_{domain}.h5"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _backbone_indices(top):
    """Per-residue (N, CA, C) atom indices from an MDTraj topology.

    Mirrors the selection logic in data.load_frames: skip HOH and any residue
    lacking a complete backbone (N-/C-terminal caps, HETATM, etc.).

    Returns (N_idx, CA_idx, C_idx, res_names, chain_ids) as plain lists.
    """
    N_idx, CA_idx, C_idx = [], [], []
    res_names, chain_ids = [], []
    for r in top.residues:
        if r.name == "HOH":
            continue
        atom_map = {}
        for a in r.atoms:
            if a.name in ("N", "CA", "C"):
                atom_map[a.name] = a.index
        if len(atom_map) < 3:
            continue
        N_idx.append(atom_map["N"])
        CA_idx.append(atom_map["CA"])
        C_idx.append(atom_map["C"])
        res_names.append(r.name)
        chain_ids.append(r.chain.index)
    return N_idx, CA_idx, C_idx, res_names, chain_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_mdcath_ids():
    """Return list of all domain IDs available in the HuggingFace mdCATH dataset.

    Each ID is a CATH domain string, e.g. '1a02F00'.
    The HuggingFace data/ directory currently holds 1000 domains.
    """
    with urllib.request.urlopen(_HF_TREE_URL, timeout=60) as resp:
        entries = json.loads(resp.read())
    return [
        e["path"].removeprefix("data/mdcath_dataset_").removesuffix(".h5")
        for e in entries
        if e.get("type") == "file" and e["path"].endswith(".h5")
    ]


def download_mdcath_entry(domain, dest_dir):
    """Download the H5 file for one mdCATH domain into dest_dir.

    Returns the local H5 path.  Files are typically 300 MB – 1.8 GB;
    the request timeout is set to 10 minutes for slow connections.
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = _HF_FILE_URL.format(domain=domain)
    h5_path = os.path.join(dest_dir, f"mdcath_dataset_{domain}.h5")
    with urllib.request.urlopen(url, timeout=600) as resp, \
            open(h5_path, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    return h5_path


def build_shard_from_h5(h5_path, dt_ps=MDCATH_DT_PS,
                         temps=None, reps=None):
    """Build a compact shard from a mdCATH H5 file.

    Reads backbone (N / CA / C) atom coordinates for every requested
    (temperature, replica) trajectory, builds SE(3) backbone frames, and
    concatenates the result into a single shard tensor.  The ``traj_breaks``
    key records the starting frame index of each trajectory after the first
    so that lag-pair sampling never crosses trajectory boundaries.

    Args:
        h5_path: path to an mdcath_dataset_<domain>.h5 file.
        dt_ps:   picoseconds per saved frame (default 1000 ps ≈ 1 ns).
        temps:   list of temperature strings to include (default: all 5).
        reps:    list of replica strings to include (default: all 5).

    Returns:
        dict with keys matching ATLAS shard format:
            R_aa       [F, N, 3]  float16  axis-angle rotation
            t          [F, N, 3]  float16  CA positions (Å)
            res_type   [N]        long     fixed-vocab residue indices
            chain_id   [N]        long
            res_index  [N]        long
            dt         float      ps per frame
            seq        list[str]  residue names
            n_res      int
            traj_breaks [K]       long     start frames of trajectories 1..K
                                           (empty if only one trajectory loaded)
    """
    if temps is None:
        temps = MDCATH_TEMPS
    if reps is None:
        reps = MDCATH_REPS

    with h5py.File(h5_path, "r") as hf:
        domain = next(iter(hf.keys()))
        grp = hf[domain]

        # Build MDTraj topology from the embedded PDB string
        pdb_raw = grp["pdb"][()]
        pdb_str = pdb_raw.decode() if isinstance(pdb_raw, bytes) else str(pdb_raw)
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
            tmp.write(pdb_str)
            tmp_path = tmp.name
        try:
            traj_top = md.load(tmp_path)
        finally:
            os.unlink(tmp_path)

        N_idx, CA_idx, C_idx, res_names, chain_ids = _backbone_indices(
            traj_top.topology)
        if not N_idx:
            raise ValueError(f"No complete backbone residues in {h5_path}")

        N_arr  = np.array(N_idx,  dtype=int)
        CA_arr = np.array(CA_idx, dtype=int)
        C_arr  = np.array(C_idx,  dtype=int)

        R_aa_parts, t_parts = [], []
        traj_break_frames = []  # start-of-trajectory frame indices (skip first)
        cum = 0

        for temp in temps:
            if temp not in grp:
                continue
            for rep in reps:
                if rep not in grp[temp]:
                    continue
                coords_ds = grp[temp][rep]["coords"]  # [F, N_atoms, 3] Å float32
                F = coords_ds.shape[0]
                if F < 2:
                    continue

                c = np.array(coords_ds)                          # [F, N_atoms, 3]
                N_xyz  = torch.tensor(c[:, N_arr,  :], dtype=torch.float32)
                CA_xyz = torch.tensor(c[:, CA_arr, :], dtype=torch.float32)
                C_xyz  = torch.tensor(c[:, C_arr,  :], dtype=torch.float32)

                R, t = g.build_frames(N_xyz, CA_xyz, C_xyz)     # [F,N,3,3], [F,N,3]
                R_aa = g.so3_log(R).half()                       # [F, N, 3] float16
                t_f  = t.half()                                  # [F, N, 3] float16

                # Split at degenerate frames so no lag pair straddles a bad frame.
                # Each contiguous run of valid frames becomes its own sub-trajectory.
                valid = R_aa.isfinite().all(dim=-1).all(dim=-1)  # [F] bool
                if valid.all():
                    if cum > 0:
                        traj_break_frames.append(cum)
                    R_aa_parts.append(R_aa)
                    t_parts.append(t_f)
                    cum += F
                else:
                    valid_np = valid.numpy()
                    seg_start = None
                    for fi in range(F + 1):
                        is_v = fi < F and valid_np[fi]
                        if is_v and seg_start is None:
                            seg_start = fi
                        elif not is_v and seg_start is not None:
                            seg_len = fi - seg_start
                            if seg_len >= 2:
                                if cum > 0:
                                    traj_break_frames.append(cum)
                                R_aa_parts.append(R_aa[seg_start:fi])
                                t_parts.append(t_f[seg_start:fi])
                                cum += seg_len
                            seg_start = None

    if not R_aa_parts:
        raise ValueError(f"No valid trajectories found in {h5_path}")

    R_aa = torch.cat(R_aa_parts, dim=0)
    t    = torch.cat(t_parts,    dim=0)

    res_type  = vocab.residue_indices(res_names)
    chain_id  = torch.tensor(chain_ids, dtype=torch.long)
    res_index = torch.arange(len(res_names), dtype=torch.long)
    traj_breaks = torch.tensor(traj_break_frames, dtype=torch.long)

    return {
        "R_aa":        R_aa,
        "t":           t,
        "res_type":    res_type,
        "chain_id":    chain_id,
        "res_index":   res_index,
        "dt":          float(dt_ps),
        "seq":         res_names,
        "n_res":       len(res_names),
        "traj_breaks": traj_breaks,
    }
