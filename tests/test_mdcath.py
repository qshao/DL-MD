"""Tests for lsmd.mdcath and traj_breaks integration."""
import io
import os
import struct
import tempfile

import h5py
import numpy as np
import pytest
import torch

from lsmd import data, mdcath as mc, vocab


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fake_pdb(n_res, chain="A"):
    """Return a minimal PDB string with backbone N/CA/C for n_res residues."""
    lines = []
    atom_idx = 1
    for i in range(n_res):
        res_num = i + 1
        for name, (x, y, z) in [
            ("N",  (0.0, 1.0, 0.0)),
            ("CA", (0.0, 0.0, 0.0)),
            ("C",  (1.5, 0.0, 0.0)),
        ]:
            lines.append(
                f"ATOM  {atom_idx:5d}  {name:<3s} ALA {chain}{res_num:4d}    "
                f"{x + i*4:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}  "
            )
            atom_idx += 1
    lines.append("END")
    return "\n".join(lines)


def _make_fake_h5(n_res=8, n_frames=12, temps=None, reps=None, path=None):
    """Write a synthetic mdCATH H5 file and return its path."""
    if temps is None:
        temps = ["320", "348"]
    if reps is None:
        reps = ["0", "1"]
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)

    n_atoms = n_res * 3  # only N, CA, C — matches our fake PDB exactly
    domain = "1aabA00"

    with h5py.File(path, "w") as hf:
        grp = hf.create_group(domain)
        grp.attrs["numResidues"] = n_res
        grp.attrs["numProteinAtoms"] = n_atoms
        grp.attrs["numChains"] = 1

        pdb_str = _write_fake_pdb(n_res)
        grp.create_dataset("pdb", data=pdb_str.encode())

        rng = np.random.default_rng(0)
        for temp in temps:
            tg = grp.create_group(temp)
            for rep in reps:
                rg = tg.create_group(rep)
                rg.attrs["numFrames"] = n_frames
                coords = rng.random((n_frames, n_atoms, 3)).astype(np.float32)
                # Spread residues so frames look geometrically valid
                for j in range(n_res):
                    offset = j * 4.0
                    coords[:, j * 3 + 0, :] += [offset, 1.0, 0.0]   # N
                    coords[:, j * 3 + 1, :] += [offset, 0.0, 0.0]   # CA
                    coords[:, j * 3 + 2, :] += [offset + 1.5, 0.0, 0.0]  # C
                rg.create_dataset("coords", data=coords)
    return path, domain


# ---------------------------------------------------------------------------
# Tests: traj_breaks in physical_lag_pairs
# ---------------------------------------------------------------------------

def test_physical_lag_pairs_no_breaks():
    # Without traj_breaks, behaviour is unchanged
    pairs = data.physical_lag_pairs(10, dt=100.0, lags_ps=[200.0])
    assert pairs.shape[1] == 3
    # lag of 200 ps / 100 ps per frame = 2 frames; starts 0..7
    assert pairs[:, 2].unique().tolist() == [2]
    assert pairs[:, 0].min() == 0
    assert pairs[:, 1].max() == 9


def test_physical_lag_pairs_with_breaks_no_cross():
    # Two trajectories of 10 frames each (total 20), break at frame 10.
    traj_breaks = torch.tensor([10], dtype=torch.long)
    pairs = data.physical_lag_pairs(20, dt=100.0, lags_ps=[200.0],
                                    traj_breaks=traj_breaks)
    tau = 2
    # No pair should straddle the break (frame 10)
    assert not ((pairs[:, 0] < 10) & (pairs[:, 1] >= 10)).any(), \
        "Found pair crossing trajectory boundary"
    # Both segments contribute pairs
    seg0_pairs = pairs[pairs[:, 0] < 10]
    seg1_pairs = pairs[pairs[:, 0] >= 10]
    assert seg0_pairs.shape[0] > 0
    assert seg1_pairs.shape[0] > 0


def test_physical_lag_pairs_empty_breaks():
    # Empty traj_breaks tensor is treated as single trajectory
    pairs_none  = data.physical_lag_pairs(10, dt=100.0, lags_ps=[200.0],
                                           traj_breaks=None)
    pairs_empty = data.physical_lag_pairs(10, dt=100.0, lags_ps=[200.0],
                                           traj_breaks=torch.zeros(0, dtype=torch.long))
    assert pairs_none.shape == pairs_empty.shape


def test_physical_lag_pairs_lag_too_large_for_segment():
    # A lag larger than each segment → no pairs generated
    traj_breaks = torch.tensor([5], dtype=torch.long)
    pairs = data.physical_lag_pairs(10, dt=100.0, lags_ps=[1000.0],
                                    traj_breaks=traj_breaks)
    assert pairs.shape[0] == 0


# ---------------------------------------------------------------------------
# Tests: build_shard_from_h5
# ---------------------------------------------------------------------------

def test_build_shard_from_h5_keys_and_dtypes():
    path, _ = _make_fake_h5(n_res=8, n_frames=12, temps=["320", "348"], reps=["0", "1"])
    try:
        shard = mc.build_shard_from_h5(path, dt_ps=500.0)
    finally:
        os.unlink(path)

    assert "R_aa" in shard
    assert "t" in shard
    assert "traj_breaks" in shard
    assert shard["R_aa"].dtype == torch.float16
    assert shard["t"].dtype == torch.float16
    assert shard["n_res"] == 8


def test_build_shard_from_h5_shapes():
    n_res, n_frames = 8, 12
    temps, reps = ["320", "348"], ["0", "1"]
    n_trajs = len(temps) * len(reps)  # 4

    path, _ = _make_fake_h5(n_res=n_res, n_frames=n_frames, temps=temps, reps=reps)
    try:
        shard = mc.build_shard_from_h5(path, dt_ps=500.0, temps=temps, reps=reps)
    finally:
        os.unlink(path)

    F_total = n_trajs * n_frames
    assert shard["R_aa"].shape == (F_total, n_res, 3)
    assert shard["t"].shape    == (F_total, n_res, 3)
    assert shard["res_type"].shape == (n_res,)
    # traj_breaks: one entry per trajectory boundary (n_trajs - 1)
    assert shard["traj_breaks"].shape == (n_trajs - 1,)
    # traj_breaks should be strictly increasing starting at n_frames
    assert shard["traj_breaks"][0].item() == n_frames
    assert (shard["traj_breaks"].diff() > 0).all()


def test_build_shard_from_h5_fixed_vocab():
    path, _ = _make_fake_h5(n_res=6)
    try:
        shard = mc.build_shard_from_h5(path, dt_ps=500.0)
    finally:
        os.unlink(path)
    # Residue types from fixed vocab must be in [0, N_AA_TYPES)
    assert int(shard["res_type"].min()) >= 0
    assert int(shard["res_type"].max()) < vocab.N_AA_TYPES


def test_build_shard_from_h5_traj_breaks_no_cross_pairs():
    """traj_breaks from a real shard must prevent cross-boundary lag pairs."""
    n_frames, n_trajs = 15, 3
    path, _ = _make_fake_h5(n_res=6, n_frames=n_frames,
                             temps=["320", "348", "379"], reps=["0"])
    try:
        shard = mc.build_shard_from_h5(path, dt_ps=100.0)
    finally:
        os.unlink(path)

    pairs = data.physical_lag_pairs(
        shard["t"].shape[0], shard["dt"], [200.0],
        traj_breaks=shard["traj_breaks"])

    if pairs.shape[0] == 0:
        return  # no pairs at all is also fine

    for wall in shard["traj_breaks"].tolist():
        cross = (pairs[:, 0] < wall) & (pairs[:, 1] >= wall)
        assert not cross.any(), f"Pair crosses boundary at frame {wall}"
