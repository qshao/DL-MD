"""Tests for lsmd/active_loop.py and the CVSpace cold-start guard."""
import pytest
import torch
from lsmd.cv_guidance import CVSpace


def test_cvspace_single_frame():
    """CVSpace.fit() must not crash on F=1 and must return 2D CV (Rg+RMSD only)."""
    N = 30
    coords = torch.randn(1, N, 3) * 10.0
    cv = CVSpace(n_pc=5)
    cv.fit(coords)  # must not raise
    assert cv.mean is not None
    assert cv.components is not None
    assert cv.components.shape == (5, N * 3)
    # All PC scores must be 0 (zero components → zero projection)
    proj = cv.project_single(coords[0])
    assert proj.shape == (7,)              # 5 PC + Rg + RMSD
    assert proj[:5].abs().max() < 1e-6    # PC scores ≈ 0


def test_cvspace_multi_frame():
    """CVSpace.fit() still works normally for F >= 2."""
    N = 20
    coords = torch.randn(10, N, 3) * 10.0
    cv = CVSpace(n_pc=3)
    cv.fit(coords)
    proj = cv.project_single(coords[0])
    assert proj.shape == (5,)   # 3 PC + Rg + RMSD
    assert proj.isfinite().all()


# ---------------------------------------------------------------------------
# Task 2: bootstrap helper tests
# ---------------------------------------------------------------------------
import os
import json
import tempfile
import numpy as np
from lsmd.active_loop import (
    _pdb_to_shard, _geometry_pass_rate, _min_rmsd_kabsch, bootstrap_check,
)


def _write_tiny_pdb(path, n_res=5):
    """Write a minimal backbone PDB (N, CA, C, O per residue)."""
    lines = ["REMARK tiny test PDB"]
    res_names = ["ALA", "GLY", "VAL", "LEU", "ILE"][:n_res]
    atom_names_full = [" N  ", " CA ", " C  ", " O  "]
    coords_per_res = [
        [0.0, 0.0, 0.0],   # N
        [1.458, 0.0, 0.0], # CA
        [2.009, 1.420, 0.0], # C
        [1.251, 2.390, 0.0], # O (approximate)
    ]
    serial = 1
    for ri, rn in enumerate(res_names):
        z_offset = ri * 3.8
        for aname, xyz in zip(atom_names_full, coords_per_res):
            x, y, z = xyz[0], xyz[1], xyz[2] + z_offset
            lines.append(
                f"ATOM  {serial:5d} {aname} {rn:3s} A{ri+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def test_pdb_to_shard():
    with tempfile.TemporaryDirectory() as tmp:
        pdb = os.path.join(tmp, "test.pdb")
        _write_tiny_pdb(pdb, n_res=5)
        shard = _pdb_to_shard(pdb)
    assert shard["t"].shape == (1, 5, 3)
    assert shard["R"].shape == (1, 5, 3, 3)
    assert shard["res_type"].shape == (5,)
    assert shard["n_res"] == 5
    assert isinstance(shard["dt"], float)


def test_geometry_pass_rate_good():
    """Ideal 3.8 Å bonds → high pass rate."""
    N = 10
    # Build ideal Cα chain: each residue 3.8 Å apart
    ca = torch.zeros(N, 3)
    for i in range(1, N):
        ca[i, 2] = i * 3.8
    proposals = [ca for _ in range(5)]
    rate = _geometry_pass_rate(proposals, ref_bond_A=3.8)
    assert rate >= 0.8


def test_geometry_pass_rate_bad():
    """Bonds at 4.6 Å (outside threshold) → low pass rate."""
    N = 10
    ca = torch.zeros(N, 3)
    for i in range(1, N):
        ca[i, 2] = i * 4.6   # stretched bonds
    proposals = [ca for _ in range(5)]
    rate = _geometry_pass_rate(proposals, ref_bond_A=3.8)
    assert rate == 0.0


def test_min_rmsd_kabsch_identical():
    """Min RMSD of structure to itself must be 0."""
    coords = torch.randn(20, 3)
    refs   = coords.unsqueeze(0)
    assert _min_rmsd_kabsch(coords, refs) < 1e-4


def test_min_rmsd_kabsch_shifted():
    """Min RMSD after translation must still be near 0 (Kabsch is translation-invariant)."""
    coords = torch.randn(20, 3)
    shifted = coords + torch.tensor([5.0, 3.0, -2.0])
    refs = shifted.unsqueeze(0)
    assert _min_rmsd_kabsch(coords, refs) < 1e-4
