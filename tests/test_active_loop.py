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
