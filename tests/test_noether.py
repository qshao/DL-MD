import math
import torch
import pytest
from lsmd.noether import noether_project


def _cross(a, b):
    return torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], dim=-1)


def test_noether_translation_removed():
    """Pure COM drift: all displacement is translation → output = t_old."""
    N = 10
    torch.manual_seed(0)
    t_old = torch.randn(N, 3)
    drift = torch.tensor([1.0, 2.0, 3.0])
    t_new = t_old + drift.unsqueeze(0)
    chain_id = torch.zeros(N, dtype=torch.long)
    t_out = noether_project(t_old, t_new, chain_id)
    assert (t_out - t_old).abs().max() < 1e-5


def test_noether_rotation_removed():
    """Pure rigid rotation: all displacement is rotational → output ≈ t_old."""
    N = 20
    angles = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    t_old = torch.stack([torch.cos(angles) * 5.0, torch.sin(angles) * 5.0,
                         torch.zeros(N)], dim=1)
    omega = torch.tensor([0.0, 0.0, 0.05])
    centroid = t_old.mean(dim=0)
    r_c = t_old - centroid
    delta_rot = _cross(omega.unsqueeze(0).expand(N, -1), r_c)
    t_new = t_old + delta_rot
    chain_id = torch.zeros(N, dtype=torch.long)
    t_out = noether_project(t_old, t_new, chain_id)
    assert (t_out - t_old).abs().max() < 1e-4


def test_noether_linear_momentum_zero():
    """Random update: net displacement (linear momentum) is zero after projection."""
    torch.manual_seed(42)
    N = 15
    t_old = torch.randn(N, 3)
    t_new = t_old + torch.randn(N, 3)
    chain_id = torch.zeros(N, dtype=torch.long)
    t_out = noether_project(t_old, t_new, chain_id)
    delta = t_out - t_old
    assert delta.sum(dim=0).abs().max() < 1e-5


def test_noether_angular_momentum_zero():
    """Random update: angular momentum is zero after projection."""
    torch.manual_seed(7)
    N = 15
    t_old = torch.randn(N, 3)
    t_new = t_old + torch.randn(N, 3)
    chain_id = torch.zeros(N, dtype=torch.long)
    t_out = noether_project(t_old, t_new, chain_id)
    delta = t_out - t_old
    centroid = t_old.mean(dim=0)
    r_c = t_old - centroid
    L = _cross(r_c, delta).sum(dim=0)
    assert L.abs().max() < 1e-4


def test_noether_two_chains_independent():
    """Each chain gets projected independently — both have L=0 and P=0."""
    torch.manual_seed(3)
    N = 20
    t_old = torch.randn(N, 3)
    t_new = t_old + torch.randn(N, 3)
    chain_id = torch.cat([torch.zeros(10, dtype=torch.long),
                          torch.ones(10, dtype=torch.long)])
    t_out = noether_project(t_old, t_new, chain_id)
    for c in [0, 1]:
        mask = chain_id == c
        delta_c = (t_out - t_old)[mask]
        assert delta_c.sum(dim=0).abs().max() < 1e-4, f"chain {c} linear momentum nonzero"
        centroid_c = t_old[mask].mean(dim=0)
        r_c = t_old[mask] - centroid_c
        L_c = _cross(r_c, delta_c).sum(dim=0)
        assert L_c.abs().max() < 1e-4, f"chain {c} angular momentum nonzero"
