import math
import torch
from lsmd import transfer_validate as tv


def test_curve_rmse_self_is_zero():
    t = torch.linspace(0.0, 100.0, 11)
    v = torch.sin(t / 10.0)
    assert tv.curve_rmse(t, v, t, v) < 1e-9


def test_curve_rmse_constant_offset():
    t = torch.linspace(0.0, 100.0, 11)
    v = torch.zeros(11)
    assert abs(tv.curve_rmse(t, v, t, v + 3.0) - 3.0) < 1e-6


def test_interp_to_grid_midpoint():
    t = torch.tensor([0.0, 10.0])
    v = torch.tensor([0.0, 10.0])
    out = tv.interp_to_grid(t, v, torch.tensor([5.0]))
    assert abs(out.item() - 5.0) < 1e-6


def test_rg_of_static_structure_is_constant():
    ca = torch.randn(1, 8, 3).repeat(5, 1, 1)  # identical frames
    rg = tv.radius_of_gyration(ca)
    assert rg.shape == (5,)
    assert (rg - rg[0]).abs().max() < 1e-6


def test_rg_js_identical_ensembles_near_zero():
    torch.manual_seed(0)
    ca = torch.randn(40, 8, 3)
    assert tv.rg_distribution_js(ca, ca.clone()) < 1e-6
