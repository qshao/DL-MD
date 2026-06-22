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


def test_pca_orders_variance_descending():
    torch.manual_seed(1)
    # Anisotropic cloud: large spread along residue-0 x, small elsewhere
    base = torch.randn(60, 4, 3) * 0.1
    base[:, 0, 0] += torch.randn(60) * 5.0
    mean, comps = tv.shared_pca(base, n_components=2)
    cv = tv.project_cv(base, mean, comps)
    assert cv.shape == (60, 2)
    assert cv[:, 0].var() >= cv[:, 1].var()


def test_pca_projection_is_zero_mean_on_fitting_set():
    torch.manual_seed(2)
    base = torch.randn(50, 4, 3)
    mean, comps = tv.shared_pca(base, n_components=2)
    cv = tv.project_cv(base, mean, comps)
    assert cv.mean(dim=0).abs().max() < 1e-6


def test_fes_identical_gaussians_low_rmse():
    torch.manual_seed(3)
    a = torch.randn(4000, 2)
    b = torch.randn(4000, 2)
    out = tv.fes_comparison(a, b, bins=20)
    assert out["fes_js"] < 0.1
    assert out["fes_rmse_kT"] < 0.6


def test_fes_disjoint_clouds_high_js():
    a = torch.randn(2000, 2) * 0.2 + torch.tensor([-5.0, 0.0])
    b = torch.randn(2000, 2) * 0.2 + torch.tensor([5.0, 0.0])
    out = tv.fes_comparison(a, b, bins=20)
    assert out["fes_js"] > 0.9


def test_free_energy_surface_empty_bins_are_nan():
    cv = torch.randn(500, 2)
    fg, _ = tv.free_energy_surface(cv, bins=30)
    assert torch.isnan(fg).any()


def test_populations_identical_ensembles_tv_zero():
    torch.manual_seed(4)
    cv = torch.randn(300, 2)
    out = tv.state_populations(cv, cv.clone(), n_states=4)
    assert out["pop_tv"] < 1e-6
    assert abs(sum(out["pop_model"]) - 1.0) < 1e-6


def test_populations_disjoint_clouds_tv_near_one():
    a = torch.randn(200, 2) * 0.1 + torch.tensor([-8.0, 0.0])
    b = torch.randn(200, 2) * 0.1 + torch.tensor([8.0, 0.0])
    # Fit clusters on a mix so both clouds get distinct centers
    mix = torch.cat([a, b], dim=0)
    out = tv.state_populations(a, b, n_states=2, seed=0)
    assert out["pop_tv"] > 0.9
