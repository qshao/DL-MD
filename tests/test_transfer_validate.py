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


def test_msd_static_structure_is_zero():
    ca = torch.randn(1, 6, 3).repeat(20, 1, 1)
    t, msd = tv.msd_curve(ca, dt_ps=10.0)
    assert msd.abs().max() < 1e-6
    assert torch.allclose(t, torch.arange(1, 11, dtype=torch.float64) * 10.0)


def test_msd_diffusion_increases_monotonically():
    torch.manual_seed(5)
    steps = torch.randn(40, 6, 3) * 0.5
    ca = steps.cumsum(dim=0)            # Brownian per residue
    _, msd = tv.msd_curve(ca, dt_ps=1.0)
    # First half should be non-decreasing on average
    assert msd[5] > msd[1]
    assert msd[-1] > msd[0]


def test_acf_lag_zero_is_one():
    torch.manual_seed(6)
    q = torch.randn(200)
    t, acf = tv.cv_autocorrelation(q, dt_ps=2.0)
    assert abs(acf[0].item() - 1.0) < 1e-6
    assert t[0].item() == 0.0


def test_relaxation_time_recovers_ou_timescale():
    # Ornstein-Uhlenbeck: q[t+1] = (1 - 1/theta) q[t] + noise
    torch.manual_seed(7)
    theta = 20.0
    n = 8000
    q = torch.zeros(n)
    for i in range(1, n):
        q[i] = (1.0 - 1.0 / theta) * q[i - 1] + torch.randn(1).item()
    dt_ps = 1.0
    t, acf = tv.cv_autocorrelation(q, dt_ps=dt_ps)
    tau = tv.relaxation_time_ps(t, acf)
    # Continuous-time relaxation time of this AR(1) is ~theta * dt_ps
    assert 0.5 * theta * dt_ps < tau < 2.0 * theta * dt_ps


def test_validate_returns_full_schema():
    torch.manual_seed(8)
    ca_model = torch.randn(60, 10, 3)
    ca_md = torch.randn(120, 10, 3)
    rep = tv.validate(ca_model, ca_md, tau_ps=2000.0, dt_md_ps=200.0)
    for section in ("structural", "thermodynamic", "kinetic"):
        assert section in rep
    assert set(rep["structural"]) >= {"rmsf_corr", "dist_js", "rg_js",
                                      "ca_bond_mean", "clash_count"}
    assert set(rep["thermodynamic"]) >= {"fes_js", "fes_rmse_kT", "pop_tv"}
    assert set(rep["kinetic"]) >= {"msd_rmse", "acf_rmse", "relax_model_ps",
                                   "relax_md_ps", "relax_ratio"}


def test_validate_identical_ensembles_have_strong_agreement():
    torch.manual_seed(9)
    ca = torch.randn(80, 10, 3)
    rep = tv.validate(ca, ca.clone(), tau_ps=200.0, dt_md_ps=200.0)
    assert rep["structural"]["dist_js"] < 1e-3
    assert rep["thermodynamic"]["pop_tv"] < 1e-3
    assert abs(rep["kinetic"]["relax_ratio"] - 1.0) < 1e-3
