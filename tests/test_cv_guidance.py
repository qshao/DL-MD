import torch
import pytest
from lsmd.cv_guidance import CVSpace, build_cv_guidance
from lsmd import featurize as feat
from lsmd import geometry as g


def _coords(F=20, N=10, seed=0):
    torch.manual_seed(seed)
    return torch.randn(F, N, 3) * 5.0


def test_fit_sets_attributes():
    cv = CVSpace(n_pc=3)
    cv.fit(_coords(F=20, N=10))
    assert cv.mean.shape == (30,)       # 3N = 30
    assert cv.components.shape == (3, 30)
    assert cv.rg_mean.ndim == 0
    assert cv.rg_std > 0
    assert cv.rmsd_std > 0


def test_project_single_shape():
    cv = CVSpace(n_pc=3)
    cv.fit(_coords(F=20, N=10))
    t = torch.randn(10, 3)
    out = cv.project_single(t)
    assert out.shape == (5,)   # n_pc=3 + Rg + RMSD = 5


def test_project_single_is_differentiable():
    cv = CVSpace(n_pc=2)
    cv.fit(_coords(F=20, N=8))
    t = torch.randn(8, 3, requires_grad=True)
    out = cv.project_single(t)
    out.sum().backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad).all()


def test_repulsion_zero_with_empty_buffer():
    cv = CVSpace(n_pc=2)
    cv.fit(_coords(F=20, N=8))
    t = torch.randn(8, 3)
    c = cv.project_single(t)
    V = cv.repulsion(c, [], sigma=1.0)
    assert V.item() == 0.0


def test_repulsion_positive_with_buffer():
    cv = CVSpace(n_pc=2)
    cv.fit(_coords(F=20, N=8))
    t = torch.randn(8, 3)
    c = cv.project_single(t)
    buf = [c.detach()]
    V = cv.repulsion(c, buf, sigma=1.0)
    assert V.item() > 0.0


def test_repulsion_gradient_flows():
    cv = CVSpace(n_pc=2)
    cv.fit(_coords(F=20, N=8))
    t = torch.randn(8, 3, requires_grad=True)
    c = cv.project_single(t)
    buf = [torch.randn(4).detach()]
    V = cv.repulsion(c, buf, sigma=1.0)
    V.backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad).all()


def test_save_load_roundtrip(tmp_path):
    cv = CVSpace(n_pc=3)
    cv.fit(_coords(F=20, N=10))
    path = str(tmp_path / "cv.pt")
    cv.save(path)
    cv2 = CVSpace.load(path)
    assert cv2.n_pc == 3
    assert torch.allclose(cv2.mean, cv.mean)
    assert torch.allclose(cv2.components, cv.components)


def _simple_setup(N=8, n_pc=2, seed=7):
    torch.manual_seed(seed)
    coords = torch.randn(20, N, 3) * 5.0
    cv_space = CVSpace(n_pc=n_pc)
    cv_space.fit(coords)
    R = g.so3_exp(torch.zeros(N, 3))  # identity rotations
    t = coords[0].clone()
    scale = torch.ones(6)
    chain_id = torch.zeros(N, dtype=torch.long)
    return cv_space, R, t, chain_id, scale


def test_build_cv_guidance_empty_buffer_is_identity():
    cv_space, R, t, chain_id, scale = _simple_setup()
    fn = build_cv_guidance(R, t, chain_id, scale, cv_space,
                           buffer=[], k_guide=0.5, sigma_cv=1.0)
    u = torch.randn(8, 6)
    out = fn(u)
    assert torch.allclose(out, u)


def test_build_cv_guidance_with_buffer_changes_u():
    cv_space, R, t, chain_id, scale = _simple_setup()
    # Put the current structure into the buffer so repulsion is strong
    with torch.no_grad():
        _, t_cur = feat.apply_update(R, t, torch.zeros(8, 6))
    cv_cur = cv_space.project_single(t_cur).detach()
    buffer = [cv_cur]
    fn = build_cv_guidance(R, t, chain_id, scale, cv_space,
                           buffer=buffer, k_guide=0.5, sigma_cv=1.0)
    u = torch.randn(8, 6) * 0.1  # Small perturbation to avoid being at the peak
    out = fn(u)
    assert not torch.allclose(out, u), "guidance should change u when buffer is non-empty"
    assert torch.isfinite(out).all()


def test_build_cv_guidance_k_guide_zero_is_identity():
    cv_space, R, t, chain_id, scale = _simple_setup()
    cv_cur = cv_space.project_single(t).detach()
    fn = build_cv_guidance(R, t, chain_id, scale, cv_space,
                           buffer=[cv_cur], k_guide=0.0, sigma_cv=1.0)
    u = torch.randn(8, 6)
    out = fn(u)
    assert torch.allclose(out, u)
