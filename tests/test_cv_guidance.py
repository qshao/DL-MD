import torch
import pytest
from lsmd.cv_guidance import CVSpace


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
