import torch
from lsmd.normalize import UpdateNorm


def test_fit_and_roundtrip():
    torch.manual_seed(0)
    u = torch.randn(1000, 6) * torch.tensor([2.0, 2.0, 2.0, 0.3, 0.3, 0.3])
    norm = UpdateNorm.fit(u)
    z = norm.normalize(u)
    # scale is the 99th percentile of |u|, so normalized 99th-pct abs ≈ 1
    q99 = z.abs().quantile(0.99, dim=0)
    assert torch.allclose(q99, torch.ones(6), atol=0.15)
    # round-trip is exact
    assert torch.allclose(norm.denormalize(norm.normalize(u)), u, atol=1e-5)


def test_state_dict_roundtrip():
    u = torch.randn(50, 6) + 1.0
    norm = UpdateNorm.fit(u)
    norm2 = UpdateNorm.from_state_dict(norm.state_dict())
    assert torch.allclose(norm.scale, norm2.scale)


def test_scale_is_floored():
    u = torch.zeros(10, 6)            # zero variance
    norm = UpdateNorm.fit(u)
    assert (norm.scale >= 1e-6).all()
    assert torch.isfinite(norm.normalize(u)).all()


def test_constructor_clamps_scale_floor():
    # __init__ is the canonical enforcement point — fit/from_state_dict delegate to it
    norm = UpdateNorm(torch.zeros(6))
    assert (norm.scale >= 1e-6).all()
    assert torch.isfinite(norm.normalize(torch.ones(6))).all()
