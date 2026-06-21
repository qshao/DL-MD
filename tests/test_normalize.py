import torch
from lsmd.normalize import UpdateNorm


def test_fit_and_roundtrip():
    torch.manual_seed(0)
    u = torch.randn(1000, 6) * torch.tensor([2.0, 2.0, 2.0, 0.3, 0.3, 0.3])
    norm = UpdateNorm.fit(u)
    # normalized columns have ~unit std
    z = norm.normalize(u)
    assert torch.allclose(z.std(0), torch.ones(6), atol=0.1)
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
