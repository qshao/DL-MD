import math
import torch
from lsmd.learned_energy import LearnedCGEnergy
from lsmd import cg_energy as cge


def _toy_protein(seed=0):
    g = torch.Generator().manual_seed(seed)
    N = 12
    t = torch.randn(N, 3, generator=g) * 5.0
    res_type = torch.randint(0, 20, (N,), generator=g)
    chain_id = torch.zeros(N, dtype=torch.long)
    return t, res_type, chain_id


def test_init_matches_cg_energy_defaults():
    t, rt, cid = _toy_protein()
    e = LearnedCGEnergy()
    got = e(t, rt, cid)
    ref = cge.total_cg_energy(t, rt, cid)   # default w=1, k_angle=10, eps=0.3
    assert torch.allclose(got, ref, atol=1e-4)


def test_params_and_position_grads_flow():
    t, rt, cid = _toy_protein()
    e = LearnedCGEnergy()
    t = t.requires_grad_(True)
    out = e(t, rt, cid)
    out.backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()
    assert all(p.grad is not None for p in e.parameters())


def test_save_load_roundtrip(tmp_path):
    e = LearnedCGEnergy()
    with torch.no_grad():
        e.log_alpha_mj += 0.5
    p = tmp_path / "energy.pt"
    e.save(str(p))
    e2 = LearnedCGEnergy.load(str(p))
    assert torch.allclose(e2.log_alpha_mj, e.log_alpha_mj)
