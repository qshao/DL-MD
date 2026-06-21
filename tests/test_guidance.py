import torch
from lsmd import guidance as gd
from lsmd import transfer_model as tm
from lsmd.model import NoiseSchedule
from lsmd import geometry as g
from lsmd import batching


def _frames(n):
    R = g.so3_exp(torch.zeros(n, 3))
    t = torch.zeros(n, 3); t[:, 0] = torch.arange(n).float() * 3.8
    return R, t


def test_guidance_step_lowers_energy_on_clashing_fixture():
    from lsmd import physics_loss as pl
    R, t = _frames(6)
    chain = torch.zeros(6, dtype=torch.long)
    scale = torch.ones(6)
    u0 = torch.zeros(6, 6); u0[:, 0] = -1.9 * torch.arange(6).float()
    before = pl.geometric_penalty(R, t, u0 * scale, chain)
    u0_g = gd.guidance_step(u0, R, t, chain, scale, gamma=0.05)
    after = pl.geometric_penalty(R, t, u0_g * scale, chain)
    assert after < before


def test_guidance_step_gamma_zero_is_identity():
    R, t = _frames(5)
    chain = torch.zeros(5, dtype=torch.long)
    u0 = torch.randn(5, 6)
    out = gd.guidance_step(u0, R, t, chain, torch.ones(6), gamma=0.0)
    assert torch.allclose(out, u0)


def test_guided_sampler_gamma_zero_matches_plain():
    n, k = 6, 4
    gr = {"node_feats": torch.randn(n, 24),
          "edge_index": torch.randint(0, n, (2, n * k)),
          "edge_feats": torch.randn(n * k, 13),
          "u_target": torch.randn(n, 6), "tau": 100.0}
    u = batching.union_collate([gr])
    net = tm.PropagatorNet(hidden=16, layers=2).eval()
    sched = NoiseSchedule(T=40)
    R, t = _frames(n)
    chain = torch.zeros(n, dtype=torch.long)

    torch.manual_seed(11)
    plain = tm.sample_ddpm_union(net, u["node_feats"], u["edge_index"],
                                 u["edge_feats"], u["tau"], u["batch"],
                                 sched, steps=5)
    torch.manual_seed(11)
    guided = gd.sample_ddpm_union_guided(net, u["node_feats"], u["edge_index"],
                                         u["edge_feats"], u["tau"], u["batch"],
                                         sched, R, t, chain, torch.ones(6),
                                         steps=5, gamma=0.0)
    assert torch.allclose(plain, guided, atol=1e-5)
