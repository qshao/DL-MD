import torch
from lsmd import physics_loss as pl
from lsmd import geometry as g


def _frames(n, seed=0):
    torch.manual_seed(seed)
    R = g.so3_exp(torch.randn(n, 3) * 0.1)
    t = torch.zeros(n, 3)
    t[:, 0] = torch.arange(n).float() * 3.8
    return R, t


def test_penalty_zero_update_on_ideal_chain_is_small():
    R, t = _frames(6)
    u = torch.zeros(6, 6)
    chain = torch.zeros(6, dtype=torch.long)
    pen = pl.geometric_penalty(R, t, u, chain, w_clash=1.0)
    assert pen.item() < 1e-4


def test_chain_break_update_penalized_more_than_preserving():
    R, t = _frames(6)
    chain = torch.zeros(6, dtype=torch.long)
    keep = torch.zeros(6, 6)
    breaker = torch.zeros(6, 6); breaker[3, 0] = 10.0
    assert pl.geometric_penalty(R, t, breaker, chain) > \
           pl.geometric_penalty(R, t, keep, chain) + 1.0


def test_no_bond_penalty_across_chain_boundary():
    R, t = _frames(6)
    chain = torch.tensor([0, 0, 0, 1, 1, 1])
    u = torch.zeros(6, 6)
    pen = pl.geometric_penalty(R, t, u, chain, w_clash=0.0)
    assert pen.item() < 1e-4


def test_penalty_is_differentiable():
    R, t = _frames(5)
    u = torch.zeros(5, 6, requires_grad=True)
    chain = torch.zeros(5, dtype=torch.long)
    pl.geometric_penalty(R, t, u, chain).backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()


# ---- P4-T2 tests ----
import random
from lsmd import data, batching
from lsmd import transfer_model as tm
from lsmd.model import NoiseSchedule


def _shard(F=20, N=8, dt=200.0, seed=0):
    torch.manual_seed(seed)
    return {"R": g.so3_exp(torch.randn(F, N, 3) * 0.1),
            "t": torch.randn(F, N, 3) * 5.0,
            "res_type": torch.randint(0, 21, (N,)),
            "chain_id": torch.zeros(N, dtype=torch.long),
            "res_index": torch.arange(N), "dt": dt}


def test_build_training_example_carries_current_frames():
    ex = data.build_training_example(_shard(N=8), i=0, tau_frames=2, k=4)
    assert ex["R_cur"].shape == (8, 3, 3)
    assert ex["t_cur"].shape == (8, 3)
    assert ex["chain_id"].shape == (8,)


def test_collate_physics_offsets_chains_per_graph():
    e0 = data.build_training_example(_shard(N=5, seed=0), 0, 2, 4)
    e1 = data.build_training_example(_shard(N=7, seed=1), 0, 2, 4)
    phys = pl.collate_physics([e0, e1])
    assert phys["R_cur"].shape == (12, 3, 3)
    assert phys["t_cur"].shape == (12, 3)
    g0 = set(phys["global_chain"][:5].tolist())
    g1 = set(phys["global_chain"][5:].tolist())
    assert not (g0 & g1)


def test_lambda_schedule_ramps_then_saturates():
    assert pl.lambda_schedule(0, 100, 0.5) == 0.0
    assert abs(pl.lambda_schedule(50, 100, 0.5) - 0.25) < 1e-6
    assert pl.lambda_schedule(200, 100, 0.5) == 0.5


def test_lam_zero_equals_plain_ddpm_loss():
    e0 = data.build_training_example(_shard(N=5, seed=0), 0, 2, 4)
    e1 = data.build_training_example(_shard(N=7, seed=1), 0, 2, 4)
    union = batching.union_collate([e0, e1])
    phys = pl.collate_physics([e0, e1])
    net = tm.PropagatorNet(hidden=16, layers=2)
    sched = NoiseSchedule(T=30)
    scale = torch.ones(6)

    torch.manual_seed(7)
    plain = tm.ddpm_loss_union(net, union["u_target"], union["node_feats"],
                               union["edge_index"], union["edge_feats"],
                               union["tau"], union["batch"], sched)
    torch.manual_seed(7)
    phys_loss = pl.ddpm_physics_loss(net, union, phys, scale, sched, lam=0.0)
    assert torch.allclose(plain, phys_loss, atol=1e-6)


def test_physics_term_raises_loss_when_lambda_positive():
    e0 = data.build_training_example(_shard(N=6, seed=2), 0, 2, 4)
    union = batching.union_collate([e0])
    phys = pl.collate_physics([e0])
    net = tm.PropagatorNet(hidden=16, layers=2)
    sched = NoiseSchedule(T=30)
    scale = torch.ones(6)

    torch.manual_seed(3)
    base = pl.ddpm_physics_loss(net, union, phys, scale, sched, lam=0.0)
    torch.manual_seed(3)
    with_phys = pl.ddpm_physics_loss(net, union, phys, scale, sched, lam=5.0)
    assert with_phys >= base
