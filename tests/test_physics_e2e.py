import torch
from lsmd import data, batching
from lsmd import geometry as g
from lsmd import physics_loss as pl
from lsmd import guidance as gd
from lsmd import transfer_model as tm
from lsmd.model import NoiseSchedule


def _shard(F=20, N=8, dt=200.0, seed=0):
    torch.manual_seed(seed)
    return {"R": g.so3_exp(torch.randn(F, N, 3) * 0.1),
            "t": torch.randn(F, N, 3) * 5.0,
            "res_type": torch.randint(0, 21, (N,)),
            "chain_id": torch.zeros(N, dtype=torch.long),
            "res_index": torch.arange(N), "dt": dt}


def test_c1_loss_backpropagates_through_net():
    e0 = data.build_training_example(_shard(N=6, seed=0), 0, 2, 4)
    e1 = data.build_training_example(_shard(N=8, seed=1), 0, 2, 4)
    union = batching.union_collate([e0, e1])
    phys = pl.collate_physics([e0, e1])
    net = tm.PropagatorNet(hidden=16, layers=2)
    sched = NoiseSchedule(T=30)
    loss = pl.ddpm_physics_loss(net, union, phys, torch.ones(6), sched, lam=1.0)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g_).all() for g_ in grads)


def test_c2_guidance_reduces_rollout_step_energy():
    n, k = 6, 4
    gr = {"node_feats": torch.randn(n, 24),
          "edge_index": torch.randint(0, n, (2, n * k)),
          "edge_feats": torch.randn(n * k, 13),
          "u_target": torch.randn(n, 6), "tau": 100.0}
    u = batching.union_collate([gr])
    net = tm.PropagatorNet(hidden=16, layers=2).eval()
    sched = NoiseSchedule(T=40)
    R = g.so3_exp(torch.zeros(n, 3))
    t = torch.zeros(n, 3); t[:, 0] = torch.arange(n).float() * 3.8
    chain = torch.zeros(n, dtype=torch.long)
    scale = torch.ones(6)

    def _energy(gamma, seed):
        torch.manual_seed(seed)
        out = gd.sample_ddpm_union_guided(net, u["node_feats"], u["edge_index"],
                                          u["edge_feats"], u["tau"], u["batch"],
                                          sched, R, t, chain, scale,
                                          steps=8, gamma=gamma)
        return pl.geometric_penalty(R, t, out * scale, chain).item()

    plain = sum(_energy(0.0, s) for s in range(5))
    guided = sum(_energy(0.2, s) for s in range(5))
    assert guided < plain
