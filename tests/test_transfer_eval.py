import torch
from lsmd import transfer_eval as te
from lsmd import transfer_train as tt
from lsmd import geometry as g


def _synthetic_shard(F=20, N=10, dt=200.0, seed=0):
    torch.manual_seed(seed)
    return {
        "R": g.so3_exp(torch.randn(F, N, 3) * 0.1),
        "t": torch.randn(F, N, 3) * 5.0,
        "res_type": torch.randint(0, 21, (N,)),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "dt": dt, "seq": ["ALA"] * N, "n_res": N,
    }


def test_rollout_shape_and_finite():
    import random
    shards = [_synthetic_shard(N=10, seed=i) for i in range(3)]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=1, steps=2, T_diff=20,
                    norm_samples=16, device="cpu", seed=0)
    net, sched, norm = te.load_checkpoint(ckpt, device="cpu")
    sh = shards[0]
    traj = te.rollout(net, sched, norm, sh["R"][0], sh["t"][0],
                      sh["res_type"], sh["chain_id"], sh["res_index"],
                      steps=4, tau_ps=200.0, k=4, diff_steps=3, device="cpu")
    assert traj.shape == (5, 10, 3)
    assert torch.isfinite(traj).all()
