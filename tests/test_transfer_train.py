import random
import torch
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


def test_sample_example_shapes():
    sh = _synthetic_shard(N=10)
    ex = tt.sample_example(sh, random.Random(0), lags_ps=[200.0, 1000.0], k=4)
    assert ex["node_feats"].shape == (10, 24)
    assert ex["u_target"].shape == (10, 6)
    assert ex["edge_feats"].shape == (10 * 4, 13)


def test_sample_example_none_when_lag_too_large():
    sh = _synthetic_shard(F=3, N=10, dt=200.0)
    ex = tt.sample_example(sh, random.Random(0), lags_ps=[2000.0], k=4)  # 10 frames
    assert ex is None


def test_union_batches_respect_node_cap():
    shards = [_synthetic_shard(N=10, seed=i) for i in range(6)]
    batches = list(tt.iter_union_batches(shards, random.Random(0),
                                         lags_ps=[200.0], k=4,
                                         max_union_nodes=25, n_batches=5))
    assert len(batches) == 5
    for b in batches:
        n = b["node_feats"].shape[0]
        # at most 2 proteins of 10 nodes fit under a 25 cap; 3 would be 30 > 25
        assert n <= 20
        # union keys present and consistent
        assert b["batch"].max().item() + 1 == b["tau"].shape[0]


def test_union_batch_emits_single_oversized_example():
    shards = [_synthetic_shard(N=40, seed=0)]
    batches = list(tt.iter_union_batches(shards, random.Random(0),
                                         lags_ps=[200.0], k=4,
                                         max_union_nodes=25, n_batches=1))
    assert batches[0]["node_feats"].shape[0] == 40  # emitted despite exceeding cap
