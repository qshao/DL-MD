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


from lsmd.normalize import UpdateNorm


def test_fit_update_norm_returns_positive_scale():
    shards = [_synthetic_shard(N=10, seed=i) for i in range(4)]
    norm = tt.fit_update_norm(shards, random.Random(0), lags_ps=[200.0],
                              k=4, n_samples=20)
    assert isinstance(norm, UpdateNorm)
    assert norm.scale.shape == (6,)
    assert (norm.scale > 0).all()


def test_train_one_step_returns_checkpoint_without_nans():
    shards = [_synthetic_shard(N=10, seed=i) for i in range(4)]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=2, steps=2, T_diff=20,
                    norm_samples=16, device="cpu", seed=0)
    assert "model_state" in ckpt and "update_norm" in ckpt
    assert ckpt["n_aa_types"] == 21
    for v in ckpt["model_state"].values():
        assert torch.isfinite(v).all()
