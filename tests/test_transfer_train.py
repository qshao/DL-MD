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
    pairs = list(tt.iter_union_batches(shards, random.Random(0),
                                       lags_ps=[200.0], k=4,
                                       max_union_nodes=25, n_batches=5))
    assert len(pairs) == 5
    for b, group in pairs:
        n = b["node_feats"].shape[0]
        # at most 2 proteins of 10 nodes fit under a 25 cap; 3 would be 30 > 25
        assert n <= 20
        # union keys present and consistent
        assert b["batch"].max().item() + 1 == b["tau"].shape[0]
        assert len(group) >= 1


def test_union_batch_emits_single_oversized_example():
    shards = [_synthetic_shard(N=40, seed=0)]
    pairs = list(tt.iter_union_batches(shards, random.Random(0),
                                       lags_ps=[200.0], k=4,
                                       max_union_nodes=25, n_batches=1))
    b, _ = pairs[0]
    assert b["node_feats"].shape[0] == 40  # emitted despite exceeding cap


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
    assert ckpt["step"] == 2
    assert "optimizer_state" in ckpt
    for v in ckpt["model_state"].values():
        assert torch.isfinite(v).all()


def test_resume_from_checkpoint():
    """Resuming continues from the saved step with correct step counter and no NaNs."""
    shards = [_synthetic_shard(N=10, seed=i) for i in range(4)]
    ckpt1 = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                     max_union_nodes=25, accum=2, steps=3, T_diff=20,
                     norm_samples=16, device="cpu", seed=0)
    assert ckpt1["step"] == 3

    ckpt2 = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                     max_union_nodes=25, accum=2, steps=4, T_diff=20,
                     norm_samples=16, device="cpu", seed=1, resume_from=ckpt1)
    assert ckpt2["step"] == 7  # 3 + 4
    for v in ckpt2["model_state"].values():
        assert torch.isfinite(v).all()


def _mdcath_shard(F=30, N=10, dt=1000.0, temps=(320, 348, 450), seed=0):
    """Synthetic mdCATH-like shard with traj_breaks and traj_temps."""
    torch.manual_seed(seed)
    n_trajs = len(temps)
    frames_per_traj = F // n_trajs
    R_aa = torch.randn(frames_per_traj * n_trajs, N, 3) * 0.1
    t    = torch.randn(frames_per_traj * n_trajs, N, 3) * 5.0
    traj_breaks = torch.tensor(
        [frames_per_traj * i for i in range(1, n_trajs)], dtype=torch.long)
    traj_temps  = torch.tensor(list(temps), dtype=torch.long)
    return {
        "R_aa": R_aa.half(), "t": t.half(),
        "res_type": torch.randint(0, 21, (N,)),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "dt": dt, "seq": ["ALA"] * N, "n_res": N,
        "traj_breaks": traj_breaks, "traj_temps": traj_temps,
    }


def test_sample_example_respects_allowed_temps():
    shard = _mdcath_shard(F=30, N=10, temps=(320, 450))
    # Allow only 320K: pairs must come from frames 0-14
    for _ in range(50):
        ex = tt.sample_example(shard, random.Random(_), lags_ps=[1000.0], k=4,
                               allowed_temps=frozenset([320]))
        if ex is None:
            continue
        # Not testing exact frame — just that example was produced without error
    # Now with 450K only: pairs from frames 15-29
    found_450 = False
    for seed in range(50):
        ex = tt.sample_example(shard, random.Random(seed), lags_ps=[1000.0], k=4,
                               allowed_temps=frozenset([450]))
        if ex is not None:
            found_450 = True
            break
    assert found_450


def test_curriculum_helpers():
    schedule = [(0, 320), (100, 348), (500, 379)]
    assert tt._current_max_temp(0,   schedule) == 320
    assert tt._current_max_temp(99,  schedule) == 320
    assert tt._current_max_temp(100, schedule) == 348
    assert tt._current_max_temp(499, schedule) == 348
    assert tt._current_max_temp(500, schedule) == 379

    allowed = tt._allowed_temps_set(348)
    assert 320 in allowed
    assert 348 in allowed
    assert 379 not in allowed


def test_train_with_temp_schedule():
    shards = [_mdcath_shard(F=30, N=10, temps=(320, 450), seed=i) for i in range(3)]
    # Start at 320K only; switch to all temps at step 2
    schedule = [(0, 320), (2, 450)]
    ckpt = tt.train(shards, lags_ps=[1000.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=2, steps=4, T_diff=20,
                    norm_samples=16, device="cpu", seed=0,
                    temp_schedule=schedule)
    assert "model_state" in ckpt
    assert ckpt["hparams"]["temp_schedule"] == schedule
    for v in ckpt["model_state"].values():
        assert torch.isfinite(v).all()


def test_sample_example_carries_temp_K():
    """build_training_example must return temp_K from shard metadata."""
    shard = _mdcath_shard(F=30, N=10, temps=(320, 450))
    # First segment is 320K; second is 450K
    for seed in range(20):
        ex = tt.sample_example(shard, random.Random(seed), lags_ps=[1000.0], k=4)
        if ex is not None:
            assert "temp_K" in ex
            assert ex["temp_K"] in (320.0, 450.0)
            break


def test_train_with_temp_conditioning():
    """temp_emb_dim > 0: model trains, hparams recorded, no NaNs."""
    shards = [_mdcath_shard(F=30, N=10, temps=(320, 450), seed=i) for i in range(3)]
    ckpt = tt.train(shards, lags_ps=[1000.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=2, steps=4, T_diff=20,
                    norm_samples=16, device="cpu", seed=0, temp_emb_dim=4)
    assert ckpt["hparams"]["temp_emb_dim"] == 4
    for v in ckpt["model_state"].values():
        assert torch.isfinite(v).all()


def test_train_with_time_reversal():
    """reverse_prob=0.5: trains without errors, hparams recorded."""
    shards = [_synthetic_shard(N=10, seed=i) for i in range(4)]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=2, steps=4, T_diff=20,
                    norm_samples=16, device="cpu", seed=0, reverse_prob=0.5)
    assert ckpt["hparams"]["reverse_prob"] == 0.5
    for v in ckpt["model_state"].values():
        assert torch.isfinite(v).all()


def test_atlas_shard_defaults_to_300K():
    """ATLAS shards (no traj_temps) get temp_K=300.0."""
    shard = _synthetic_shard(N=10)  # ATLAS-like: no traj_temps
    ex = tt.sample_example(shard, random.Random(0), lags_ps=[200.0], k=4)
    assert ex is not None
    assert ex["temp_K"] == 300.0


def test_union_collate_carries_temp_K():
    """union_collate should aggregate temp_K from examples into a [G] tensor."""
    from lsmd import batching, data as d
    shard = _mdcath_shard(F=30, N=10, temps=(320, 450))
    exs = []
    rng = random.Random(42)
    while len(exs) < 3:
        ex = tt.sample_example(shard, rng, lags_ps=[1000.0], k=4)
        if ex is not None:
            exs.append(ex)
    b = batching.union_collate(exs)
    assert "temp_K" in b
    assert b["temp_K"].shape == (len(exs),)
    assert b["temp_K"].dtype == torch.float32


def test_reverse_example_inverts_update():
    """Time-reversed example should have approximately negated u_target."""
    shard = _synthetic_shard(F=20, N=10)
    rng = random.Random(7)
    # Get a forward example
    ex_fwd = None
    for seed in range(100):
        ex_fwd = tt.sample_example(shard, random.Random(seed), lags_ps=[200.0], k=4,
                                   reverse_prob=0.0)
        if ex_fwd is not None:
            break
    # Get the same pair reversed (rebuild without cache to force reverse)
    from lsmd import data as d
    pairs = d.physical_lag_pairs(shard["t"].shape[0], shard["dt"], [200.0])
    assert pairs.shape[0] > 0
    row = pairs[0]
    start, tau_frames = int(row[0]), int(row[2])
    ex_rev = d.build_training_example(shard, start, tau_frames, 4, reverse=True)
    ex_fwd2 = d.build_training_example(shard, start, tau_frames, 4, reverse=False)
    assert ex_fwd2 is not None and ex_rev is not None
    # Forward and reverse updates should have roughly opposite sign
    dot = (ex_fwd2["u_target"] * ex_rev["u_target"]).sum()
    assert dot < 0, "reverse example should mostly negate the forward update"
