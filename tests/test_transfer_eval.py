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


def test_evaluate_keys_and_finite():
    torch.manual_seed(0)
    ca_model = torch.randn(8, 10, 3) * 2.0
    ca_md = torch.randn(12, 10, 3) * 2.0
    m = te.evaluate(ca_model, ca_md)
    for key in ("rmsf_corr", "dist_js", "ca_bond_mean", "clash_count"):
        assert key in m
    assert -1.0 <= m["rmsf_corr"] <= 1.0
    assert 0.0 <= m["dist_js"] <= 1.0
    assert torch.isfinite(torch.tensor(m["ca_bond_mean"]))


def test_evaluate_identical_ensembles_have_high_rmsf_corr():
    torch.manual_seed(1)
    ca = torch.randn(10, 12, 3) * 2.0
    m = te.evaluate(ca, ca)
    assert m["rmsf_corr"] > 0.99
    assert m["dist_js"] < 1e-3


from lsmd.cv_guidance import CVSpace


def test_rollout_with_cv_space_runs_and_has_correct_shape():
    import random
    shards = [_synthetic_shard(N=10, seed=i) for i in range(3)]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=1, steps=2, T_diff=20,
                    norm_samples=16, device="cpu", seed=0)
    net, sched, norm = te.load_checkpoint(ckpt, device="cpu")
    sh = shards[0]
    # Fit CVSpace on training frames
    cv_space = CVSpace(n_pc=2)
    cv_space.fit(sh["t"])   # sh["t"]: [F, N, 3]
    cv_buffer = []
    traj = te.rollout(net, sched, norm, sh["R"][0], sh["t"][0],
                      sh["res_type"], sh["chain_id"], sh["res_index"],
                      steps=4, tau_ps=200.0, k=4, diff_steps=3, device="cpu",
                      cv_space=cv_space, cv_buffer=cv_buffer, k_guide=0.05,
                      sigma_cv=1.0, guide_warmup=0)
    assert traj.shape == (5, 10, 3)
    assert torch.isfinite(traj).all()


def test_rollout_cv_none_matches_original():
    """cv_space=None must reproduce the original rollout exactly."""
    import random
    shards = [_synthetic_shard(N=10, seed=i) for i in range(3)]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=1, steps=2, T_diff=20,
                    norm_samples=16, device="cpu", seed=42)
    net, sched, norm = te.load_checkpoint(ckpt, device="cpu")
    sh = shards[0]
    torch.manual_seed(0)
    traj_orig = te.rollout(net, sched, norm, sh["R"][0], sh["t"][0],
                           sh["res_type"], sh["chain_id"], sh["res_index"],
                           steps=3, tau_ps=200.0, k=4, diff_steps=3, device="cpu")
    torch.manual_seed(0)
    traj_cv = te.rollout(net, sched, norm, sh["R"][0], sh["t"][0],
                         sh["res_type"], sh["chain_id"], sh["res_index"],
                         steps=3, tau_ps=200.0, k=4, diff_steps=3, device="cpu",
                         cv_space=None, cv_buffer=None)
    assert torch.allclose(traj_orig, traj_cv)
