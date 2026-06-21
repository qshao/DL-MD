import torch
from lsmd import data
from lsmd import geometry as g


def _synthetic_frames(F=20, N=10, dt=200.0):
    R = g.so3_exp(torch.randn(F, N, 3) * 0.1)
    t = torch.randn(F, N, 3) * 5.0
    return {
        "R": R, "t": t,
        "res_type": torch.randint(0, 21, (N,)),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "dt": dt,
    }


def test_physical_lag_pairs_converts_ps_to_frames():
    # dt=200 ps/frame; lags 200 and 1000 ps -> 1 and 5 frames
    pairs = data.physical_lag_pairs(num_frames=20, dt=200.0, lags_ps=[200.0, 1000.0])
    taus = sorted(set(pairs[:, 2].tolist()))
    assert taus == [1, 5]
    # end = start + tau_frames, all within range
    assert (pairs[:, 1] == pairs[:, 0] + pairs[:, 2]).all()
    assert pairs[:, 1].max().item() <= 19


def test_physical_lag_pairs_skips_too_large():
    pairs = data.physical_lag_pairs(num_frames=4, dt=200.0, lags_ps=[2000.0])  # 10 frames
    assert pairs.shape[0] == 0


def test_build_training_example_shapes_and_tau():
    fr = _synthetic_frames(F=20, N=10, dt=200.0)
    ex = data.build_training_example(fr, i=0, tau_frames=5, k=4)
    assert ex["node_feats"].shape == (10, 24)
    assert ex["u_target"].shape == (10, 6)
    assert ex["edge_feats"].shape == (10 * 4, 13)
    assert ex["edge_index"].shape == (2, 10 * 4)
    assert ex["tau"] == 5 * 200.0            # physical ps


def test_build_training_example_zero_update_for_identical_frames():
    fr = _synthetic_frames(F=20, N=10)
    fr["t"][7] = fr["t"][0]                    # frame 7 identical to frame 0
    fr["R"][7] = fr["R"][0]
    ex = data.build_training_example(fr, i=0, tau_frames=7, k=4)
    assert ex["u_target"].abs().max().item() < 1e-4
