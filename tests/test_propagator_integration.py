"""End-to-end integration test for the transferable propagator.

Proves that build_training_example → union_collate → PropagatorNet →
sample_ddpm_union → apply_update composes into a valid single propagator step.

Uses synthetic frames (always runs) and real WT trajectory (skipped if absent).
"""
import os
import torch
from lsmd import data, batching, geometry as g
from lsmd import featurize as f
from lsmd.transfer_model import PropagatorNet, sample_ddpm_union
from lsmd.model import NoiseSchedule
from lsmd.normalize import UpdateNorm


def _synthetic_frames(F=10, N=12, dt=200.0):
    R = g.so3_exp(torch.randn(F, N, 3) * 0.1)
    t = torch.randn(F, N, 3) * 5.0
    return {"R": R, "t": t,
            "res_type": torch.randint(0, 21, (N,)),
            "chain_id": torch.zeros(N, dtype=torch.long),
            "res_index": torch.arange(N), "dt": dt}


def test_single_propagator_step_produces_valid_frames():
    torch.manual_seed(0)
    fr = _synthetic_frames(N=12)
    ex = data.build_training_example(fr, i=0, tau_frames=2, k=4)

    # fit normalization from a few example updates, normalize is identity-safe here
    norm = UpdateNorm.fit(ex["u_target"])
    u = batching.union_collate([ex])
    net = PropagatorNet(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)

    u_sample = sample_ddpm_union(net, u["node_feats"], u["edge_index"],
                                 u["edge_feats"], u["tau"], u["batch"],
                                 sched, steps=5)
    u_sample = norm.denormalize(u_sample)          # [12, 6]
    assert u_sample.shape == (12, 6)
    assert torch.isfinite(u_sample).all()

    # apply update to current frames → next frames are valid rotations + finite
    R_next, t_next = f.apply_update(fr["R"][0], fr["t"][0], u_sample)
    assert R_next.shape == (12, 3, 3) and t_next.shape == (12, 3)
    assert torch.isfinite(R_next).all() and torch.isfinite(t_next).all()
    # rotations stay orthonormal (R R^T = I)
    eye = torch.eye(3).expand(12, 3, 3)
    assert torch.allclose(R_next @ R_next.transpose(-1, -2), eye, atol=1e-3)


def test_real_wt_step_if_available():
    trr, gro = "WT/WT-sol6.trr", "WT/WT-sol6.gro"
    if not (os.path.exists(trr) and os.path.exists(gro)):
        import pytest
        pytest.skip("WT trajectory not present")
    fd = data.load_frames(trr, gro)              # provides R [F,N,3,3], t [F,N,3]
    from lsmd import vocab
    # re-key residue types onto the fixed vocab via residue names is done in the
    # ATLAS pipeline (Plan 2); here we just confirm frames feed the propagator.
    fr = {"R": fd["R"], "t": fd["t"],
          "res_type": fd["res_type"].clamp(max=vocab.N_AA_TYPES - 1),
          "chain_id": fd["chain_id"], "res_index": fd["res_index"], "dt": 200.0}
    ex = data.build_training_example(fr, i=0, tau_frames=2, k=8)
    u = batching.union_collate([ex])
    net = PropagatorNet(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)
    u_sample = sample_ddpm_union(net, u["node_feats"], u["edge_index"],
                                 u["edge_feats"], u["tau"], u["batch"],
                                 sched, steps=5)
    R_next, t_next = f.apply_update(fr["R"][0], fr["t"][0], u_sample)
    assert torch.isfinite(R_next).all() and torch.isfinite(t_next).all()
