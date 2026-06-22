import inspect
import pytest
from lsmd import transfer_eval as te


def test_rollout_has_noether_parameter():
    sig = inspect.signature(te.rollout)
    assert "noether" in sig.parameters
    assert sig.parameters["noether"].default is False


def test_rollout_noether_does_not_change_shape():
    """rollout with noether=True returns same shape as noether=False."""
    import os, torch
    from lsmd import transfer_eval as te
    CKPT = "checkpoints/v2_256h_90k.pt"
    if not os.path.exists(CKPT):
        pytest.skip("checkpoint not available")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    net, sched, norm = te.load_checkpoint(ckpt, device="cpu")
    N = 8
    t0 = torch.randn(N, 3) * 5
    R0 = torch.eye(3).unsqueeze(0).expand(N, -1, -1).clone()
    res_type = torch.zeros(N, dtype=torch.long)
    chain_id = torch.zeros(N, dtype=torch.long)
    res_index = torch.arange(N)
    traj = te.rollout(net, sched, norm, R0, t0, res_type, chain_id, res_index,
                      steps=2, tau_ps=2000.0, k=4, diff_steps=2,
                      noether=True, device="cpu")
    assert traj.shape == (3, N, 3)
