import inspect
import torch
from lsmd import transfer_train as tt


def test_train_exposes_phase3_kwargs():
    params = inspect.signature(tt.train).parameters
    for name in ["energy_ckpt", "lam_energy", "lam_fdt", "phys_warmup", "w_hi", "w_lo"]:
        assert name in params


def test_collate_physics_carries_res_type_and_targets():
    # Two tiny examples; verify the new keys are present and shaped correctly.
    def ex(N, gi_res):
        return {
            "R_cur": torch.eye(3).expand(N, 3, 3).contiguous(),
            "t_cur": torch.randn(N, 3),
            "chain_id": torch.zeros(N, dtype=torch.long),
            "res_type": gi_res,
            "u_cut": 1.5,
            "sigma_md_tau": 0.04,
        }
    from lsmd.physics_loss import collate_physics
    examples = [ex(4, torch.zeros(4, dtype=torch.long)),
                ex(3, torch.ones(3, dtype=torch.long))]
    out = collate_physics(examples)
    assert out["res_type"].shape == (7,)
    assert out["chain_id"].shape == (7,)
    assert out["u_cut"].shape == (2,) and out["sigma_md_tau"].shape == (2,)
    assert torch.allclose(out["u_cut"], torch.tensor([1.5, 1.5]))
