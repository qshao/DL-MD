import math
import torch
from lsmd.learned_energy import LearnedCGEnergy, score_matching_loss, inverse_density_weights
from lsmd import cg_energy as cge


def _toy_protein(seed=0):
    g = torch.Generator().manual_seed(seed)
    N = 12
    t = torch.randn(N, 3, generator=g) * 5.0
    res_type = torch.randint(0, 20, (N,), generator=g)
    chain_id = torch.zeros(N, dtype=torch.long)
    return t, res_type, chain_id


def test_init_matches_cg_energy_defaults():
    t, rt, cid = _toy_protein()
    e = LearnedCGEnergy()
    got = e(t, rt, cid)
    ref = cge.total_cg_energy(t, rt, cid)   # default w=1, k_angle=10, eps=0.3
    assert torch.allclose(got, ref, atol=1e-4)


def test_params_and_position_grads_flow():
    t, rt, cid = _toy_protein()
    e = LearnedCGEnergy()
    t = t.requires_grad_(True)
    out = e(t, rt, cid)
    out.backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()
    assert all(p.grad is not None for p in e.parameters())


def test_save_load_roundtrip(tmp_path):
    e = LearnedCGEnergy()
    with torch.no_grad():
        e.log_alpha_mj += 0.5
    p = tmp_path / "energy.pt"
    e.save(str(p))
    e2 = LearnedCGEnergy.load(str(p))
    assert torch.allclose(e2.log_alpha_mj, e.log_alpha_mj)


def test_score_matching_loss_finite_and_differentiable():
    t, rt, cid = _toy_protein(seed=1)
    e = LearnedCGEnergy()
    torch.manual_seed(0)
    loss = score_matching_loss(e, t, rt, cid, sigma=0.5)
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in e.parameters())


def test_score_matching_reduces_on_harmonic_toy():
    # A 1-param harmonic energy U = 0.5*c*||t||^2 ; score-matching should drive
    # c toward the value implied by the (zero-centred) data + noise.
    torch.manual_seed(0)
    data = torch.randn(200, 4, 3)            # zero-mean cloud

    class Harmonic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_c = torch.nn.Parameter(torch.tensor(2.0))   # start too stiff
        def forward(self, t, res_type, chain_id):
            return 0.5 * self.log_c.exp() * (t ** 2).sum()

    h = Harmonic()
    opt = torch.optim.Adam(h.parameters(), lr=0.05)
    rt = torch.zeros(4, dtype=torch.long); cid = torch.zeros(4, dtype=torch.long)
    first = None
    for step in range(300):
        opt.zero_grad()
        i = torch.randint(0, data.shape[0], ()).item()
        loss = score_matching_loss(h, data[i], rt, cid, sigma=0.5, kT=1.0)
        loss.backward(); opt.step()
        if step == 0:
            first = float(loss)
    # averaged loss should be well below the initial mis-specified loss
    avg_last = sum(float(score_matching_loss(h, data[j], rt, cid, sigma=0.5, kT=1.0))
                   for j in range(20)) / 20
    assert avg_last < first


def test_inverse_density_weights_upweight_sparse():
    # 100 points in a dense cluster + 5 sparse outliers
    dense = torch.zeros(100, 2)
    sparse = torch.tensor([[10.0, 10.0]]).repeat(5, 1) + torch.randn(5, 2) * 0.01
    cv = torch.cat([dense, sparse], dim=0)
    w = inverse_density_weights(cv, bins=20, clip=50.0)
    assert w.shape == (105,)
    assert w[100:].mean() > w[:100].mean()      # sparse outliers up-weighted
    assert (w >= 1.0 / 50.0).all() and (w <= 50.0).all()
