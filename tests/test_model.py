import torch
from lsmd import model as m


def _dummy_inputs(n=6, node_dim=8, edge_dim=13):
    node_feats = torch.randn(n, node_dim)
    edge_index = torch.randint(0, n, (2, 20))
    edge_feats = torch.randn(20, edge_dim)
    return node_feats, edge_index, edge_feats


# ── tau_embedding ──────────────────────────────────────────────────────────────

def test_tau_embedding_scalar_shape():
    emb = m.tau_embedding(50, dim=16)
    assert emb.shape == (16,)


def test_tau_embedding_batched_shape():
    taus = torch.tensor([10.0, 50.0, 200.0])
    emb = m.tau_embedding(taus, dim=16)
    assert emb.shape == (3, 16)


def test_tau_embedding_varies_with_tau():
    emb10 = m.tau_embedding(10, dim=16)
    emb200 = m.tau_embedding(200, dim=16)
    assert not torch.allclose(emb10, emb200)


# ── FlowNet unbatched ──────────────────────────────────────────────────────────

def test_flownet_unbatched_shape():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    u_s = torch.randn(6, 6)
    v = net(u_s, torch.tensor(0.4), nf, ei, ef, tau=50)
    assert v.shape == (6, 6)


def test_flownet_batched_shape():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    B = 4
    u_s = torch.randn(B, 6, 6)
    tau_b = torch.tensor([10.0, 25.0, 50.0, 100.0])
    s_b = torch.rand(B)
    v = net(u_s, s_b, nf, ei, ef, tau=tau_b)
    assert v.shape == (B, 6, 6)


def test_batched_matches_unbatched():
    """Batched forward must produce the same result as B sequential unbatched calls."""
    torch.manual_seed(42)
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    net.eval()

    taus = [10, 50, 100]
    s_vals = [0.1, 0.5, 0.9]
    u_list = [torch.randn(6, 6) for _ in taus]

    # Unbatched reference
    v_ref = torch.stack([
        net(u, torch.tensor(s), nf, ei, ef, tau=tau)
        for u, s, tau in zip(u_list, s_vals, taus)
    ])  # [3, 6, 6]

    # Batched
    u_batch = torch.stack(u_list)               # [3, 6, 6]
    s_batch = torch.tensor(s_vals)              # [3]
    tau_batch = torch.tensor(taus, dtype=torch.float32)  # [3]
    v_batch = net(u_batch, s_batch, nf, ei, ef, tau=tau_batch)

    assert torch.allclose(v_ref, v_batch, atol=1e-5), \
        f"max diff: {(v_ref - v_batch).abs().max().item():.2e}"


def test_different_tau_gives_different_output():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    u_s = torch.randn(6, 6)
    s = torch.tensor(0.5)
    v10 = net(u_s, s, nf, ei, ef, tau=10)
    v200 = net(u_s, s, nf, ei, ef, tau=200)
    assert not torch.allclose(v10, v200)


# ── cfm_loss ───────────────────────────────────────────────────────────────────

def test_cfm_loss_unbatched():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    u_target = torch.randn(6, 6)
    loss = m.cfm_loss(net, u_target, nf, ei, ef, tau=50)
    assert loss.shape == ()
    assert loss.item() > 0


def test_cfm_loss_batched():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    B = 4
    u_target = torch.randn(B, 6, 6)
    tau_b = torch.tensor([10.0, 25.0, 50.0, 100.0])
    loss = m.cfm_loss(net, u_target, nf, ei, ef, tau=tau_b)
    assert loss.shape == ()
    assert loss.item() > 0


def test_cfm_can_overfit_constant_target():
    torch.manual_seed(0)
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=64, layers=2)
    u_target = torch.randn(6, 6)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = m.cfm_loss(net, u_target, nf, ei, ef, tau=50, sigma=0.1)
        loss.backward()
        opt.step()
    samples = m.sample(net, nf, ei, ef, K=8, tau=50, steps=50, sigma=0.1)
    assert samples.shape == (8, 6, 6)
    assert (samples.mean(0) - u_target).abs().mean() < 0.3


# ── sample ─────────────────────────────────────────────────────────────────────

def test_sample_shape():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    samples = m.sample(net, nf, ei, ef, K=8, tau=50, steps=20, sigma=0.2)
    assert samples.shape == (8, 6, 6)


def test_sampler_is_diverse():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    samples = m.sample(net, nf, ei, ef, K=8, tau=50, steps=20, sigma=0.2)
    assert samples.std(0).mean() > 0.0


# ── NoiseSchedule ──────────────────────────────────────────────────────────────

def test_noise_schedule_shape():
    sched = m.NoiseSchedule(T=100)
    for attr in ("alphas_bar", "sqrt_alphas_bar",
                 "sqrt_one_minus_alphas_bar", "posterior_variance"):
        assert getattr(sched, attr).shape == (101,), attr


def test_noise_schedule_values():
    sched = m.NoiseSchedule(T=100)
    # index 0 = ᾱ_0 = 1 (clean); index T = ᾱ_T ≈ 0 (fully noisy)
    assert abs(sched.alphas_bar[0].item() - 1.0) < 1e-5
    assert sched.alphas_bar[-1].item() < 0.01
    # sqrt_alphas_bar must be monotone decreasing
    assert (sched.sqrt_alphas_bar[1:] <= sched.sqrt_alphas_bar[:-1]).all()
    # all values non-negative
    assert (sched.posterior_variance >= 0).all()


# ── ddpm_loss ──────────────────────────────────────────────────────────────────

def test_ddpm_loss_unbatched():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    u_target = torch.randn(6, 6)
    loss = m.ddpm_loss(net, u_target, nf, ei, ef, tau=50, schedule=sched)
    assert loss.shape == ()
    assert loss.item() > 0


def test_ddpm_loss_batched():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    B = 4
    u_target = torch.randn(B, 6, 6)
    tau_b = torch.tensor([10.0, 25.0, 50.0, 100.0])
    loss = m.ddpm_loss(net, u_target, nf, ei, ef, tau=tau_b, schedule=sched)
    assert loss.shape == ()
    assert loss.item() > 0


def test_ddpm_loss_weighted():
    torch.manual_seed(7)
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    u_target = torch.randn(4, 6, 6)
    tau_b = torch.tensor([10.0, 25.0, 50.0, 100.0])

    # Uniform weights == no weights
    torch.manual_seed(0)
    loss_no_w = m.ddpm_loss(net, u_target, nf, ei, ef, tau=tau_b, schedule=sched)
    torch.manual_seed(0)
    loss_unif = m.ddpm_loss(net, u_target, nf, ei, ef, tau=tau_b, schedule=sched,
                             pair_weights=torch.ones(4))
    assert torch.isclose(loss_no_w, loss_unif, atol=1e-5)

    # Non-uniform weights give different result
    torch.manual_seed(0)
    loss_w = m.ddpm_loss(net, u_target, nf, ei, ef, tau=tau_b, schedule=sched,
                          pair_weights=torch.tensor([0.0, 0.0, 2.0, 2.0]))
    assert not torch.isclose(loss_no_w, loss_w, atol=1e-3)


def test_ddpm_can_overfit_constant_target():
    torch.manual_seed(0)
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=64, layers=2)
    sched = m.NoiseSchedule(T=20)   # small T → fewer noise levels → easier to overfit
    u_target = torch.randn(6, 6)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        m.ddpm_loss(net, u_target, nf, ei, ef, tau=50, schedule=sched).backward()
        opt.step()
    samples = m.sample_ddpm(net, nf, ei, ef, K=8, tau=50, schedule=sched,
                             steps=20, eta=1.0)
    assert samples.shape == (8, 6, 6)
    assert (samples.mean(0) - u_target).abs().mean() < 0.5


# ── sample_ddpm ────────────────────────────────────────────────────────────────

def test_sample_ddpm_shape():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    samples = m.sample_ddpm(net, nf, ei, ef, K=8, tau=50, schedule=sched, steps=10)
    assert samples.shape == (8, 6, 6)


def test_sample_ddpm_diverse():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    # eta=1 → stochastic reverse → samples must differ
    samples = m.sample_ddpm(net, nf, ei, ef, K=8, tau=50, schedule=sched,
                             steps=10, eta=1.0)
    assert samples.std(0).mean() > 0.0


def test_sample_ddpm_eta0_deterministic():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    torch.manual_seed(42)
    s1 = m.sample_ddpm(net, nf, ei, ef, K=4, tau=50, schedule=sched,
                        steps=10, eta=0.0)
    torch.manual_seed(42)
    s2 = m.sample_ddpm(net, nf, ei, ef, K=4, tau=50, schedule=sched,
                        steps=10, eta=0.0)
    assert torch.allclose(s1, s2)


# ── point_dim parameter ────────────────────────────────────────────────────────

def test_flownet_point_dim_3():
    nf, ei, ef = _dummy_inputs(n=6, node_dim=8, edge_dim=4)
    net = m.FlowNet(node_dim=8, edge_dim=4, hidden=32, layers=2, point_dim=3)
    u_s = torch.randn(6, 3)
    out = net(u_s, torch.tensor(0.4), nf, ei, ef, tau=50)
    assert out.shape == (6, 3)


def test_sample_ddpm_point_dim_3():
    nf, ei, ef = _dummy_inputs(n=6, node_dim=8, edge_dim=4)
    net = m.FlowNet(node_dim=8, edge_dim=4, hidden=32, layers=2, point_dim=3)
    sched = m.NoiseSchedule(T=50)
    samples = m.sample_ddpm(net, nf, ei, ef, K=5, tau=50, schedule=sched, steps=10)
    assert samples.shape == (5, 6, 3)


def test_default_point_dim_is_6():
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=16, layers=1)
    assert net.point_dim == 6
