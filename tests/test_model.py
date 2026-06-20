import torch
from lsmd import model as m


def _dummy_inputs(n=6, node_dim=8, edge_dim=13):
    node_feats = torch.randn(n, node_dim)
    edge_index = torch.randint(0, n, (2, 20))
    edge_feats = torch.randn(20, edge_dim)
    return node_feats, edge_index, edge_feats


def test_flownet_output_shape():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    u_s = torch.randn(6, 6)
    s = torch.tensor(0.4)
    v = net(u_s, s, nf, ei, ef)
    assert v.shape == (6, 6)


def test_cfm_can_overfit_constant_target():
    torch.manual_seed(0)
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=64, layers=2)
    u_target = torch.randn(6, 6)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = m.cfm_loss(net, u_target, nf, ei, ef, sigma=0.1)
        loss.backward()
        opt.step()
    samples = m.sample(net, nf, ei, ef, K=8, steps=50, sigma=0.1)
    assert samples.shape == (8, 6, 6)
    # sampled mean should be near the target it was trained to reproduce
    assert (samples.mean(0) - u_target).abs().mean() < 0.3


def test_sampler_is_diverse():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    samples = m.sample(net, nf, ei, ef, K=8, steps=20, sigma=0.2)
    spread = samples.std(0).mean()
    assert spread > 0.0
