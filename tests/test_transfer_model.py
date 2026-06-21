import torch
from lsmd import batching
from lsmd.model import NoiseSchedule
from lsmd import transfer_model as tm


def _toy_graph(n, k=4, tau=100.0):
    R_t = torch.randn(n, 24)
    e = n * k
    return {
        "node_feats": R_t,
        "edge_index": torch.randint(0, n, (2, e)),
        "edge_feats": torch.randn(e, 13),
        "u_target": torch.randn(n, 6),
        "tau": tau,
    }


def test_forward_shape():
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    net = tm.PropagatorNet(hidden=32, layers=2)
    s = torch.rand(2)
    out = net(u["u_target"], s, u["node_feats"], u["edge_index"],
              u["edge_feats"], u["tau"], u["batch"])
    assert out.shape == (13, 6)


def test_graphs_do_not_interact():
    # Output on graph-0 nodes is identical whether or not graph-1 is in the batch
    torch.manual_seed(0)
    g0 = _toy_graph(5, tau=100.0)
    g1 = _toy_graph(8, tau=100.0)
    net = tm.PropagatorNet(hidden=32, layers=2).eval()

    solo = batching.union_collate([g0])
    pair = batching.union_collate([g0, g1])
    with torch.no_grad():
        out_solo = net(solo["u_target"], torch.tensor([0.3]),
                       solo["node_feats"], solo["edge_index"],
                       solo["edge_feats"], solo["tau"], solo["batch"])
        out_pair = net(pair["u_target"], torch.tensor([0.3, 0.7]),
                       pair["node_feats"], pair["edge_index"],
                       pair["edge_feats"], pair["tau"], pair["batch"])
    assert torch.allclose(out_solo, out_pair[:5], atol=1e-5)


def test_ddpm_loss_is_finite_scalar():
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    net = tm.PropagatorNet(hidden=32, layers=2)
    sched = NoiseSchedule(T=50)
    loss = tm.ddpm_loss_union(net, u["u_target"], u["node_feats"],
                              u["edge_index"], u["edge_feats"], u["tau"],
                              u["batch"], sched)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_sampler_shape_single_graph():
    g0 = _toy_graph(7)
    u = batching.union_collate([g0])
    net = tm.PropagatorNet(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)
    out = tm.sample_ddpm_union(net, u["node_feats"], u["edge_index"],
                               u["edge_feats"], u["tau"], u["batch"], sched,
                               steps=5)
    assert out.shape == (7, 6)
