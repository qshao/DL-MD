import torch
from lsmd import batching


def _toy_graph(n, e, f=24, p=6, de=13, tau=100.0):
    return {
        "node_feats": torch.randn(n, f),
        "edge_index": torch.randint(0, n, (2, e)),
        "edge_feats": torch.randn(e, de),
        "u_target": torch.randn(n, p),
        "tau": tau,
    }


def test_union_concatenates_and_offsets():
    g0 = _toy_graph(5, 8, tau=50.0)
    g1 = _toy_graph(7, 10, tau=200.0)
    u = batching.union_collate([g0, g1])
    assert u["node_feats"].shape == (12, 24)
    assert u["u_target"].shape == (12, 6)
    assert u["edge_feats"].shape == (18, 13)
    assert u["edge_index"].shape == (2, 18)
    # batch vector tags nodes by graph
    assert u["batch"].tolist() == [0] * 5 + [1] * 7
    assert torch.equal(u["tau"], torch.tensor([50.0, 200.0]))


def test_second_graph_edges_are_offset_into_union():
    g0 = _toy_graph(5, 8)
    g1 = _toy_graph(7, 10)
    u = batching.union_collate([g0, g1])
    # graph-1 edges must index nodes 5..11, never 0..4
    second = u["edge_index"][:, 8:]
    assert second.min().item() >= 5
    assert second.max().item() <= 11


def test_no_cross_graph_edges():
    g0 = _toy_graph(5, 8)
    g1 = _toy_graph(7, 10)
    u = batching.union_collate([g0, g1])
    src, dst = u["edge_index"]
    # every edge stays within one graph
    assert torch.equal(u["batch"][src], u["batch"][dst])
