import torch
from lsmd import geometry as g
from lsmd import featurize as f
from lsmd import vocab


def _rand_frames(n):
    R = g.so3_exp(torch.randn(n, 3) * 0.5)
    t = torch.randn(n, 3) * 5.0
    return R, t


def test_frame_graph_shapes_and_edge_dim():
    R, t = _rand_frames(20)
    ei, ef = f.frame_graph(R, t, k=6)
    assert ei.shape == (2, 20 * 6)
    assert ef.shape == (20 * 6, 13)


def test_frame_graph_edges_are_invariant():
    R, t = _rand_frames(12)
    ei, ef = f.frame_graph(R, t, k=4)
    # Apply a global rigid transform; the SAME kNN topology, invariant feats.
    Rg = g.so3_exp(torch.tensor([[0.2, -0.4, 0.6]])).expand(12, 3, 3)
    tg = torch.tensor([3.0, -2.0, 1.0])
    R2 = Rg @ R
    t2 = (Rg @ t.unsqueeze(-1)).squeeze(-1) + tg
    ei2, ef2 = f.frame_graph(R2, t2, k=4)
    assert torch.equal(ei, ei2)            # topology unchanged by rigid motion
    assert torch.allclose(ef, ef2, atol=1e-4)


def test_frame_node_features_shape_and_layout():
    res_type = torch.tensor([0, 20, 7])      # ALA, UNK, GLY
    chain_id = torch.tensor([0, 0, 1])
    res_index = torch.tensor([0, 1, 2])
    nf = f.frame_node_features(res_type, chain_id, res_index)
    assert nf.shape == (3, 24)
    # first 21 columns are the one-hot block
    assert torch.allclose(nf[0, :vocab.N_AA_TYPES],
                          torch.nn.functional.one_hot(torch.tensor(0), 21).float())
    assert nf[1, vocab.UNK_INDEX].item() == 1.0
    # chain id sits at column 21
    assert nf[2, vocab.N_AA_TYPES].item() == 1.0
