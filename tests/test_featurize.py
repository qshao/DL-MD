import torch
from lsmd import geometry as g
from lsmd import featurize as f


def _rand_frames(n):
    R = g.so3_exp(torch.randn(n, 3) * 0.5)
    t = torch.randn(n, 3)
    return R, t


def test_update_roundtrip():
    R_t, t_t = _rand_frames(5)
    R_f, t_f = _rand_frames(5)
    u = f.relative_update(R_t, t_t, R_f, t_f)
    R_f2, t_f2 = f.apply_update(R_t, t_t, u)
    assert torch.allclose(R_f, R_f2, atol=1e-4)
    assert torch.allclose(t_f, t_f2, atol=1e-4)


def test_update_is_invariant_to_global_transform():
    R_t, t_t = _rand_frames(5)
    R_f, t_f = _rand_frames(5)
    u = f.relative_update(R_t, t_t, R_f, t_f)
    # apply a global rotation+translation to both endpoints
    Rg = g.so3_exp(torch.tensor([[0.3, -0.5, 0.2]])).expand(5, 3, 3)
    tg = torch.tensor([1.0, 2.0, -3.0])
    R_t2, t_t2 = Rg @ R_t, (Rg @ t_t.unsqueeze(-1)).squeeze(-1) + tg
    R_f2, t_f2 = Rg @ R_f, (Rg @ t_f.unsqueeze(-1)).squeeze(-1) + tg
    u2 = f.relative_update(R_t2, t_t2, R_f2, t_f2)
    assert torch.allclose(u, u2, atol=1e-4)


def test_knn_graph_shape_and_neighbors():
    t = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0], [6.0, 0, 0]])
    ei = f.knn_graph(t, k=1)
    assert ei.shape[0] == 2
    # nearest neighbor of node 0 is node 1
    nbr_of_0 = ei[1][ei[0] == 0]
    assert 1 in nbr_of_0.tolist()


def test_edge_features_invariant():
    R, t = _rand_frames(6)
    ei = f.knn_graph(t, k=2)
    feats = f.edge_features(R, t, ei)
    Rg = g.so3_exp(torch.tensor([[0.1, 0.7, -0.2]])).expand(6, 3, 3)
    tg = torch.tensor([2.0, -1.0, 4.0])
    R2, t2 = Rg @ R, (Rg @ t.unsqueeze(-1)).squeeze(-1) + tg
    feats2 = f.edge_features(R2, t2, ei)
    assert torch.allclose(feats, feats2, atol=1e-4)


def test_ca_displacement_identical_is_zero():
    X = torch.randn(12, 3)
    d = f.ca_displacement(X, X)
    assert d.shape == (12, 3)
    assert d.abs().max().item() < 1e-5


def test_ca_displacement_pure_translation():
    X = torch.randn(8, 3)
    shift = torch.tensor([1.0, 2.0, -3.0])
    Y = X + shift
    d = f.ca_displacement(X, Y)
    # Kabsch removes the global translation → near-zero internal displacement
    assert d.abs().max().item() < 1e-4


def test_ca_displacement_rotation_invariant_norms():
    torch.manual_seed(0)
    X_i = torch.randn(15, 3)
    X_j = X_i + 0.1 * torch.randn(15, 3)        # small internal change
    d1 = f.ca_displacement(X_i, X_j)
    # Rotate BOTH frames by the same proper rotation Q
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    d2 = f.ca_displacement(X_i @ Q.T, X_j @ Q.T)
    # displacement is equivariant → per-node magnitudes are invariant
    assert torch.allclose(d1.norm(dim=-1), d2.norm(dim=-1), atol=1e-4)


def test_ca_displacement_batched():
    X_i = torch.randn(4, 10, 3)
    X_j = torch.randn(4, 10, 3)
    d = f.ca_displacement(X_i, X_j)
    assert d.shape == (4, 10, 3)


def test_ca_graph_shapes():
    X = torch.randn(20, 3)
    ei, ef = f.ca_graph(X, k=6)
    assert ei.shape[0] == 2
    assert ei.shape[1] == ef.shape[0]
    assert ef.shape[1] == 4
    assert ei.shape[1] == 20 * 6        # k edges per node
