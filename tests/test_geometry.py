import torch
from lsmd import geometry as g


def test_build_frames_orthonormal():
    N = torch.tensor([[-0.5, 1.4, 0.0]])
    CA = torch.tensor([[0.0, 0.0, 0.0]])
    C = torch.tensor([[1.5, 0.0, 0.0]])
    R, t = g.build_frames(N, CA, C)
    # columns orthonormal
    gram = R[0].T @ R[0]
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)
    assert torch.isclose(torch.det(R[0]), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(t[0], CA[0])
    # dtype checks
    assert R.dtype == torch.float32
    assert t.dtype == torch.float32
    # e1-direction assertion
    e1_expected = (C[0] - CA[0])
    e1_expected = e1_expected / e1_expected.norm()
    assert torch.allclose(R[0, :, 0], e1_expected, atol=1e-5)


def test_so3_exp_log_roundtrip():
    omega = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
    R = g.so3_exp(omega)
    omega2 = g.so3_log(R)
    assert torch.allclose(omega, omega2, atol=1e-5)


def test_compose_invert_identity():
    R, t = g.so3_exp(torch.tensor([[0.2, 0.1, -0.3]])), torch.tensor([[1.0, 2.0, 3.0]])
    Ri, ti = g.invert(R, t)
    Rc, tc = g.compose(R, t, Ri, ti)
    assert torch.allclose(Rc[0], torch.eye(3), atol=1e-5)
    assert torch.allclose(tc[0], torch.zeros(3), atol=1e-5)


def test_place_backbone_reproduces_frame():
    # placing ideal atoms then rebuilding the frame returns the same frame
    R = g.so3_exp(torch.tensor([[0.3, -0.1, 0.2]]))
    t = torch.tensor([[1.0, -2.0, 0.5]])
    atoms = g.place_backbone(R, t)  # [1,4,3] N,CA,C,O
    R2, t2 = g.build_frames(atoms[:, 0], atoms[:, 1], atoms[:, 2])
    assert torch.allclose(R, R2, atol=1e-4)
    assert torch.allclose(t, t2, atol=1e-4)


def test_kabsch_identity():
    X = torch.randn(10, 3)
    R, t = g.kabsch(X, X)
    assert torch.allclose(R, torch.eye(3), atol=1e-5)
    assert torch.allclose(t, torch.zeros(3), atol=1e-5)


def test_kabsch_recovers_known_transform():
    torch.manual_seed(0)
    Y = torch.randn(20, 3)
    # Build a known rotation via QR (proper rotation) and a translation
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    trans = torch.tensor([1.0, -2.0, 3.0])
    X = Y @ Q.T + trans                     # X is Y rotated+translated
    R, t = g.kabsch(X, Y)                    # align Y onto X
    Y_aligned = Y @ R.transpose(-1, -2) + t
    assert torch.allclose(Y_aligned, X, atol=1e-4)
    assert abs(torch.linalg.det(R).item() - 1.0) < 1e-4   # proper rotation


def test_kabsch_batched():
    torch.manual_seed(1)
    X = torch.randn(4, 15, 3)
    Y = torch.randn(4, 15, 3)
    R, t = g.kabsch(X, Y)
    assert R.shape == (4, 3, 3)
    assert t.shape == (4, 3)
    Y_aligned = Y @ R.transpose(-1, -2) + t.unsqueeze(-2)
    # alignment reduces RMSD vs unaligned
    rmsd_before = (X - Y).norm(dim=-1).mean()
    rmsd_after = (X - Y_aligned).norm(dim=-1).mean()
    assert rmsd_after <= rmsd_before + 1e-5
