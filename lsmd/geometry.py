import torch

# Ideal local backbone coordinates (Angstrom) in the residue frame
# (CA at origin; e1 along CA->C; N in +e2 half-plane). O is approximate.
IDEAL_LOCAL = {
    "N":  [-0.522, 1.362, 0.0],
    "CA": [0.0, 0.0, 0.0],
    "C":  [1.525, 0.0, 0.0],
    "O":  [2.158, -1.056, 0.0],
}
_ATOM_ORDER = ["N", "CA", "C", "O"]


def _normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def build_frames(N, CA, C):
    e1 = _normalize(C - CA)
    u = N - CA
    e2 = _normalize(u - (u * e1).sum(-1, keepdim=True) * e1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)  # columns
    return R, CA.clone()


def so3_exp(omega):
    theta = omega.norm(dim=-1, keepdim=True)
    small = theta < 1e-6
    axis = omega / theta.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], -1),
        torch.stack([z, zero, -x], -1),
        torch.stack([-y, x, zero], -1),
    ], -2)
    th = theta[..., None]
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(K.shape)
    R = eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    # near zero, first-order fallback
    K0 = torch.stack([
        torch.stack([zero, -omega[..., 2], omega[..., 1]], -1),
        torch.stack([omega[..., 2], zero, -omega[..., 0]], -1),
        torch.stack([-omega[..., 1], omega[..., 0], zero], -1),
    ], -2)
    R_small = eye + K0
    return torch.where(small[..., None], R_small, R)


def so3_log(R):
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1) / 2).clamp(-1.0, 1.0)
    theta = torch.acos(cos)[..., None]
    vee = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], -1)
    small = theta < 1e-6
    omega = torch.where(small, 0.5 * vee, (theta / (2 * torch.sin(theta).clamp_min(1e-8))) * vee)
    return omega


def compose(R1, t1, R2, t2):
    R = R1 @ R2
    t = (R1 @ t2.unsqueeze(-1)).squeeze(-1) + t1
    return R, t


def invert(R, t):
    Rinv = R.transpose(-1, -2)
    tinv = -(Rinv @ t.unsqueeze(-1)).squeeze(-1)
    return Rinv, tinv


def place_backbone(R, t):
    local = torch.tensor([IDEAL_LOCAL[a] for a in _ATOM_ORDER],
                         device=R.device, dtype=R.dtype)  # [4,3]
    # global = t + R @ local
    placed = (R.unsqueeze(-3) @ local.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(-2)
    return placed  # [...,4,3]


def kabsch(X, Y):
    """Rigid transform aligning Y onto X (minimizes ‖Y@R.T + t − X‖).

    Args:
        X: target points [P, 3] or [B, P, 3]
        Y: source points [P, 3] or [B, P, 3]

    Returns:
        (R, t): R [3,3] or [B,3,3] (proper rotation, det=+1),
                t [3] or [B,3] s.t. Y @ R.transpose(-1,-2) + t ≈ X.
    """
    muX = X.mean(dim=-2, keepdim=True)            # [...,1,3]
    muY = Y.mean(dim=-2, keepdim=True)
    Xc = X - muX
    Yc = Y - muY
    H = Yc.transpose(-1, -2) @ Xc                 # [...,3,3]
    U, _, Vt = torch.linalg.svd(H)
    V = Vt.transpose(-1, -2)
    d = torch.linalg.det(V @ U.transpose(-1, -2))  # [...] sign for proper rotation
    D = torch.eye(3, device=X.device, dtype=X.dtype).expand_as(H).clone()
    D[..., 2, 2] = d
    R = V @ D @ U.transpose(-1, -2)               # [...,3,3]
    # t = muX - muY @ R.T
    t = muX.squeeze(-2) - (muY @ R.transpose(-1, -2)).squeeze(-2)
    return R, t
