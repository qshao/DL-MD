import torch

# ── Angle energy ──────────────────────────────────────────────────────────────

def angle_energy(t: torch.Tensor,
                 chain_id: torch.Tensor,
                 k_angle: float = 10.0,
                 theta0: float = 2.094) -> torch.Tensor:
    """Harmonic CA-CA-CA angle energy.

    Args:
        t:        [N, 3] CA positions.
        chain_id: [N] long chain assignment.
        k_angle:  force constant in kcal/mol/rad².
        theta0:   equilibrium angle in radians (2.094 rad = 120°).

    Returns:
        Scalar energy in kcal/mol.
    """
    E = t.new_zeros(())
    for c in chain_id.unique():
        mask = (chain_id == c).nonzero(as_tuple=True)[0]
        if mask.shape[0] < 3:
            continue
        pos = t[mask]                   # [nc, 3]
        v1 = pos[:-2] - pos[1:-1]      # [nc-2, 3]
        v2 = pos[2:]  - pos[1:-1]      # [nc-2, 3]
        norms1 = v1.norm(dim=-1).clamp_min(1e-8)
        norms2 = v2.norm(dim=-1).clamp_min(1e-8)
        cos_theta = (v1 * v2).sum(-1) / (norms1 * norms2)
        # Use atan2 for numerical stability at extreme angles
        sin_cross = torch.cross(v1, v2, dim=-1).norm(dim=-1) / (norms1 * norms2)
        theta = torch.atan2(sin_cross, cos_theta)
        E = E + (k_angle * (theta - theta0) ** 2).sum()
    return E
