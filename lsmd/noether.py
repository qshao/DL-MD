import torch


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched 3-D cross product a × b; last dim must be 3."""
    return torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], dim=-1)


def noether_project(t_old: torch.Tensor,
                    t_new: torch.Tensor,
                    chain_id: torch.Tensor) -> torch.Tensor:
    """Remove net linear and angular momentum per chain from a displacement.

    Applied after apply_update + bond_constraint inside rollout() to eliminate
    the spurious COM drift and rigid-body rotation the diffusion model adds.

    Args:
        t_old:    [N, 3] CA positions before this step (= traj[-1] in rollout).
        t_new:    [N, 3] CA positions after apply_update + bond_constraint.
        chain_id: [N] long, chain assignment (0-indexed, contiguous).

    Returns:
        [N, 3] corrected CA positions.
    """
    delta = (t_new - t_old).clone()

    for c in chain_id.unique():
        mask = (chain_id == c)
        d = delta[mask]               # [nc, 3]
        r = t_old[mask]               # [nc, 3]

        # Step 1 — zero linear momentum: subtract mean displacement
        d = d - d.mean(dim=0)

        # Step 2 — zero angular momentum
        centroid = r.mean(dim=0)
        r_c = r - centroid                                # [nc, 3]
        L = _cross(r_c, d).sum(dim=0)                    # [3] angular momentum
        # Inertia tensor: I = sum_i(|r_i|^2 * I_3 - r_i r_i^T)
        r2 = (r_c * r_c).sum(dim=-1)                     # [nc]
        I = (r2.sum() * torch.eye(3, device=d.device, dtype=d.dtype)
             - r_c.T @ r_c)                              # [3, 3]
        omega = torch.linalg.pinv(I) @ L                 # [3]
        nc = d.shape[0]
        d = d - _cross(omega.unsqueeze(0).expand(nc, -1), r_c)

        delta[mask] = d

    return t_old + delta
