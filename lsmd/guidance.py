"""C2 differentiable energy guidance for the transferable sampler (Plan 4).

Reconstruction guidance: at each reverse step the model's clean-update estimate
x0_hat is nudged down the gradient of the chain-aware geometric energy, enforcing
physics during generation. gamma=0 recovers the plain sampler exactly.
"""
import torch

from lsmd.physics_loss import geometric_penalty


def guidance_step(u0_hat, R_cur, t_cur, global_chain, scale, gamma, *,
                  rama_pot=None, **pen_kw):
    """One reconstruction-guidance nudge on the clean-update estimate."""
    if gamma == 0.0:
        return u0_hat
    with torch.enable_grad():
        x0 = u0_hat.detach().requires_grad_(True)
        pen = geometric_penalty(R_cur, t_cur, x0 * scale.to(x0), global_chain,
                                rama_pot=rama_pot, **pen_kw)
        grad = torch.autograd.grad(pen, x0)[0]
    return u0_hat - gamma * grad


@torch.no_grad()
def sample_ddpm_union_guided(net, node_feats, edge_index, edge_feats, tau, batch,
                             schedule, R_cur, t_cur, global_chain, scale, *,
                             steps=50, eta=1.0, sigma_init=1.0, gamma=0.0,
                             rama_pot=None):
    """Plan-1 reverse sampler with per-step energy guidance on x0_hat.

    gamma=0 reproduces sample_ddpm_union exactly under identical RNG.
    """
    T = schedule.T
    N = node_feats.shape[0]
    device, dtype = node_feats.device, node_feats.dtype
    u = torch.randn(N, net.point_dim, device=device, dtype=dtype) * sigma_init

    t_full = torch.round(torch.linspace(T - 1, 0, steps + 1, device=device)).long().clamp(0, T - 1)
    for i in range(steps):
        t = t_full[i].item()
        t_prev = t_full[i + 1].item()
        s = torch.full((tau.shape[0],), t / T, dtype=dtype, device=device)
        eps_pred = net(u, s, node_feats, edge_index, edge_feats, tau, batch)

        sqrt_ab_t = schedule.sqrt_alphas_bar[t].to(dtype)
        sqrt_1mab_t = schedule.sqrt_one_minus_alphas_bar[t].to(dtype)
        ab_prev = schedule.alphas_bar[t_prev].to(dtype)
        ab_t = schedule.alphas_bar[t].to(dtype)

        u0_hat = (u - sqrt_1mab_t * eps_pred) / sqrt_ab_t.clamp_min(1e-8)
        u0_hat = guidance_step(u0_hat, R_cur, t_cur, global_chain, scale, gamma,
                               rama_pot=rama_pot)
        pv = (1 - ab_prev) / (1 - ab_t).clamp_min(1e-8) * (1 - ab_t / ab_prev.clamp_min(1e-8))
        sigma_t = eta * pv.clamp_min(0.0).sqrt()
        dir_coeff = (1 - ab_prev - sigma_t ** 2).clamp_min(0.0).sqrt()
        z = torch.randn_like(u) if t_prev > 0 else torch.zeros_like(u)
        u = ab_prev.sqrt() * u0_hat + dir_coeff * eps_pred + sigma_t * z
    return u
