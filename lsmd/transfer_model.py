"""Union-graph propagator network for transferable protein dynamics.

Operates on a flat disjoint-union graph ([ΣN, ...] nodes + a batch vector),
so multiple proteins of different sizes train in one forward pass. Predicts a
per-residue SE(3) local update via DDPM epsilon-prediction, conditioned on the
current-state graph and physical lag tau.
"""
import torch
import torch.nn as nn
from lsmd.model import tau_embedding


def _scatter_mean(src, index, dim_size):
    """Mean of `src` rows grouped by `index` (0..dim_size-1)."""
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp_min(1.0).reshape(-1, *([1] * (src.dim() - 1)))


class UnionMessageLayer(nn.Module):
    """Flat message-passing layer over a union edge_index ([ΣN, H] nodes)."""

    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h, edge_index, edge_feats):
        src, dst = edge_index                       # [E]
        N, H = h.shape
        msg = self.msg(torch.cat([h[src], h[dst], edge_feats], dim=-1))  # [E,H]
        agg = torch.zeros(N, H, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(N, 1, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        return h + self.upd(torch.cat([h, agg], dim=-1))


class PropagatorNet(nn.Module):
    """DDPM epsilon-predictor over a union graph.

    temp_emb_dim > 0 conditions the model on simulation temperature (K), which
    is crucial when training across temperatures (mdCATH 320-450K): the model
    must predict larger fluctuations at higher T.  Default 8 for new models;
    set to 0 to reproduce the original architecture (for loading old checkpoints).
    """

    def __init__(self, node_dim=24, edge_dim=13, hidden=128, layers=4,
                 tau_emb_dim=16, point_dim=6, temp_emb_dim=8):
        super().__init__()
        self.tau_emb_dim = tau_emb_dim
        self.temp_emb_dim = temp_emb_dim
        self.point_dim = point_dim
        in_dim = node_dim + point_dim + 1 + tau_emb_dim + temp_emb_dim
        self.embed = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(
            [UnionMessageLayer(hidden, edge_dim) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, point_dim),
        )

    def forward(self, u, s, node_feats, edge_index, edge_feats, tau, batch,
                temp_K=None):
        """Predict per-node epsilon.

        Args:
            u:          [ΣN, point_dim] noisy update.
            s:          [G] flow-time per graph.
            node_feats: [ΣN, node_dim]
            edge_index: [2, ΣE] (union indices)
            edge_feats: [ΣE, edge_dim]
            tau:        [G] physical lag (ps) per graph.
            batch:      [ΣN] long, node→graph.
            temp_K:     [G] simulation temperature in Kelvin (optional; ignored
                        when temp_emb_dim == 0).

        Returns:
            [ΣN, point_dim]
        """
        batch = batch.to(node_feats.device)
        s = torch.as_tensor(s, dtype=u.dtype, device=u.device)
        s_nodes = s[batch].unsqueeze(-1)                       # [ΣN,1]
        tau_emb = tau_embedding(tau, dim=self.tau_emb_dim,
                                device=u.device, dtype=u.dtype)  # [G, tau_dim]
        tau_nodes = tau_emb[batch]                             # [ΣN, tau_dim]
        parts = [node_feats, u, s_nodes, tau_nodes]
        if self.temp_emb_dim > 0:
            if temp_K is None:
                temp_K = torch.full((tau.shape[0],), 300.0,
                                    device=u.device, dtype=u.dtype)
            temp_emb = tau_embedding(temp_K / 300.0, dim=self.temp_emb_dim,
                                     device=u.device, dtype=u.dtype)
            parts.append(temp_emb[batch])
        h = self.embed(torch.cat(parts, dim=-1))
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)
        return self.head(h)


def ddpm_loss_union(net, u_target, node_feats, edge_index, edge_feats, tau,
                    batch, schedule, graph_weights=None, temp_K=None):
    """DDPM epsilon-prediction loss over a union batch.

    Each graph gets its own noise level; per-graph node-mean losses are then
    averaged (optionally weighted) so large proteins don't dominate.
    """
    batch = batch.to(u_target.device)
    G = tau.shape[0]
    T = schedule.T
    t_min = max(1, T // 20)
    t_idx = torch.randint(t_min, T + 1, (G,), device=u_target.device)   # [G]
    t_nodes = t_idx[batch]                                              # [ΣN]
    eps = torch.randn_like(u_target)

    sqrt_ab = schedule.sqrt_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    sqrt_1mab = schedule.sqrt_one_minus_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    noisy = sqrt_ab * u_target + sqrt_1mab * eps

    s = (t_idx.float() / T).to(u_target.dtype)                         # [G]
    pred = net(noisy, s, node_feats, edge_index, edge_feats, tau, batch,
               temp_K=temp_K)

    node_se = ((pred - eps) ** 2).mean(dim=-1)                         # [ΣN]
    per_graph = _scatter_mean(node_se, batch, G)                       # [G]
    if graph_weights is not None:
        w = graph_weights.to(per_graph)
        return (w * per_graph).mean()
    return per_graph.mean()


@torch.no_grad()
def sample_ddpm_union(net, node_feats, edge_index, edge_feats, tau, batch,
                      schedule, steps=50, eta=1.0, sigma_init=1.0, temp_K=None,
                      guidance_fn=None):
    """Reverse-diffusion sampler over a union graph (one update per node).

    eta=1.0 → stochastic DDPM; eta=0.0 → deterministic DDIM.  Use eta=0 with
    steps=10-20 for fast rollout (10-20x speedup with little quality loss).

    guidance_fn: optional callable(u0_hat) -> u0_hat_guided. Applied to the
        Tweedie x0-estimate at every denoising step (C2 guidance). The guided
        u0_hat is used consistently: eps is recomputed from it so the DDPM
        posterior remains well-formed.
    """
    T = schedule.T
    N = node_feats.shape[0]
    device, dtype = node_feats.device, node_feats.dtype
    u = torch.randn(N, net.point_dim, device=device, dtype=dtype) * sigma_init

    t_full = torch.round(torch.linspace(T - 1, 0, steps + 1, device=device)).long().clamp(0, T - 1)
    for i in range(steps):
        t = t_full[i].item()
        t_prev = t_full[i + 1].item()
        s = torch.full((tau.shape[0],), t / T, dtype=dtype, device=device)   # [G]
        eps_pred = net(u, s, node_feats, edge_index, edge_feats, tau, batch,
                       temp_K=temp_K)

        sqrt_ab_t = schedule.sqrt_alphas_bar[t].to(dtype)
        sqrt_1mab_t = schedule.sqrt_one_minus_alphas_bar[t].to(dtype)
        ab_prev = schedule.alphas_bar[t_prev].to(dtype)
        ab_t = schedule.alphas_bar[t].to(dtype)

        u0_hat = (u - sqrt_1mab_t * eps_pred) / sqrt_ab_t.clamp_min(1e-8)
        if guidance_fn is not None:
            u0_hat = guidance_fn(u0_hat)
            # Recompute eps from guided u0_hat so posterior stays consistent.
            eps_pred = (u - sqrt_ab_t * u0_hat) / sqrt_1mab_t.clamp_min(1e-8)
        pv = (1 - ab_prev) / (1 - ab_t).clamp_min(1e-8) * (1 - ab_t / ab_prev.clamp_min(1e-8))
        sigma_t = eta * pv.clamp_min(0.0).sqrt()
        dir_coeff = (1 - ab_prev - sigma_t ** 2).clamp_min(0.0).sqrt()
        z = torch.randn_like(u) if t_prev > 0 else torch.zeros_like(u)
        u = ab_prev.sqrt() * u0_hat + dir_coeff * eps_pred + sigma_t * z
    return u


class StructuralEncoder(nn.Module):
    """Per-node context from the static graph + lag tau. Run once per step.

    Carries the expensive L message-passing layers. Independent of the noisy
    update u and the diffusion flow-time s, so its output can be cached across
    all reverse-diffusion steps of one propagation step.
    """

    def __init__(self, node_dim=24, edge_dim=13, hidden=128, layers=4,
                 tau_emb_dim=16, temp_emb_dim=8):
        super().__init__()
        self.tau_emb_dim = tau_emb_dim
        self.temp_emb_dim = temp_emb_dim
        self.embed = nn.Linear(node_dim + tau_emb_dim + temp_emb_dim, hidden)
        self.layers = nn.ModuleList(
            [UnionMessageLayer(hidden, edge_dim) for _ in range(layers)]
        )

    def forward(self, node_feats, edge_index, edge_feats, tau, batch,
                temp_K=None):
        batch = batch.to(node_feats.device)
        tau_emb = tau_embedding(tau, dim=self.tau_emb_dim,
                                device=node_feats.device, dtype=node_feats.dtype)
        tau_nodes = tau_emb[batch]
        parts = [node_feats, tau_nodes]
        if self.temp_emb_dim > 0:
            if temp_K is None:
                temp_K = torch.full((tau.shape[0],), 300.0,
                                    device=node_feats.device, dtype=node_feats.dtype)
            temp_emb = tau_embedding(temp_K / 300.0, dim=self.temp_emb_dim,
                                     device=node_feats.device, dtype=node_feats.dtype)
            parts.append(temp_emb[batch])
        h = self.embed(torch.cat(parts, dim=-1))
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)
        return h


class Denoiser(nn.Module):
    """Lightweight per-step epsilon predictor over cached context.

    Injects the noisy update u and diffusion flow-time s into the cached
    structural context, runs n_denoise_layers message layers (default 1; 0 = a
    pure per-node MLP), and predicts epsilon.
    """

    def __init__(self, hidden=128, edge_dim=13, point_dim=6, n_denoise_layers=1):
        super().__init__()
        self.point_dim = point_dim
        self.inject = nn.Linear(hidden + point_dim + 1, hidden)
        self.layers = nn.ModuleList(
            [UnionMessageLayer(hidden, edge_dim) for _ in range(n_denoise_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, point_dim),
        )

    def forward(self, u, s, context, edge_index, edge_feats, batch):
        batch = batch.to(context.device)
        s = torch.as_tensor(s, dtype=u.dtype, device=u.device)
        s_nodes = s[batch].unsqueeze(-1)
        h = self.inject(torch.cat([context, u, s_nodes], dim=-1))
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)
        return self.head(h)


class CachedPropagator(nn.Module):
    """Encoder + denoiser propagator with a cacheable structural pass.

    `forward` has the same signature as PropagatorNet.forward (drop-in for
    ddpm_loss_union). `encode`/`denoise` expose the split so a sampler can run
    the expensive encoder once and the cheap denoiser per reverse step.
    """

    def __init__(self, node_dim=24, edge_dim=13, hidden=128, layers=4,
                 tau_emb_dim=16, point_dim=6, n_denoise_layers=1, temp_emb_dim=8):
        super().__init__()
        self.point_dim = point_dim
        self.encoder = StructuralEncoder(node_dim, edge_dim, hidden, layers,
                                        tau_emb_dim, temp_emb_dim)
        self.denoiser = Denoiser(hidden, edge_dim, point_dim, n_denoise_layers)

    def encode(self, node_feats, edge_index, edge_feats, tau, batch, temp_K=None):
        return self.encoder(node_feats, edge_index, edge_feats, tau, batch,
                            temp_K=temp_K)

    def denoise(self, u, s, context, edge_index, edge_feats, batch):
        return self.denoiser(u, s, context, edge_index, edge_feats, batch)

    def forward(self, u, s, node_feats, edge_index, edge_feats, tau, batch,
                temp_K=None):
        context = self.encode(node_feats, edge_index, edge_feats, tau, batch,
                              temp_K=temp_K)
        return self.denoise(u, s, context, edge_index, edge_feats, batch)


@torch.no_grad()
def sample_ddpm_union_cached(net, node_feats, edge_index, edge_feats, tau, batch,
                             schedule, steps=50, eta=1.0, sigma_init=1.0,
                             temp_K=None, guidance_fn=None):
    """Reverse-diffusion sampler that encodes the static graph once.

    `net` must expose `encode`/`denoise` (a CachedPropagator). Identical reverse
    math to sample_ddpm_union; the only difference is the structural context is
    computed once and reused across all reverse steps. eta=0 -> deterministic
    DDIM (use with a small `steps` for fast rollout).

    guidance_fn: same interface as in sample_ddpm_union.
    """
    T = schedule.T
    N = node_feats.shape[0]
    device, dtype = node_feats.device, node_feats.dtype
    context = net.encode(node_feats, edge_index, edge_feats, tau, batch,
                         temp_K=temp_K)
    u = torch.randn(N, net.point_dim, device=device, dtype=dtype) * sigma_init

    t_full = torch.round(torch.linspace(T - 1, 0, steps + 1, device=device)).long().clamp(0, T - 1)
    for i in range(steps):
        t = t_full[i].item()
        t_prev = t_full[i + 1].item()
        s = torch.full((tau.shape[0],), t / T, dtype=dtype, device=device)
        eps_pred = net.denoise(u, s, context, edge_index, edge_feats, batch)

        sqrt_ab_t = schedule.sqrt_alphas_bar[t].to(dtype)
        sqrt_1mab_t = schedule.sqrt_one_minus_alphas_bar[t].to(dtype)
        ab_prev = schedule.alphas_bar[t_prev].to(dtype)
        ab_t = schedule.alphas_bar[t].to(dtype)

        u0_hat = (u - sqrt_1mab_t * eps_pred) / sqrt_ab_t.clamp_min(1e-8)
        if guidance_fn is not None:
            u0_hat = guidance_fn(u0_hat)
            eps_pred = (u - sqrt_ab_t * u0_hat) / sqrt_1mab_t.clamp_min(1e-8)
        pv = (1 - ab_prev) / (1 - ab_t).clamp_min(1e-8) * (1 - ab_t / ab_prev.clamp_min(1e-8))
        sigma_t = eta * pv.clamp_min(0.0).sqrt()
        dir_coeff = (1 - ab_prev - sigma_t ** 2).clamp_min(0.0).sqrt()
        z = torch.randn_like(u) if t_prev > 0 else torch.zeros_like(u)
        u = ab_prev.sqrt() * u0_hat + dir_coeff * eps_pred + sigma_t * z
    return u
