"""Flow-matching graph network for long-stride protein MD.

All inputs and outputs live in the invariant delta-space, so a plain
graph net suffices — equivariance is guaranteed by construction.

Both FlowNet and MessageLayer support unbatched [N, ...] and batched
[B, N, ...] inputs via a shared forward path.
"""
import torch
import torch.nn as nn


def tau_embedding(tau, dim=16, device=None, dtype=torch.float32):
    """Log-sinusoidal embedding of lag time tau.

    Uses log(tau) so the embedding varies smoothly across orders of magnitude.
    Works for any tau at inference, including values not seen during training.

    Args:
        tau:    Scalar int/float/tensor, or [B] tensor for batched input.
        dim:    Embedding dimension (must be even).
        device: Target device.
        dtype:  Target dtype.

    Returns:
        [dim] for scalar input, [B, dim] for [B] input.
    """
    tau = torch.as_tensor(tau, dtype=dtype, device=device)
    scalar = tau.dim() == 0
    if scalar:
        tau = tau.unsqueeze(0)          # [1]
    log_tau = torch.log(tau.clamp_min(1.0))          # [B]
    half = dim // 2
    freqs = torch.arange(half, dtype=dtype, device=device) / half
    freqs = 10.0 ** freqs               # [half] — log-spaced frequencies
    args = log_tau.unsqueeze(-1) * freqs.unsqueeze(0)   # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
    return emb.squeeze(0) if scalar else emb


class MessageLayer(nn.Module):
    """Single message-passing layer with mean aggregation and residual update.

    Accepts both unbatched [N, H] and batched [B, N, H] node features.
    The graph topology (edge_index, edge_feats) is shared across the batch.
    """

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
        """
        Args:
            h:          [N, H] or [B, N, H]
            edge_index: [2, E]
            edge_feats: [E, edge_dim]
        Returns:
            same shape as h
        """
        src, dst = edge_index           # [E]
        unbatched = h.dim() == 2
        if unbatched:
            h = h.unsqueeze(0)          # [1, N, H]

        B, N, H = h.shape
        E = edge_index.shape[1]

        h_src = h[:, src]               # [B, E, H]
        h_dst = h[:, dst]               # [B, E, H]
        ef = edge_feats.unsqueeze(0).expand(B, -1, -1)  # [B, E, edge_dim]

        msg = self.msg(torch.cat([h_src, h_dst, ef], dim=-1))  # [B, E, H]

        # Scatter-sum messages to destination nodes
        agg = torch.zeros(B, N, H, device=h.device, dtype=h.dtype)
        idx = dst.unsqueeze(0).unsqueeze(-1).expand(B, E, H)
        agg.scatter_add_(1, idx, msg)

        # Degree normalization — same for all batch items (shared graph)
        deg = torch.zeros(N, 1, device=h.device, dtype=h.dtype)
        deg.scatter_add_(0, dst.unsqueeze(-1),
                         torch.ones(E, 1, device=h.device, dtype=h.dtype))
        agg = agg / deg.unsqueeze(0).clamp_min(1.0)    # [B, N, H]

        out = h + self.upd(torch.cat([h, agg], dim=-1))
        return out.squeeze(0) if unbatched else out


class FlowNet(nn.Module):
    """Conditional flow-matching graph network.

    Operates entirely in the invariant delta-space. Accepts both unbatched
    [N, point_dim] and batched [B, N, point_dim] inputs; the batch path is used during
    mini-batch training and for parallel sampling.
    """

    def __init__(self, node_dim, edge_dim, hidden=64, layers=3, tau_emb_dim=16, point_dim=6):
        super().__init__()
        self.tau_emb_dim = tau_emb_dim
        self.point_dim = point_dim
        # input: node features + u (point_dim) + flow-time s (1) + tau embedding
        self.embed = nn.Linear(node_dim + point_dim + 1 + tau_emb_dim, hidden)
        self.layers = nn.ModuleList(
            [MessageLayer(hidden, edge_dim) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, point_dim),
        )

    def forward(self, u_s, s, node_feats, edge_index, edge_feats, tau):
        """Predict velocity in delta-space conditioned on lag time tau.

        Args:
            u_s:        [N, point_dim] or [B, N, point_dim]
            s:          Scalar, or [B] tensor (one flow-time per batch item).
            node_feats: [N, node_dim]  (broadcast across batch)
            edge_index: [2, E]
            edge_feats: [E, edge_dim]
            tau:        Scalar or [B] tensor — lag time in frames.

        Returns:
            velocity:   Same shape as u_s.
        """
        unbatched = u_s.dim() == 2
        if unbatched:
            u_s = u_s.unsqueeze(0)      # [1, N, 6]

        B, N, _ = u_s.shape

        # Flow-time embedding: [B, N, 1]
        s_t = torch.as_tensor(s, dtype=u_s.dtype, device=u_s.device)
        if s_t.dim() == 0:
            s_col = s_t.reshape(1, 1, 1).expand(B, N, 1)
        else:
            s_col = s_t.reshape(B, 1, 1).expand(B, N, 1)

        # Tau embedding: [B, N, tau_emb_dim]
        tau_emb_raw = tau_embedding(tau, dim=self.tau_emb_dim,
                                    device=u_s.device, dtype=u_s.dtype)
        if tau_emb_raw.dim() == 1:      # scalar tau → [tau_emb_dim]
            tau_emb = tau_emb_raw.reshape(1, 1, -1).expand(B, N, -1)
        else:                           # batched tau → [B, tau_emb_dim]
            tau_emb = tau_emb_raw.unsqueeze(1).expand(B, N, -1)

        # Node features: [N, F] → [B, N, F]
        nf = node_feats.unsqueeze(0).expand(B, -1, -1)

        h = self.embed(torch.cat([nf, u_s, s_col, tau_emb], dim=-1))  # [B, N, hidden]
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)

        out = self.head(h)              # [B, N, 6]
        return out.squeeze(0) if unbatched else out


def cfm_loss(net, u_target, node_feats, edge_index, edge_feats, tau, sigma=0.1):
    """Conditional flow-matching (rectified-flow) loss.

    Supports both [N, 6] (single pair) and [B, N, 6] (mini-batch) targets.
    For batched inputs a separate flow-time s is sampled per batch item,
    giving better coverage of the flow path.

    Args:
        net:        FlowNet instance
        u_target:   [N, 6] or [B, N, 6]
        node_feats: [N, node_dim]
        edge_index: [2, E]
        edge_feats: [E, edge_dim]
        tau:        Scalar or [B] tensor — lag time(s) in frames.
        sigma:      Prior scale.

    Returns:
        loss: Scalar MSE.
    """
    u0 = torch.randn_like(u_target) * sigma
    batched = u_target.dim() == 3
    if batched:
        B = u_target.shape[0]
        s = torch.rand(B, device=u_target.device, dtype=u_target.dtype)    # [B]
        u_s = (1 - s[:, None, None]) * u0 + s[:, None, None] * u_target
    else:
        s = torch.rand((), device=u_target.device, dtype=u_target.dtype)
        u_s = (1 - s) * u0 + s * u_target
    target_v = u_target - u0
    pred_v = net(u_s, s, node_feats, edge_index, edge_feats, tau)
    return ((pred_v - target_v) ** 2).mean()


@torch.no_grad()
def sample(net, node_feats, edge_index, edge_feats, K, tau, steps=50, sigma=0.1):
    """Draw K samples by Euler integration of the learned flow.

    All K samples are processed in a single batched forward pass per Euler
    step, fully utilising GPU parallelism.

    Args:
        net:        FlowNet instance
        node_feats: [N, node_dim]
        edge_index: [2, E]
        edge_feats: [E, edge_dim]
        K:          Number of samples.
        tau:        Desired lag time (frames) — any value, scalar.
        steps:      Euler integration steps.
        sigma:      Prior scale (must match training).

    Returns:
        samples: [K, N, net.point_dim]
    """
    n = node_feats.shape[0]
    # All K draws in one [K, N, point_dim] tensor — batched forward each step
    u = torch.randn(K, n, net.point_dim, device=node_feats.device, dtype=node_feats.dtype) * sigma  # [K, N, point_dim]
    for i in range(steps):
        s = torch.tensor(i / steps, dtype=node_feats.dtype, device=node_feats.device)
        v = net(u, s, node_feats, edge_index, edge_feats, tau)  # [K, N, 6]
        u = u + v / steps
    return u


class NoiseSchedule(nn.Module):
    """Cosine DDPM noise schedule.

    Buffers are indexed 0..T: alphas_bar[t] = ᾱ_t where ᾱ_0 = 1 (clean),
    ᾱ_T ≈ 0 (fully noisy).  All buffers move with .to(device).
    """

    def __init__(self, T=200):
        super().__init__()
        self.T = T
        t = torch.arange(T + 1, dtype=torch.float32)
        s = 0.008  # offset prevents singularity at t=0
        f = torch.cos((t / T + s) / (1 + s) * (torch.pi / 2)) ** 2
        ab = f / f[0]                                          # [T+1]
        betas = torch.zeros(T + 1)
        betas[1:] = (1 - ab[1:] / ab[:-1]).clamp(0, 0.999)   # β_t
        post_var = torch.zeros(T + 1)
        post_var[1:] = (
            betas[1:] * (1 - ab[:-1]) / (1 - ab[1:]).clamp_min(1e-8)
        ).clamp_min(0)
        self.register_buffer("alphas_bar", ab)
        self.register_buffer("sqrt_alphas_bar", ab.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_bar",
                             (1 - ab).clamp_min(0).sqrt())
        self.register_buffer("betas", betas)
        self.register_buffer("posterior_variance", post_var)


def ddpm_loss(net, u_target, node_feats, edge_index, edge_feats, tau, schedule,
              pair_weights=None, sigma_aug=0.0):
    """DDPM ε-prediction loss.

    Supports [N, 6] (single pair) and [B, N, 6] (mini-batch) targets.
    Per-batch-item noise level t is sampled from [t_min, T] so the network
    learns the score at every noise level simultaneously.

    Args:
        net:          FlowNet instance.
        u_target:     [N, 6] or [B, N, 6] — clean target updates.
        node_feats:   [N, node_dim]
        edge_index:   [2, E]
        edge_feats:   [E, edge_dim]
        tau:          Scalar or [B] tensor — lag time(s) in frames.
        schedule:     NoiseSchedule instance (on same device as u_target).
        pair_weights: Optional [B] per-sample loss weights (density correction).
        sigma_aug:    Target augmentation noise scale (0 to disable).

    Returns:
        Scalar loss.
    """
    batched = u_target.dim() == 3
    if not batched:
        u_target = u_target.unsqueeze(0)    # [1, N, 6]
    B = u_target.shape[0]
    T = schedule.T
    t_min = max(1, T // 20)

    if sigma_aug > 0.0:
        u_target = u_target + sigma_aug * torch.randn_like(u_target)

    t_idx = torch.randint(t_min, T + 1, (B,), device=u_target.device)   # [B]
    eps = torch.randn_like(u_target)

    sqrt_ab   = schedule.sqrt_alphas_bar[t_idx].to(u_target.dtype)        # [B]
    sqrt_1mab = schedule.sqrt_one_minus_alphas_bar[t_idx].to(u_target.dtype)
    noisy_u = sqrt_ab[:, None, None] * u_target + sqrt_1mab[:, None, None] * eps

    s = (t_idx.float() / T).to(u_target.dtype)                            # [B]
    pred_eps = net(noisy_u, s, node_feats, edge_index, edge_feats, tau)

    per_sample = ((pred_eps - eps) ** 2).mean(dim=(-2, -1))               # [B]
    if pair_weights is not None:
        w = pair_weights.to(device=u_target.device, dtype=u_target.dtype)
        return (w * per_sample).mean()
    return per_sample.mean()


@torch.no_grad()
def sample_ddpm(net, node_feats, edge_index, edge_feats, K, tau, schedule,
                steps=50, eta=1.0, sigma_init=1.0):
    """DDPM/DDIM unified reverse-process sampler.

    Runs `steps` uniformly-strided denoising steps from t=T-1 down to t=0.
    eta=1.0 → full DDPM (stochastic, diverse, Boltzmann-stationary).
    eta=0.0 → DDIM (deterministic, faster, less diverse).
    sigma_init > 1.0 → broader prior for exploration beyond training data.

    Args:
        net:         FlowNet instance (in eval mode recommended).
        node_feats:  [N, node_dim]
        edge_index:  [2, E]
        edge_feats:  [E, edge_dim]
        K:           Number of samples to draw in parallel.
        tau:         Scalar lag time in frames.
        schedule:    NoiseSchedule (on same device as node_feats).
        steps:       Number of denoising steps.
        eta:         Stochasticity scale (1=DDPM, 0=DDIM).
        sigma_init:  Scale of the initial noise.

    Returns:
        samples: [K, N, net.point_dim]
    """
    T = schedule.T
    N = node_feats.shape[0]
    device = node_feats.device
    dtype = node_feats.dtype

    u = torch.randn(K, N, net.point_dim, device=device, dtype=dtype) * sigma_init

    # Strided timesteps T-1..0, steps+1 values
    t_full = torch.round(
        torch.linspace(T - 1, 0, steps + 1, device=device)
    ).long().clamp(0, T - 1)   # [steps+1]

    for i in range(steps):
        t = t_full[i].item()
        t_prev = t_full[i + 1].item()

        s = torch.tensor(t / T, dtype=dtype, device=device)
        eps_pred = net(u, s, node_feats, edge_index, edge_feats, tau)   # [K, N, net.point_dim]

        sqrt_ab_t   = schedule.sqrt_alphas_bar[t].to(dtype)
        sqrt_1mab_t = schedule.sqrt_one_minus_alphas_bar[t].to(dtype)
        ab_prev     = schedule.alphas_bar[t_prev].to(dtype)

        # Predicted clean update
        u0_hat = (u - sqrt_1mab_t * eps_pred) / sqrt_ab_t.clamp_min(1e-8)

        # DDPM/DDIM variance — use strided predecessor ab_prev, not single-step pv
        ab_t = schedule.alphas_bar[t].to(dtype)
        pv = (1 - ab_prev) / (1 - ab_t).clamp_min(1e-8) * \
             (1 - ab_t / ab_prev.clamp_min(1e-8))
        sigma_t   = eta * pv.clamp_min(0.0).sqrt()
        dir_coeff = (1 - ab_prev - sigma_t ** 2).clamp_min(0.0).sqrt()

        z = torch.randn_like(u) if t_prev > 0 else torch.zeros_like(u)
        u = ab_prev.sqrt() * u0_hat + dir_coeff * eps_pred + sigma_t * z

    return u
