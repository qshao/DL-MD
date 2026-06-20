"""Flow-matching graph network for long-stride protein MD.

All inputs and outputs live in the invariant delta-space, so a plain
graph net suffices — equivariance is guaranteed by construction.
"""
import torch
import torch.nn as nn


def tau_embedding(tau, dim=16, device=None, dtype=torch.float32):
    """Log-sinusoidal embedding of a lag-time scalar.

    Uses log(tau) so the embedding varies smoothly across orders of magnitude.
    At inference, any tau value (seen or unseen during training) can be passed.

    Args:
        tau:    Lag time in frames — Python int, float, or scalar tensor.
        dim:    Embedding dimension (must be even).
        device: Target device.
        dtype:  Target dtype.

    Returns:
        emb: [dim] float32 tensor
    """
    tau = torch.as_tensor(tau, dtype=dtype, device=device)
    log_tau = torch.log(tau.clamp_min(1.0))                   # log-scale
    half = dim // 2
    freqs = torch.arange(half, dtype=dtype, device=device) / half
    freqs = 10.0 ** freqs                                      # log-spaced frequencies
    args = log_tau * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [dim]


class MessageLayer(nn.Module):
    """Single message-passing layer with mean aggregation and residual update."""

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
        src, dst = edge_index
        msg = self.msg(torch.cat([h[src], h[dst], edge_feats], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(h.shape[0], 1, device=h.device).index_add_(
            0, dst, torch.ones(dst.shape[0], 1, device=h.device))
        agg = agg / deg.clamp_min(1.0)
        return h + self.upd(torch.cat([h, agg], dim=-1))


class FlowNet(nn.Module):
    """Conditional flow-matching graph network.

    Operates entirely in the invariant delta-space.  Takes node features,
    edge features, the current interpolated update u_s, and the flow-time
    scalar s, and predicts the rectified-flow velocity field.
    """

    def __init__(self, node_dim, edge_dim, hidden=64, layers=3, tau_emb_dim=16):
        super().__init__()
        self.tau_emb_dim = tau_emb_dim
        # input: node features + u (6) + flow-time s (1) + tau embedding (tau_emb_dim)
        self.embed = nn.Linear(node_dim + 6 + 1 + tau_emb_dim, hidden)
        self.layers = nn.ModuleList(
            [MessageLayer(hidden, edge_dim) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, u_s, s, node_feats, edge_index, edge_feats, tau):
        """Predict velocity in delta-space conditioned on lag time tau.

        Args:
            u_s:        Interpolated update  [N, 6]  float32
            s:          Flow-time scalar      []      float32
            node_feats: Node features         [N, node_dim]
            edge_index: Edge indices          [2, E]
            edge_feats: Edge features         [E, edge_dim]
            tau:        Lag time (frames)     scalar int/float/tensor

        Returns:
            velocity:   Predicted velocity    [N, 6]  float32
        """
        n = node_feats.shape[0]
        s_col = (
            torch.as_tensor(s, dtype=u_s.dtype, device=u_s.device)
            .reshape(1, 1)
            .expand(n, 1)
        )
        tau_emb = (
            tau_embedding(tau, dim=self.tau_emb_dim,
                          device=u_s.device, dtype=u_s.dtype)
            .reshape(1, -1)
            .expand(n, -1)
        )
        h = self.embed(torch.cat([node_feats, u_s, s_col, tau_emb], dim=-1))
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)
        return self.head(h)


def cfm_loss(net, u_target, node_feats, edge_index, edge_feats, tau, sigma=0.1):
    """Conditional flow-matching (rectified-flow) loss.

    Samples a random flow-time s ~ Uniform[0,1] and a random prior sample
    u0 ~ N(0, sigma^2), linearly interpolates to u_s, then regresses the
    network's velocity prediction onto the straight-line target velocity.

    Args:
        net:        FlowNet instance
        u_target:   Ground-truth delta-space update  [N, 6]
        node_feats: Node features                    [N, node_dim]
        edge_index: Edge indices                     [2, E]
        edge_feats: Edge features                    [E, edge_dim]
        tau:        Lag time used for this pair (frames) — conditions the net
        sigma:      Prior scale (small motions)

    Returns:
        loss: Scalar MSE loss
    """
    u0 = torch.randn_like(u_target) * sigma
    s = torch.rand(())
    u_s = (1 - s) * u0 + s * u_target
    target_v = u_target - u0
    pred_v = net(u_s, s, node_feats, edge_index, edge_feats, tau)
    return ((pred_v - target_v) ** 2).mean()


@torch.no_grad()
def sample(net, node_feats, edge_index, edge_feats, K, tau, steps=50, sigma=0.1):
    """Draw K samples by Euler integration of the learned flow.

    Integrates from u ~ N(0, sigma^2) at s=0 toward s=1 using K
    independent noise draws.

    Args:
        net:        FlowNet instance
        node_feats: Node features  [N, node_dim]
        edge_index: Edge indices   [2, E]
        edge_feats: Edge features  [E, edge_dim]
        K:          Number of samples
        tau:        Desired lag time (frames) — can differ from training taus
        steps:      Number of Euler integration steps
        sigma:      Prior scale (must match training)

    Returns:
        samples: [K, N, 6] float32
    """
    n = node_feats.shape[0]
    outs = []
    for _ in range(K):
        u = torch.randn(n, 6, device=node_feats.device, dtype=node_feats.dtype) * sigma
        for i in range(steps):
            s = torch.tensor(i / steps, dtype=node_feats.dtype, device=node_feats.device)
            v = net(u, s, node_feats, edge_index, edge_feats, tau)
            u = u + v / steps
        outs.append(u)
    return torch.stack(outs, dim=0)
