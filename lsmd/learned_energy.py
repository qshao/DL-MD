"""Phase 3 learnable conservative energy and Stage-1 fitting utilities.

LearnedCGEnergy wraps the cg_energy.py local terms (WCA + angle + MJ contacts)
with a small set of log-space learnable coefficients, initialized to reproduce
cg_energy.total_cg_energy defaults. Energetic parameters are LEARNED from MD
data (no hand-specified values); the M&J matrix shape is initialization only.
"""
import math

import torch
import torch.nn as nn

from lsmd import cg_energy as cge


def _soft_mj_energy(t, res_type, mj_matrix, *, cutoff=8.0, steepness=2.0):
    """MJ contact energy with soft sigmoid gate — fully differentiable w.r.t. t.

    The original cge.mj_contact_energy uses a hard dist<cutoff boolean mask which
    produces zero gradient w.r.t. positions almost everywhere, making score-matching
    unable to fit the MJ parameters. This version replaces the hard mask with
    sigmoid((cutoff - dist) * steepness), giving smooth, nonzero gradients near
    the contact boundary.
    """
    N = t.shape[0]
    diff = t.unsqueeze(0) - t.unsqueeze(1)                   # [N, N, 3]
    dist = (diff * diff).sum(-1).clamp_min(1e-8).sqrt()      # [N, N]
    idx = torch.arange(N, device=t.device)
    seq_sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    upper_tri = idx.unsqueeze(1) < idx.unsqueeze(0)
    not_unk = (res_type < 20).unsqueeze(1) & (res_type < 20).unsqueeze(0)
    mask = (upper_tri & (seq_sep > 3) & not_unk).float()
    gate = torch.sigmoid(steepness * (cutoff - dist))         # ~1 inside, ~0 outside
    ri = res_type.clamp(max=19)
    e_ij = mj_matrix[ri.unsqueeze(1), ri.unsqueeze(0)]        # [N, N]
    return (e_ij * gate * mask).sum()


class LearnedCGEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        # log-space scalars for WCA + angle terms
        self.log_k_angle  = nn.Parameter(torch.tensor(math.log(10.0)))  # k = 10
        self.log_wca_eps  = nn.Parameter(torch.tensor(math.log(0.3)))   # ε = 0.3
        self.log_w_angle  = nn.Parameter(torch.zeros(()))
        self.log_w_wca    = nn.Parameter(torch.zeros(()))
        # Learnable 20×20 symmetric MJ matrix with soft sigmoid contact gate.
        # Hard-cutoff MJ gives zero positional gradient (and thus zero parameter
        # gradient) almost everywhere — score-matching cannot fit it. The soft gate
        # provides continuous gradients near the contact boundary.
        # Initialized from MJ defaults; symmetrized in forward().
        self.mj_matrix_raw = nn.Parameter(cge.MJ_MATRIX.clone())

    def forward(self, t, res_type, chain_id):
        k_ang  = self.log_k_angle.exp()
        eps    = self.log_wca_eps.exp()
        w_ang  = self.log_w_angle.exp()
        w_wca  = self.log_w_wca.exp()
        mj_mat = (self.mj_matrix_raw + self.mj_matrix_raw.T) / 2  # enforce symmetry
        E = t.new_zeros(())
        E = E + w_wca * cge._wca_energy(t, chain_id, sigma=4.5, eps=eps) / 2
        E = E + w_ang * cge.angle_energy(t, chain_id, k_angle=k_ang, theta0=2.094)
        E = E + _soft_mj_energy(t, res_type, mj_mat.to(t.device))
        return E

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        m = cls()
        m.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
        m.to(map_location)
        return m


def score_matching_loss(energy, t, res_type, chain_id, *, sigma=0.5, kT=0.593):
    """Denoising score-matching loss (Vincent 2011) for one CA frame.

    Perturbs t with Gaussian noise of scale sigma and trains the model score
    -∇U_θ(x_noisy)/kT to match the denoising target (x_clean - x_noisy)/sigma².
    The energy's locality makes this a local, corpus-poolable fit.
    """
    noise = sigma * torch.randn_like(t)
    t_noisy = (t + noise).requires_grad_(True)
    U = energy(t_noisy, res_type, chain_id)
    grad = torch.autograd.grad(U, t_noisy, create_graph=True)[0]
    score_model = -grad / kT
    score_target = (t.detach() - t_noisy.detach()) / (sigma ** 2)
    return ((score_model - score_target) ** 2).mean()


def inverse_density_weights(cv, *, bins=30, clip=10.0):
    """Per-frame inverse-density weights over a 2-D CV space.

    Frames in over-represented bins get smaller weights so dominant basins do
    not dominate the energy fit. Weights are normalized to mean 1 then clipped.

    Args:
        cv:   [F, 2] collective-variable coordinates (e.g. shared-PCA top 2).
    Returns:
        [F] weights in [1/clip, clip], mean ≈ 1 (pre-clip).
    """
    cv = cv.double()
    lo = cv.min(dim=0).values
    hi = cv.max(dim=0).values
    span = (hi - lo).clamp_min(1e-8)
    # bin index per frame in each dimension
    ij = ((cv - lo) / span * (bins - 1)).round().long().clamp(0, bins - 1)
    flat = ij[:, 0] * bins + ij[:, 1]                  # [F]
    counts = torch.bincount(flat, minlength=bins * bins).double()
    w = 1.0 / counts[flat]                              # inverse density
    w = w / w.mean()
    return w.clamp(1.0 / clip, clip).to(torch.float32)


def langevin_sample(energy, t0, res_type, chain_id, *,
                    n_steps=2000, dt=1e-3, kT=0.593, stride=10):
    """Overdamped Langevin sampling from p ∝ exp(-U/kT) (γ = 1, reduced units).

    Update: x ← x - dt·∇U(x) + sqrt(2·kT·dt)·N(0, I).
    Returns the collected samples [S, N, 3] (one every `stride` steps).
    """
    if n_steps <= 0:
        raise ValueError(f"langevin_sample requires n_steps >= 1, got {n_steps}")
    t = t0.clone()
    samples = []
    noise_scale = (2.0 * kT * dt) ** 0.5
    for step in range(n_steps):
        t = t.detach().requires_grad_(True)
        U = energy(t, res_type, chain_id)
        grad = torch.autograd.grad(U, t)[0]
        t = (t - dt * grad + noise_scale * torch.randn_like(t)).detach()
        if step % stride == 0:
            samples.append(t.clone())
    return torch.stack(samples, dim=0)


def frame_energy_cut(energy, t, res_type, chain_id, *, pct=95.0):
    """High-percentile per-residue energy ceiling over MD frames.

    Returns the `pct`-percentile of U_θ(frame)/N across frames, so the Stage-2
    hinge ceiling is comparable across protein sizes.
    """
    N = t.shape[1]
    with torch.no_grad():
        per = torch.tensor([float(energy(t[i], res_type, chain_id)) / max(N, 1)
                            for i in range(t.shape[0])])
    return float(torch.quantile(per, pct / 100.0))


def md_step_cov(t, dt_md_ps, tau_ps):
    """Mean per-atom, per-coordinate squared CA displacement at lag τ.

    Args:
        t:         [F, N, 3] MD CA frames.
        dt_md_ps:  MD frame spacing (ps).
        tau_ps:    physical lag (ps); converted to a frame lag by rounding.
    Returns:
        scalar float: mean squared displacement per atom/coord, E[Δx²].
    """
    lag = max(1, int(round(float(tau_ps) / float(dt_md_ps))))
    disp = t[lag:] - t[:-lag]                  # [F-lag, N, 3]
    return float((disp ** 2).mean())
