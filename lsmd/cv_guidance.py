"""CV-space repulsion guidance for conformation exploration.

CVSpace fits a PCA basis on a training shard's Cα frames and exposes
differentiable project_single() and repulsion() methods so that
build_cv_guidance() can inject history-dependent steering into the
existing DDPM guidance hook (same interface as _build_wca_guidance).
"""
import torch
from lsmd import featurize as feat


class CVSpace:
    """PCA + Rg + RMSD collective-variable space for one protein.

    All stored tensors are float32 on CPU; .to(device) is called lazily
    inside project_single so the guidance_fn closure stays device-agnostic.
    """

    def __init__(self, n_pc: int = 3):
        self.n_pc = n_pc
        self.mean = None        # [3N] float32
        self.components = None  # [n_pc, 3N] float32
        self.rg_mean = None     # scalar float32
        self.rg_std = None      # scalar float32  (clamped > 0)
        self.rmsd_std = None    # scalar float32  (clamped > 0)

    def fit(self, coords: torch.Tensor) -> None:
        """Fit PCA basis from training shard Cα frames.

        Args:
            coords: [F, N, 3] Cα positions from the training shard.
        """
        F, N, _ = coords.shape
        X = coords.reshape(F, N * 3).float()
        mean = X.mean(dim=0)                          # [3N]
        _, _, Vh = torch.linalg.svd(X - mean, full_matrices=False)
        self.mean = mean.cpu()
        self.components = Vh[:self.n_pc].cpu()        # [n_pc, 3N]

        centroid = coords.mean(dim=1, keepdim=True)   # [F, 1, 3]
        rg = ((coords - centroid) ** 2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()  # [F]
        self.rg_mean = rg.mean().float().cpu()
        self.rg_std = rg.std().float().clamp_min(1e-8).cpu()

        mean_ca = mean.reshape(N, 3)
        rmsd = ((coords.float() - mean_ca.unsqueeze(0)) ** 2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()
        self.rmsd_std = rmsd.std().float().clamp_min(1e-8).cpu()

    def project_single(self, t: torch.Tensor) -> torch.Tensor:
        """Project one Cα frame onto the CV basis.

        Args:
            t: [N, 3] Cα positions, float32. May have requires_grad=True.

        Returns:
            cv: [n_pc + 2] float32 — [PC1..PCn_pc, Rg_norm, RMSD_norm].
                Differentiable w.r.t. t.
        """
        dev = t.device
        x_flat = t.reshape(-1).float()
        pc = self.components.to(dev) @ (x_flat - self.mean.to(dev))   # [n_pc]

        centroid = t.float().mean(dim=0)
        rg = ((t.float() - centroid) ** 2).sum(-1).mean().clamp_min(1e-8).sqrt()
        rg_norm = (rg - self.rg_mean.to(dev)) / self.rg_std.to(dev)

        mean_ca = self.mean.to(dev).reshape(-1, 3)
        rmsd = ((t.float() - mean_ca) ** 2).sum(-1).mean().clamp_min(1e-8).sqrt()
        rmsd_norm = rmsd / self.rmsd_std.to(dev)

        return torch.cat([pc, rg_norm.unsqueeze(0), rmsd_norm.unsqueeze(0)])

    def project_batch(self, coords: torch.Tensor) -> torch.Tensor:
        """Project a batch of Cα frames onto the CV basis.

        Replaces a Python for-loop of project_single calls with a single
        batched matmul — substantially faster for large training shards.

        Args:
            coords: [F, N, 3] Cα positions, float32.

        Returns:
            cv: [F, n_pc + 2] float32 — [PC1..PCn_pc, Rg_norm, RMSD_norm].
        """
        F, N, _ = coords.shape
        dev = self.mean.device
        X = coords.reshape(F, N * 3).float().to(dev)       # [F, 3N]
        pc = (X - self.mean) @ self.components.T            # [F, n_pc]

        coords_dev = X.reshape(F, N, 3)                    # [F, N, 3] — free view of X
        centroid = coords_dev.mean(dim=1, keepdim=True)     # [F, 1, 3]
        rg = ((coords_dev - centroid) ** 2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()
        rg_norm = (rg - self.rg_mean) / self.rg_std         # [F]

        mean_ca = self.mean.reshape(N, 3)                   # [N, 3]
        rmsd = ((coords_dev - mean_ca) ** 2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()
        rmsd_norm = rmsd / self.rmsd_std                    # [F]

        return torch.cat([pc, rg_norm.unsqueeze(1), rmsd_norm.unsqueeze(1)], dim=1)

    def to(self, device) -> "CVSpace":
        """Move all stored tensors to device in-place and return self."""
        self.mean = self.mean.to(device)
        self.components = self.components.to(device)
        self.rg_mean = self.rg_mean.to(device)
        self.rg_std = self.rg_std.to(device)
        self.rmsd_std = self.rmsd_std.to(device)
        return self

    def repulsion(self, cv: torch.Tensor, buffer,
                  sigma: float) -> torch.Tensor:
        """Gaussian repulsion potential from all structures in buffer.

        Args:
            cv: [n_cv] current CV vector, connected to computation graph.
            buffer: list of [n_cv] detached CV tensors, or pre-stacked [B, n_cv]
                    tensor (accepted structures). Pre-stacking avoids repeated
                    allocation inside the DDPM denoising loop.
            sigma: Gaussian width in normalized CV units.

        Returns:
            V: scalar — sum of repulsive Gaussians. Zero when buffer is empty.
        """
        if isinstance(buffer, torch.Tensor):
            if buffer.shape[0] == 0:
                return torch.zeros((), device=cv.device, dtype=cv.dtype)
            buf = buffer.to(device=cv.device, dtype=cv.dtype)
        elif not buffer:
            return torch.zeros((), device=cv.device, dtype=cv.dtype)
        else:
            buf = torch.stack([b.to(cv.device).to(cv.dtype) for b in buffer])
        diff = cv.unsqueeze(0) - buf                           # [B, n_cv]
        dists_sq = (diff ** 2).sum(-1)                        # [B]
        return torch.exp(-dists_sq / (2.0 * sigma ** 2)).sum()

    def save(self, path: str) -> None:
        torch.save({
            "n_pc": self.n_pc, "mean": self.mean,
            "components": self.components,
            "rg_mean": self.rg_mean, "rg_std": self.rg_std,
            "rmsd_std": self.rmsd_std,
        }, path)

    @classmethod
    def load(cls, path: str) -> "CVSpace":
        d = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(n_pc=d["n_pc"])
        obj.mean = d["mean"]
        obj.components = d["components"]
        obj.rg_mean = d["rg_mean"]
        obj.rg_std = d["rg_std"]
        obj.rmsd_std = d["rmsd_std"]
        return obj


def build_cv_guidance(R, t, chain_id, scale, cv_space, buffer, k_guide, sigma_cv):
    """Build a CV-space repulsion guidance callable for one rollout step.

    Mirrors _build_wca_guidance in transfer_eval.py: uses torch.enable_grad()
    internally, accepts detached tensors, returns guidance_fn(u0_hat).

    Args:
        R:         [N, 3, 3] current residue rotation matrices (detached).
        t:         [N, 3] current Cα positions (detached).
        chain_id:  [N] long (unused here but kept for API symmetry with WCA).
        scale:     [6] UpdateNorm de-normalization scale (detached).
        cv_space:  CVSpace instance (fitted).
        buffer:    list of [n_cv] detached CV tensors (accepted structures so far).
        k_guide:   Guidance step size (normalized-update units). 0.0 → identity.
        sigma_cv:  Gaussian width in normalized CV units.

    Returns:
        guidance_fn(u0_hat [N,6]) -> u0_hat_guided [N,6].
        Returns u0_hat unchanged when buffer is empty or k_guide == 0.
    """
    _buf_len = buffer.shape[0] if isinstance(buffer, torch.Tensor) else len(buffer)
    if k_guide == 0.0 or _buf_len == 0:
        return lambda u: u

    R_ref = R.detach()
    t_ref = t.detach()
    sc = scale.detach()
    # Pre-stack buffer once and move to the compute device to avoid repeated
    # O(B) allocation and CPU→GPU transfer inside the denoising loop.
    if isinstance(buffer, torch.Tensor):
        buf_stacked = buffer.to(t_ref.device)
    else:
        buf_stacked = torch.stack(buffer).to(t_ref.device)  # [B, n_cv]

    def guidance_fn(u0_hat):
        with torch.enable_grad():
            u0 = u0_hat.detach().requires_grad_(True)
            _, t_pred = feat.apply_update(R_ref, t_ref, u0 * sc)
            cv = cv_space.project_single(t_pred)
            V = cv_space.repulsion(cv, buf_stacked, sigma_cv)
            V.backward()
        grad = u0.grad.detach()
        grad_norm = grad.norm().clamp_min(1e-8)
        grad_n = grad / grad_norm
        return (u0_hat - k_guide * grad_norm.clamp_max(1.0) * grad_n).detach()

    return guidance_fn
