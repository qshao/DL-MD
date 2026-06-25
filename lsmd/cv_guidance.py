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
        rg = ((coords - centroid) ** 2).sum(-1).mean(-1).sqrt()  # [F]
        self.rg_mean = rg.mean().float().cpu()
        self.rg_std = rg.std().float().clamp_min(1e-8).cpu()

        mean_ca = mean.reshape(N, 3)
        rmsd = ((coords.float() - mean_ca.unsqueeze(0)) ** 2).sum(-1).mean(-1).sqrt()
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
        rg = ((t.float() - centroid) ** 2).sum(-1).mean().sqrt()
        rg_norm = (rg - self.rg_mean.to(dev)) / self.rg_std.to(dev)

        mean_ca = self.mean.to(dev).reshape(-1, 3)
        rmsd = ((t.float() - mean_ca) ** 2).sum(-1).mean().sqrt()
        rmsd_norm = rmsd / self.rmsd_std.to(dev)

        return torch.cat([pc, rg_norm.unsqueeze(0), rmsd_norm.unsqueeze(0)])

    def repulsion(self, cv: torch.Tensor, buffer: list,
                  sigma: float) -> torch.Tensor:
        """Gaussian repulsion potential from all structures in buffer.

        Args:
            cv: [n_cv] current CV vector, connected to computation graph.
            buffer: list of [n_cv] detached CV tensors (accepted structures).
            sigma: Gaussian width in normalized CV units.

        Returns:
            V: scalar — sum of repulsive Gaussians. Zero when buffer is empty.
        """
        if not buffer:
            return torch.zeros((), device=cv.device, dtype=cv.dtype)
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
