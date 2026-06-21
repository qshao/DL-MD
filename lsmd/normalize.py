"""Corpus-level normalization of per-residue SE(3) updates.

Update units (Angstrom for translation, radians for rotation) are already
consistent across proteins, so a single global per-component scale suffices to
put the DDPM target near unit variance.
"""
import torch


class UpdateNorm:
    def __init__(self, scale):
        self.scale = scale.clamp_min(1e-6)      # [point_dim], always >= 1e-6

    @classmethod
    def fit(cls, updates, clip_quantile=0.99):
        """Fit per-component scale from a sample of updates [M, point_dim].

        Uses the ``clip_quantile`` absolute-value percentile as the scale
        (default 99th percentile) rather than std so that extreme outliers
        from high-temperature MD or degenerate backbone frames do not inflate
        the scale and corrupt normalization.  Any NaN/Inf rows are dropped
        before fitting.
        """
        flat = updates.reshape(-1, updates.shape[-1])
        # Drop rows containing NaN or Inf
        valid = flat.isfinite().all(dim=-1)
        flat = flat[valid]
        if flat.shape[0] < 2:
            raise ValueError(
                f"UpdateNorm.fit needs at least 2 finite samples, got {flat.shape[0]}"
            )
        scale = flat.abs().quantile(clip_quantile, dim=0)
        return cls(scale)

    def normalize(self, u):
        return u / self.scale.to(u)

    def denormalize(self, u):
        return u * self.scale.to(u)

    def state_dict(self):
        return {"scale": self.scale.clone()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["scale"].clone())
