"""Corpus-level normalization of per-residue SE(3) updates.

Update units (Angstrom for translation, radians for rotation) are already
consistent across proteins, so a single global per-component scale suffices to
put the DDPM target near unit variance.
"""
import torch


class UpdateNorm:
    def __init__(self, scale):
        self.scale = scale                       # [point_dim]

    @classmethod
    def fit(cls, updates):
        """Fit per-component scale from a sample of updates [M, point_dim]."""
        scale = updates.reshape(-1, updates.shape[-1]).std(dim=0).clamp_min(1e-6)
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
