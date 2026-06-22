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


class LearnedCGEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        # log-space → always positive; init reproduces cg_energy defaults
        self.log_alpha_mj = nn.Parameter(torch.zeros(()))               # α = 1
        self.log_k_angle  = nn.Parameter(torch.tensor(math.log(10.0)))  # k = 10
        self.log_wca_eps  = nn.Parameter(torch.tensor(math.log(0.3)))   # ε = 0.3
        self.log_w_mj     = nn.Parameter(torch.zeros(()))               # w = 1
        self.log_w_angle  = nn.Parameter(torch.zeros(()))
        self.log_w_wca    = nn.Parameter(torch.zeros(()))

    def forward(self, t, res_type, chain_id):
        alpha = self.log_alpha_mj.exp()
        k_ang = self.log_k_angle.exp()
        eps   = self.log_wca_eps.exp()
        w_mj  = self.log_w_mj.exp()
        w_ang = self.log_w_angle.exp()
        w_wca = self.log_w_wca.exp()
        E = t.new_zeros(())
        E = E + w_wca * cge._wca_energy(t, chain_id, sigma=4.5, eps=eps) / 2
        E = E + w_ang * cge.angle_energy(t, chain_id, k_angle=k_ang, theta0=2.094)
        E = E + w_mj * alpha * cge.mj_contact_energy(t, res_type, chain_id, cutoff=8.0)
        return E

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        m = cls()
        m.load_state_dict(torch.load(path, map_location=map_location))
        return m
