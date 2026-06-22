"""Kinetic + thermodynamic validation metrics for the transferable propagator.

Pure functions over Cα coordinate tensors [F, N, 3]. Rollout is the caller's
responsibility. Kinetic metrics live on a physical-time axis (picoseconds) so
model (step = tau_ps) and MD (step = dt) trajectories can be compared fairly.
"""
import math

import torch

from lsmd import geometry as g
from lsmd.validation import _hist_js


def interp_to_grid(time_ps, value, grid_ps):
    """Linear interpolation of (time_ps, value) onto grid_ps (1-D tensors).

    Points outside [time_ps[0], time_ps[-1]] hold the nearest endpoint value.
    """
    time_ps = torch.as_tensor(time_ps, dtype=torch.float64)
    value = torch.as_tensor(value, dtype=torch.float64)
    grid = torch.as_tensor(grid_ps, dtype=torch.float64)
    idx = torch.searchsorted(time_ps, grid).clamp(1, time_ps.shape[0] - 1)
    x0, x1 = time_ps[idx - 1], time_ps[idx]
    y0, y1 = value[idx - 1], value[idx]
    w = ((grid - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
    return y0 + w * (y1 - y0)


def curve_rmse(time_a, val_a, time_b, val_b, n=50):
    """RMSE between two curves over their overlapping time range (n grid points)."""
    time_a = torch.as_tensor(time_a, dtype=torch.float64)
    time_b = torch.as_tensor(time_b, dtype=torch.float64)
    lo = max(float(time_a[0]), float(time_b[0]))
    hi = min(float(time_a[-1]), float(time_b[-1]))
    if hi <= lo:
        return float("nan")
    grid = torch.linspace(lo, hi, n, dtype=torch.float64)
    a = interp_to_grid(time_a, val_a, grid)
    b = interp_to_grid(time_b, val_b, grid)
    return torch.sqrt(((a - b) ** 2).mean()).item()


def radius_of_gyration(ca):
    """Per-frame radius of gyration. ca: [F, N, 3] -> [F]."""
    centroid = ca.mean(dim=1, keepdim=True)            # [F, 1, 3]
    sq = ((ca - centroid) ** 2).sum(dim=-1)            # [F, N]
    return sq.mean(dim=1).sqrt()                       # [F]


def rg_distribution_js(ca_model, ca_md, bins=30):
    """JS divergence between the two radius-of-gyration distributions ([0, 1])."""
    return _hist_js(radius_of_gyration(ca_model), radius_of_gyration(ca_md), bins)


def shared_pca(ca_ref, n_components=2):
    """PCA basis from a reference ensemble.

    ca_ref: [F, N, 3]. Returns (mean [N*3], components [n_components, N*3]).
    Use the same basis to project both ensembles into a shared CV space.
    """
    F, N, _ = ca_ref.shape
    X = ca_ref.reshape(F, N * 3).double()
    mean = X.mean(dim=0)
    _, _, Vh = torch.linalg.svd(X - mean, full_matrices=False)
    return mean, Vh[:n_components]


def project_cv(ca, mean, components):
    """Project [F, N, 3] onto the PCA basis -> [F, n_components]."""
    X = ca.reshape(ca.shape[0], -1).double()
    return (X - mean) @ components.T
