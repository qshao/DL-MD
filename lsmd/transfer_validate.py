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


def _hist2d(cv, xedges, yedges):
    """2-D bin counts of cv[F, 2] given bin edges. Returns [bins, bins]."""
    bins = xedges.shape[0] - 1
    xi = torch.bucketize(cv[:, 0].double(), xedges[1:-1])
    yi = torch.bucketize(cv[:, 1].double(), yedges[1:-1])
    counts = torch.zeros(bins * bins, dtype=torch.float64)
    counts.scatter_add_(0, xi * bins + yi,
                        torch.ones(cv.shape[0], dtype=torch.float64))
    return counts.view(bins, bins)


def _edges_from_ranges(ranges, bins):
    """Create bin edges from (xlo, xhi), (ylo, yhi) ranges. Returns (xedges, yedges)."""
    (xlo, xhi), (ylo, yhi) = ranges
    xedges = torch.linspace(xlo, xhi, bins + 1, dtype=torch.float64)
    yedges = torch.linspace(ylo, yhi, bins + 1, dtype=torch.float64)
    return xedges, yedges


def free_energy_surface(cv, bins=30, kT=1.0, ranges=None):
    """2-D free energy F = -kT ln P from CV[F, 2]. Empty bins are nan.

    Args:
        cv: [F, 2] coordinate tensor
        bins: number of bins per dimension
        kT: temperature parameter
        ranges: ((xlo, xhi), (ylo, yhi)) or None to auto-detect

    Returns:
        (F_grid [bins, bins], (xedges, yedges))
    """
    cv = cv.double()
    if ranges is None:
        ranges = ((cv[:, 0].min().item(), cv[:, 0].max().item()),
                  (cv[:, 1].min().item(), cv[:, 1].max().item()))
    xedges, yedges = _edges_from_ranges(ranges, bins)
    counts = _hist2d(cv, xedges, yedges)
    P = counts / counts.sum()
    F_grid = torch.full((bins, bins), float("nan"), dtype=torch.float64)
    nz = counts > 0
    F_grid[nz] = -kT * torch.log(P[nz])
    return F_grid, (xedges, yedges)


def fes_comparison(cv_model, cv_md, bins=30, kT=1.0, min_count=5):
    """JS divergence of two CV densities + FES RMSE over jointly well-sampled bins.

    Args:
        cv_model: [F, 2] coordinate tensor (model ensemble)
        cv_md: [F, 2] coordinate tensor (MD ensemble)
        bins: number of bins per dimension
        kT: temperature parameter
        min_count: minimum bin count for RMSE inclusion

    Returns:
        {"fes_js": float, "fes_rmse_kT": float}
    """
    cv_model, cv_md = cv_model.double(), cv_md.double()
    allcv = torch.cat([cv_model, cv_md], dim=0)
    ranges = ((allcv[:, 0].min().item(), allcv[:, 0].max().item()),
              (allcv[:, 1].min().item(), allcv[:, 1].max().item()))
    xedges, yedges = _edges_from_ranges(ranges, bins)
    cm = _hist2d(cv_model, xedges, yedges)
    cd = _hist2d(cv_md, xedges, yedges)
    pm = (cm + 1e-12) / (cm + 1e-12).sum()
    pd = (cd + 1e-12) / (cd + 1e-12).sum()
    mix = 0.5 * (pm + pd)
    js = 0.5 * (pm * torch.log(pm / mix)).sum() + 0.5 * (pd * torch.log(pd / mix)).sum()
    js = (js / math.log(2)).clamp(0.0, 1.0).item()

    well = (cm >= min_count) & (cd >= min_count)
    if well.any():
        fm = -kT * torch.log(pm[well]); fm = fm - fm.min()
        fd = -kT * torch.log(pd[well]); fd = fd - fd.min()
        rmse = torch.sqrt(((fm - fd) ** 2).mean()).item()
    else:
        rmse = float("nan")
    return {"fes_js": js, "fes_rmse_kT": rmse}
