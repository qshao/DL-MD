# Physics Validation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable suite that quantifies the kinetic and thermodynamic physicality of generated trajectories against reference MD, and record a baseline for the current `v2_256h_90k` checkpoint.

**Architecture:** One new pure-function module (`lsmd/transfer_validate.py`) holding structural, thermodynamic, and kinetic metrics on Cα coordinate tensors `[F, N, 3]`; one CLI driver (`scripts/validate_physics.py`) that rolls the model out, calls the suite, and writes a JSON report; one test module with analytically-known synthetic cases.

**Tech Stack:** Python, PyTorch, pytest. Reuses `lsmd.validation._hist_js`, `lsmd.geometry.kabsch`, `lsmd.transfer_eval.rollout`.

## Global Constraints

- Dependencies limited to `numpy`, `torch`, `scipy` (already available). No `pyemma`/`deeptime`; no full MSM. TICA-style implied timescales are out of scope.
- All metric functions are pure: they take Cα coordinate tensors `[F, N, 3]` (or derived CV tensors) and return numbers/dicts. Rollout is the caller's responsibility.
- Metric functions operate on `float32`/`float64` CPU tensors. Internal math uses `float64` for numerical stability.
- Kinetic metrics (MSD, ACF) are returned as `(time_ps, value)` 1-D tensors and compared only after interpolation onto a shared physical-time grid. Model frame `i` is at `i * tau_ps`; MD frame `j` is at `j * dt_md_ps` where `dt_md_ps = shard["dt"]`.
- Reuse `lsmd.validation._hist_js` for 1-D JS divergence and `lsmd.geometry.kabsch` for alignment; do not reimplement them.
- The report JSON must include `"heldout": false` and the list of protein IDs — these baselines measure fit quality, not generalization.

---

### Task 1: Curve comparison helpers

**Files:**
- Create: `lsmd/transfer_validate.py`
- Test: `tests/test_transfer_validate.py`

**Interfaces:**
- Produces:
  - `interp_to_grid(time_ps, value, grid_ps) -> Tensor` — linear interp of a 1-D curve onto `grid_ps`, holding endpoints outside range.
  - `curve_rmse(time_a, val_a, time_b, val_b, n=50) -> float` — RMSE over the overlapping time range on an `n`-point grid.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_validate.py
import math
import torch
from lsmd import transfer_validate as tv


def test_curve_rmse_self_is_zero():
    t = torch.linspace(0.0, 100.0, 11)
    v = torch.sin(t / 10.0)
    assert tv.curve_rmse(t, v, t, v) < 1e-9


def test_curve_rmse_constant_offset():
    t = torch.linspace(0.0, 100.0, 11)
    v = torch.zeros(11)
    assert abs(tv.curve_rmse(t, v, t, v + 3.0) - 3.0) < 1e-6


def test_interp_to_grid_midpoint():
    t = torch.tensor([0.0, 10.0])
    v = torch.tensor([0.0, 10.0])
    out = tv.interp_to_grid(t, v, torch.tensor([5.0]))
    assert abs(out.item() - 5.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lsmd.transfer_validate'`

- [ ] **Step 3: Write minimal implementation**

```python
# lsmd/transfer_validate.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_validate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_validate.py tests/test_transfer_validate.py
git commit -m "feat: curve interpolation and RMSE helpers for validation suite"
```

---

### Task 2: Radius-of-gyration distribution JS

**Files:**
- Modify: `lsmd/transfer_validate.py`
- Test: `tests/test_transfer_validate.py`

**Interfaces:**
- Consumes: `_hist_js` (from `lsmd.validation`).
- Produces:
  - `radius_of_gyration(ca) -> Tensor[F]` — per-frame Rg.
  - `rg_distribution_js(ca_model, ca_md, bins=30) -> float` — JS divergence of the two Rg histograms in [0, 1].

- [ ] **Step 1: Write the failing test**

```python
def test_rg_of_static_structure_is_constant():
    ca = torch.randn(1, 8, 3).repeat(5, 1, 1)  # identical frames
    rg = tv.radius_of_gyration(ca)
    assert rg.shape == (5,)
    assert (rg - rg[0]).abs().max() < 1e-6


def test_rg_js_identical_ensembles_near_zero():
    torch.manual_seed(0)
    ca = torch.randn(40, 8, 3)
    assert tv.rg_distribution_js(ca, ca.clone()) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_validate.py -k rg -v`
Expected: FAIL — `AttributeError: module 'lsmd.transfer_validate' has no attribute 'radius_of_gyration'`

- [ ] **Step 3: Write minimal implementation**

```python
def radius_of_gyration(ca):
    """Per-frame radius of gyration. ca: [F, N, 3] -> [F]."""
    centroid = ca.mean(dim=1, keepdim=True)            # [F, 1, 3]
    sq = ((ca - centroid) ** 2).sum(dim=-1)            # [F, N]
    return sq.mean(dim=1).sqrt()                       # [F]


def rg_distribution_js(ca_model, ca_md, bins=30):
    """JS divergence between the two radius-of-gyration distributions ([0, 1])."""
    return _hist_js(radius_of_gyration(ca_model), radius_of_gyration(ca_md), bins)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_validate.py -k rg -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_validate.py tests/test_transfer_validate.py
git commit -m "feat: radius-of-gyration distribution JS metric"
```

---

### Task 3: Shared PCA collective-variable basis

**Files:**
- Modify: `lsmd/transfer_validate.py`
- Test: `tests/test_transfer_validate.py`

**Interfaces:**
- Produces:
  - `shared_pca(ca_ref, n_components=2) -> (mean[N*3], components[n_components, N*3])` — PCA basis fit on a reference ensemble via SVD.
  - `project_cv(ca, mean, components) -> Tensor[F, n_components]` — project frames onto the basis.

- [ ] **Step 1: Write the failing test**

```python
def test_pca_orders_variance_descending():
    torch.manual_seed(1)
    # Anisotropic cloud: large spread along residue-0 x, small elsewhere
    base = torch.randn(60, 4, 3) * 0.1
    base[:, 0, 0] += torch.randn(60) * 5.0
    mean, comps = tv.shared_pca(base, n_components=2)
    cv = tv.project_cv(base, mean, comps)
    assert cv.shape == (60, 2)
    assert cv[:, 0].var() >= cv[:, 1].var()


def test_pca_projection_is_zero_mean_on_fitting_set():
    torch.manual_seed(2)
    base = torch.randn(50, 4, 3)
    mean, comps = tv.shared_pca(base, n_components=2)
    cv = tv.project_cv(base, mean, comps)
    assert cv.mean(dim=0).abs().max() < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_validate.py -k pca -v`
Expected: FAIL — `AttributeError: ... has no attribute 'shared_pca'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_validate.py -k pca -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_validate.py tests/test_transfer_validate.py
git commit -m "feat: shared PCA collective-variable basis"
```

---

### Task 4: Free-energy surface comparison

**Files:**
- Modify: `lsmd/transfer_validate.py`
- Test: `tests/test_transfer_validate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (self-contained 2-D histogramming).
- Produces:
  - `free_energy_surface(cv, bins=30, kT=1.0, ranges=None) -> (F_grid[bins,bins], (xedges, yedges))` — `F = -kT ln P`, empty bins `nan`.
  - `fes_comparison(cv_model, cv_md, bins=30, kT=1.0, min_count=5) -> {"fes_js": float, "fes_rmse_kT": float}` — JS of the two densities and FES RMSE over bins well-sampled (`>= min_count`) in both.

- [ ] **Step 1: Write the failing test**

```python
def test_fes_identical_gaussians_low_rmse():
    torch.manual_seed(3)
    a = torch.randn(4000, 2)
    b = torch.randn(4000, 2)
    out = tv.fes_comparison(a, b, bins=20)
    assert out["fes_js"] < 0.1
    assert out["fes_rmse_kT"] < 0.6


def test_fes_disjoint_clouds_high_js():
    a = torch.randn(2000, 2) * 0.2 + torch.tensor([-5.0, 0.0])
    b = torch.randn(2000, 2) * 0.2 + torch.tensor([5.0, 0.0])
    out = tv.fes_comparison(a, b, bins=20)
    assert out["fes_js"] > 0.9


def test_free_energy_surface_empty_bins_are_nan():
    cv = torch.randn(500, 2)
    fg, _ = tv.free_energy_surface(cv, bins=30)
    assert torch.isnan(fg).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_validate.py -k fes -v`
Expected: FAIL — `AttributeError: ... has no attribute 'fes_comparison'`

- [ ] **Step 3: Write minimal implementation**

```python
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
    (xlo, xhi), (ylo, yhi) = ranges
    xedges = torch.linspace(xlo, xhi, bins + 1, dtype=torch.float64)
    yedges = torch.linspace(ylo, yhi, bins + 1, dtype=torch.float64)
    return xedges, yedges


def free_energy_surface(cv, bins=30, kT=1.0, ranges=None):
    """2-D free energy F = -kT ln P from CV[F, 2]. Empty bins are nan."""
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
    """JS of two CV densities + FES RMSE over jointly well-sampled bins."""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_validate.py -k fes -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_validate.py tests/test_transfer_validate.py
git commit -m "feat: free-energy surface JS and RMSE comparison"
```

---

### Task 5: Metastable-state populations

**Files:**
- Modify: `lsmd/transfer_validate.py`
- Test: `tests/test_transfer_validate.py`

**Interfaces:**
- Produces:
  - `state_populations(cv_model, cv_md, n_states=6, seed=0) -> {"pop_model": list, "pop_md": list, "pop_tv": float}` — k-means clusters fit on MD CV points; both ensembles assigned to nearest center; `pop_tv` is total-variation distance of the population vectors.

- [ ] **Step 1: Write the failing test**

```python
def test_populations_identical_ensembles_tv_zero():
    torch.manual_seed(4)
    cv = torch.randn(300, 2)
    out = tv.state_populations(cv, cv.clone(), n_states=4)
    assert out["pop_tv"] < 1e-6
    assert abs(sum(out["pop_model"]) - 1.0) < 1e-6


def test_populations_disjoint_clouds_tv_near_one():
    a = torch.randn(200, 2) * 0.1 + torch.tensor([-8.0, 0.0])
    b = torch.randn(200, 2) * 0.1 + torch.tensor([8.0, 0.0])
    # Fit clusters on a mix so both clouds get distinct centers
    mix = torch.cat([a, b], dim=0)
    out = tv.state_populations(a, b, n_states=2, seed=0)
    assert out["pop_tv"] > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_validate.py -k populations -v`
Expected: FAIL — `AttributeError: ... has no attribute 'state_populations'`

- [ ] **Step 3: Write minimal implementation**

```python
def _kmeans(points, n_states, seed=0, iters=50):
    """Lloyd's k-means with fixed seed. points: [M, D] -> centers [n_states, D]."""
    gen = torch.Generator().manual_seed(seed)
    init = torch.randperm(points.shape[0], generator=gen)[:n_states]
    centers = points[init].clone()
    for _ in range(iters):
        assign = torch.cdist(points, centers).argmin(dim=1)
        for k in range(n_states):
            m = assign == k
            if m.any():
                centers[k] = points[m].mean(dim=0)
    return centers


def state_populations(cv_model, cv_md, n_states=6, seed=0):
    """Cluster MD CV space; compare metastable-state populations.

    Returns model/MD population vectors and their total-variation distance
    (0 = identical populations, 1 = disjoint).
    """
    cv_model, cv_md = cv_model.double(), cv_md.double()
    # Fit centers on the pooled CV space so both ensembles' modes are covered.
    centers = _kmeans(torch.cat([cv_model, cv_md], dim=0), n_states, seed=seed)

    def pops(cv):
        a = torch.cdist(cv, centers).argmin(dim=1)
        c = torch.bincount(a, minlength=n_states).double()
        return c / c.sum().clamp_min(1.0)

    pm, pd = pops(cv_model), pops(cv_md)
    tv_dist = 0.5 * (pm - pd).abs().sum().item()
    return {"pop_model": pm.tolist(), "pop_md": pd.tolist(), "pop_tv": tv_dist}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_validate.py -k populations -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_validate.py tests/test_transfer_validate.py
git commit -m "feat: metastable-state population comparison via k-means"
```

---

### Task 6: Mean-squared-displacement curve

**Files:**
- Modify: `lsmd/transfer_validate.py`
- Test: `tests/test_transfer_validate.py`

**Interfaces:**
- Consumes: `g.kabsch` (alignment).
- Produces:
  - `msd_curve(ca, dt_ps, max_lag=None) -> (time_ps[L], msd[L])` — internal MSD vs lag after Kabsch-aligning every frame to frame 0; `time_ps[l] = l * dt_ps`.

- [ ] **Step 1: Write the failing test**

```python
def test_msd_static_structure_is_zero():
    ca = torch.randn(1, 6, 3).repeat(20, 1, 1)
    t, msd = tv.msd_curve(ca, dt_ps=10.0)
    assert msd.abs().max() < 1e-6
    assert torch.allclose(t, torch.arange(1, 11, dtype=torch.float64) * 10.0)


def test_msd_diffusion_increases_monotonically():
    torch.manual_seed(5)
    steps = torch.randn(40, 6, 3) * 0.5
    ca = steps.cumsum(dim=0)            # Brownian per residue
    _, msd = tv.msd_curve(ca, dt_ps=1.0)
    # First half should be non-decreasing on average
    assert msd[5] > msd[1]
    assert msd[-1] > msd[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_validate.py -k msd -v`
Expected: FAIL — `AttributeError: ... has no attribute 'msd_curve'`

- [ ] **Step 3: Write minimal implementation**

```python
def msd_curve(ca, dt_ps, max_lag=None):
    """Internal mean-squared displacement vs lag.

    ca: [F, N, 3]. Each frame is Kabsch-aligned to frame 0 to remove global
    translation/rotation, so MSD reflects internal motion (same convention as
    RMSF). Returns (time_ps [L], msd [L]) with time_ps[l] = l * dt_ps.
    """
    F = ca.shape[0]
    if max_lag is None:
        max_lag = F // 2
    max_lag = min(max_lag, F - 1)
    ref = ca[0].double()
    aligned = torch.empty(F, ca.shape[1], 3, dtype=torch.float64)
    for f in range(F):
        R, t = g.kabsch(ref, ca[f].double())
        aligned[f] = ca[f].double() @ R.T + t
    times, msds = [], []
    for lag in range(1, max_lag + 1):
        d = aligned[lag:] - aligned[:-lag]              # [F-lag, N, 3]
        times.append(lag * dt_ps)
        msds.append((d ** 2).sum(dim=-1).mean().item())
    return (torch.tensor(times, dtype=torch.float64),
            torch.tensor(msds, dtype=torch.float64))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_validate.py -k msd -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_validate.py tests/test_transfer_validate.py
git commit -m "feat: internal MSD curve on physical time axis"
```

---

### Task 7: CV autocorrelation and relaxation time

**Files:**
- Modify: `lsmd/transfer_validate.py`
- Test: `tests/test_transfer_validate.py`

**Interfaces:**
- Produces:
  - `cv_autocorrelation(cv_1d, dt_ps, max_lag=None) -> (time_ps[L], acf[L])` — normalized time autocorrelation; `acf[0] == 1`.
  - `relaxation_time_ps(time_ps, acf) -> float` — trapezoidal integral of `acf` up to its first zero crossing.

- [ ] **Step 1: Write the failing test**

```python
def test_acf_lag_zero_is_one():
    torch.manual_seed(6)
    q = torch.randn(200)
    t, acf = tv.cv_autocorrelation(q, dt_ps=2.0)
    assert abs(acf[0].item() - 1.0) < 1e-6
    assert t[0].item() == 0.0


def test_relaxation_time_recovers_ou_timescale():
    # Ornstein-Uhlenbeck: q[t+1] = (1 - 1/theta) q[t] + noise
    torch.manual_seed(7)
    theta = 20.0
    n = 8000
    q = torch.zeros(n)
    for i in range(1, n):
        q[i] = (1.0 - 1.0 / theta) * q[i - 1] + torch.randn(1).item()
    dt_ps = 1.0
    t, acf = tv.cv_autocorrelation(q, dt_ps=dt_ps)
    tau = tv.relaxation_time_ps(t, acf)
    # Continuous-time relaxation time of this AR(1) is ~theta * dt_ps
    assert 0.5 * theta * dt_ps < tau < 2.0 * theta * dt_ps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_validate.py -k "acf or relaxation" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'cv_autocorrelation'`

- [ ] **Step 3: Write minimal implementation**

```python
def cv_autocorrelation(cv_1d, dt_ps, max_lag=None):
    """Normalized time autocorrelation of a 1-D CV series.

    C(l) = mean_t[dq(t) dq(t+l)] / var(q), with dq = q - mean(q). acf[0] = 1.
    Returns (time_ps [L], acf [L]) with time_ps[l] = l * dt_ps.
    """
    q = cv_1d.double()
    F = q.shape[0]
    if max_lag is None:
        max_lag = F // 2
    max_lag = min(max_lag, F - 1)
    dq = q - q.mean()
    var = (dq * dq).mean().clamp_min(1e-12)
    times, acf = [], []
    for lag in range(0, max_lag + 1):
        c = (dq[lag:] * dq[:F - lag]).mean() / var
        times.append(lag * dt_ps)
        acf.append(c.item())
    return (torch.tensor(times, dtype=torch.float64),
            torch.tensor(acf, dtype=torch.float64))


def relaxation_time_ps(time_ps, acf):
    """Integral relaxation time: trapezoid integral of acf to first zero crossing."""
    time_ps = torch.as_tensor(time_ps, dtype=torch.float64)
    acf = torch.as_tensor(acf, dtype=torch.float64)
    cross = (acf <= 0).nonzero()
    end = int(cross[0].item()) if cross.numel() > 0 else acf.shape[0]
    if end < 2:
        return 0.0
    return torch.trapz(acf[:end], time_ps[:end]).item()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_validate.py -k "acf or relaxation" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_validate.py tests/test_transfer_validate.py
git commit -m "feat: CV autocorrelation and integral relaxation time"
```

---

### Task 8: Top-level `validate` driver

**Files:**
- Modify: `lsmd/transfer_validate.py`
- Test: `tests/test_transfer_validate.py`

**Interfaces:**
- Consumes: all metric functions above; `lsmd.validation.rmsf_profile`, `distance_matrix_js`, `ca_geometry`.
- Produces:
  - `validate(ca_model, ca_md, *, tau_ps, dt_md_ps, kT=1.0, n_states=6) -> dict` with keys `structural`, `thermodynamic`, `kinetic` (schema in the spec).

- [ ] **Step 1: Write the failing test**

```python
def test_validate_returns_full_schema():
    torch.manual_seed(8)
    ca_model = torch.randn(60, 10, 3)
    ca_md = torch.randn(120, 10, 3)
    rep = tv.validate(ca_model, ca_md, tau_ps=2000.0, dt_md_ps=200.0)
    for section in ("structural", "thermodynamic", "kinetic"):
        assert section in rep
    assert set(rep["structural"]) >= {"rmsf_corr", "dist_js", "rg_js",
                                      "ca_bond_mean", "clash_count"}
    assert set(rep["thermodynamic"]) >= {"fes_js", "fes_rmse_kT", "pop_tv"}
    assert set(rep["kinetic"]) >= {"msd_rmse", "acf_rmse", "relax_model_ps",
                                   "relax_md_ps", "relax_ratio"}


def test_validate_identical_ensembles_have_strong_agreement():
    torch.manual_seed(9)
    ca = torch.randn(80, 10, 3)
    rep = tv.validate(ca, ca.clone(), tau_ps=200.0, dt_md_ps=200.0)
    assert rep["structural"]["dist_js"] < 1e-3
    assert rep["thermodynamic"]["pop_tv"] < 1e-3
    assert abs(rep["kinetic"]["relax_ratio"] - 1.0) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_validate.py -k validate -v`
Expected: FAIL — `AttributeError: ... has no attribute 'validate'`

- [ ] **Step 3: Write minimal implementation**

```python
def validate(ca_model, ca_md, *, tau_ps, dt_md_ps, kT=1.0, n_states=6):
    """Full kinetic + thermodynamic + structural validation report.

    ca_model: [F_m, N, 3] generated CA frames (model time step = tau_ps).
    ca_md:    [F_d, N, 3] reference MD CA frames (time step = dt_md_ps).
    """
    from lsmd import validation as val

    # --- structural ---
    rmsf = val.rmsf_profile(ca_model, ca_md)
    bonds, clashes = [], []
    for fr in ca_model:
        geo = val.ca_geometry(fr)
        bonds.append(geo["ca_bond_mean"])
        clashes.append(geo["clash_count"])
    structural = {
        "rmsf_corr": rmsf["corr"],
        "dist_js": val.distance_matrix_js(ca_model, ca_md),
        "rg_js": rg_distribution_js(ca_model, ca_md),
        "ca_bond_mean": float(sum(bonds) / len(bonds)),
        "clash_count": float(sum(clashes) / len(clashes)),
    }

    # --- shared CV basis (fit on MD) ---
    mean, comps = shared_pca(ca_md, n_components=2)
    cv_model = project_cv(ca_model, mean, comps)
    cv_md = project_cv(ca_md, mean, comps)

    # --- thermodynamic ---
    thermo = fes_comparison(cv_model, cv_md, kT=kT)
    thermo["pop_tv"] = state_populations(cv_model, cv_md, n_states=n_states)["pop_tv"]

    # --- kinetic (shared physical time axis) ---
    tm, mm = msd_curve(ca_model, tau_ps)
    td, md = msd_curve(ca_md, dt_md_ps)
    tam, am = cv_autocorrelation(cv_model[:, 0], tau_ps)
    tad, ad = cv_autocorrelation(cv_md[:, 0], dt_md_ps)
    rm = relaxation_time_ps(tam, am)
    rd = relaxation_time_ps(tad, ad)
    kinetic = {
        "msd_rmse": curve_rmse(tm, mm, td, md),
        "acf_rmse": curve_rmse(tam, am, tad, ad),
        "relax_model_ps": rm,
        "relax_md_ps": rd,
        "relax_ratio": (rm / rd) if rd > 0 else float("nan"),
    }
    return {"structural": structural, "thermodynamic": thermo, "kinetic": kinetic}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_validate.py -k validate -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_validate.py tests/test_transfer_validate.py
git commit -m "feat: top-level validate() driver aggregating all metrics"
```

---

### Task 9: CLI driver `scripts/validate_physics.py`

**Files:**
- Create: `scripts/validate_physics.py`
- Test: `tests/test_validate_physics_cli.py`

**Interfaces:**
- Consumes: `lsmd.transfer_eval.load_checkpoint`/`rollout`, `lsmd.transfer_validate.validate`, `lsmd.geometry.so3_exp`.
- Produces: a CLI writing a JSON report with `heldout`, `checkpoint`, `settings`, `proteins`, `summary` keys. Importable helper `build_report(checkpoint_dict, shard_paths, settings, device) -> dict` for testing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_physics_cli.py
import torch
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "validate_physics",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "validate_physics.py"))
validate_physics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_physics)


def test_summarize_means():
    proteins = {
        "a": {"structural": {"rmsf_corr": 0.4, "dist_js": 0.01},
              "thermodynamic": {"fes_js": 0.1}, "kinetic": {"relax_ratio": 1.0}},
        "b": {"structural": {"rmsf_corr": 0.6, "dist_js": 0.03},
              "thermodynamic": {"fes_js": 0.3}, "kinetic": {"relax_ratio": 1.2}},
    }
    s = validate_physics.summarize(proteins)
    assert abs(s["mean_rmsf_corr"] - 0.5) < 1e-9
    assert abs(s["mean_dist_js"] - 0.02) < 1e-9
    assert abs(s["mean_fes_js"] - 0.2) < 1e-9
    assert abs(s["mean_relax_ratio"] - 1.1) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validate_physics_cli.py -v`
Expected: FAIL — `FileNotFoundError` / module load error (script does not exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/validate_physics.py
"""Baseline physics validation of a transferable checkpoint against MD shards.

Rolls the model out per shard, computes the kinetic + thermodynamic + structural
metric suite (lsmd.transfer_validate.validate), and writes a JSON report.

NOTE: these baselines measure fit quality, not generalization — all ATLAS shards
were seen in training and there is no held-out split yet. The report records
"heldout": false accordingly.

Usage
-----
python scripts/validate_physics.py \\
    --checkpoint checkpoints/v2_256h_90k.pt \\
    --shard data/atlas/3u7t_A.pt --shard data/atlas/1z0b_A.pt \\
    --steps 200 --tau_ps 2000 --diff_steps 20 --eta 1.0 \\
    --out validation_baseline.json
"""
import argparse
import json
import os

import torch

from lsmd import geometry as g
from lsmd import transfer_eval as te
from lsmd import transfer_validate as tv


def _protein_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def build_report(ckpt, shard_paths, settings, device):
    """Run rollout + validate for each shard. Returns the proteins dict."""
    net, sched, norm = te.load_checkpoint(ckpt, device=device)
    k_eff = ckpt["hparams"].get("k", settings["k"])
    proteins = {}
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if "R_aa" in shard:
            R0 = g.so3_exp(shard["R_aa"][0].float())
        else:
            R0 = shard["R"][0]
        t0 = shard["t"][0].float()
        traj = te.rollout(
            net, sched, norm, R0, t0,
            shard["res_type"], shard["chain_id"], shard["res_index"],
            steps=settings["steps"], tau_ps=settings["tau_ps"], k=k_eff,
            diff_steps=settings["diff_steps"], eta=settings["eta"],
            temp_K=settings["temp_K"],
            bond_constraint_iters=settings["bond_constraint_iters"],
            max_update_norm=settings["max_update_norm"],
            wca_sigma=settings["wca_sigma"], wca_eps=settings["wca_eps"],
            wca_lam=settings["wca_lam"], device=device).cpu()
        rep = tv.validate(traj, shard["t"].float(),
                          tau_ps=settings["tau_ps"], dt_md_ps=float(shard["dt"]),
                          kT=settings["kT"], n_states=settings["n_states"])
        rep["n_res"] = int(shard["n_res"])
        proteins[_protein_id(path)] = rep
    return proteins


def summarize(proteins):
    """Mean headline metrics across proteins."""
    def mean(getter):
        vals = [getter(p) for p in proteins.values()]
        return float(sum(vals) / len(vals)) if vals else float("nan")
    return {
        "mean_rmsf_corr": mean(lambda p: p["structural"]["rmsf_corr"]),
        "mean_dist_js": mean(lambda p: p["structural"]["dist_js"]),
        "mean_fes_js": mean(lambda p: p["thermodynamic"]["fes_js"]),
        "mean_relax_ratio": mean(lambda p: p["kinetic"]["relax_ratio"]),
    }


def main():
    ap = argparse.ArgumentParser(description="Physics validation baseline")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", action="append", required=True, dest="shards",
                    help="MD shard .pt (repeatable)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tau_ps", type=float, default=2000.0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--diff_steps", type=int, default=20)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--temp_K", type=float, default=300.0)
    ap.add_argument("--wca_sigma", type=float, default=4.5)
    ap.add_argument("--wca_eps", type=float, default=0.3)
    ap.add_argument("--wca_lam", type=float, default=0.05)
    ap.add_argument("--bond_constraint_iters", type=int, default=5)
    ap.add_argument("--max_update_norm", type=float, default=3.0)
    ap.add_argument("--n_states", type=int, default=6)
    ap.add_argument("--kT", type=float, default=1.0)
    ap.add_argument("--out", default="validation_baseline.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    settings = {
        "steps": args.steps, "tau_ps": args.tau_ps, "k": args.k,
        "diff_steps": args.diff_steps, "eta": args.eta, "temp_K": args.temp_K,
        "wca_sigma": args.wca_sigma, "wca_eps": args.wca_eps,
        "wca_lam": args.wca_lam, "bond_constraint_iters": args.bond_constraint_iters,
        "max_update_norm": args.max_update_norm, "n_states": args.n_states,
        "kT": args.kT,
    }
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    proteins = build_report(ckpt, args.shards, settings, device)
    report = {
        "heldout": False,
        "checkpoint": args.checkpoint,
        "settings": settings,
        "proteins": proteins,
        "summary": summarize(proteins),
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validate_physics_cli.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_physics.py tests/test_validate_physics_cli.py
git commit -m "feat: validate_physics CLI driver and report summary"
```

---

### Task 10: Run the baseline and commit the artifact

**Files:**
- Create: `validation_baseline.json`
- Modify: `docs/superpowers/specs/2026-06-22-physics-validation-suite-design.md` (mark Phase 1 done)

**Interfaces:**
- Consumes: `scripts/validate_physics.py`.
- Produces: committed baseline numbers for `v2_256h_90k`.

- [ ] **Step 1: Run the full suite once to confirm it executes end-to-end**

Run:
```bash
python scripts/validate_physics.py \
    --checkpoint checkpoints/v2_256h_90k.pt \
    --shard data/atlas/3u7t_A.pt \
    --steps 200 --tau_ps 2000 --diff_steps 20 --eta 1.0 \
    --out /tmp/validation_smoke.json
```
Expected: prints a `summary` dict with four `mean_*` keys, no traceback.

- [ ] **Step 2: Run the full baseline over all six proteins**

Run:
```bash
python scripts/validate_physics.py \
    --checkpoint checkpoints/v2_256h_90k.pt \
    --shard data/atlas/3u7t_A.pt --shard data/atlas/4p3a_B.pt \
    --shard data/atlas/1b2s_F.pt --shard data/atlas/2y4x_B.pt \
    --shard data/atlas/1z0b_A.pt --shard data/atlas/6ovk_R.pt \
    --steps 300 --tau_ps 2000 --diff_steps 20 --eta 1.0 \
    --out validation_baseline.json
```
Expected: writes `validation_baseline.json` with six proteins and a summary.

- [ ] **Step 3: Sanity-check the artifact**

Run: `python -c "import json; r=json.load(open('validation_baseline.json')); assert r['heldout'] is False; assert len(r['proteins'])==6; print(r['summary'])"`
Expected: prints the summary dict, no assertion error.

- [ ] **Step 4: Run the whole test suite**

Run: `pytest tests/ -q`
Expected: all tests pass (existing suite + the new validation tests).

- [ ] **Step 5: Commit**

```bash
git add validation_baseline.json docs/superpowers/specs/2026-06-22-physics-validation-suite-design.md
git commit -m "chore: Phase 1 physics validation baseline for v2_256h_90k"
```

---

## Self-Review

**Spec coverage:**
- Structural metrics (RMSF, dist-JS, geometry, Rg) → Tasks 2, 8. ✓
- Thermodynamic (shared PCA CV, FES JS+RMSE, state populations) → Tasks 3, 4, 5, 8. ✓
- Kinetic (MSD, ACF, relaxation time) on shared physical time axis → Tasks 1, 6, 7, 8. ✓
- Report schema with `heldout` flag + summary → Task 9. ✓
- Baseline artifact for `v2_256h_90k` over the six named proteins → Task 10. ✓
- Testing strategy (static/diffusion MSD, OU ACF, Gaussian FES, identical/disjoint populations, interp/RMSE, PCA variance ordering) → Tasks 1–8 tests. ✓
- Dependency limit, time-axis alignment, reuse of `_hist_js`/`kabsch` → Global Constraints + Tasks 1, 6. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step is complete.

**Type consistency:** `validate(...)` returns the three-section dict consumed by `build_report`/`summarize` in Task 9; `msd_curve`/`cv_autocorrelation` both return `(time_ps, value)` tensors consumed by `curve_rmse` and `relaxation_time_ps`; `shared_pca` returns `(mean, components)` consumed by `project_cv`; `fes_comparison`/`state_populations` return the dict keys read in Task 8. Consistent across tasks.
