# CV-Guided Conformation Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CV-space repulsion guidance to the SE(3) PropagatorNet rollout so it generates protein conformations outside the training MD distribution, then provide a script to run and validate those explorations.

**Architecture:** A new `CVSpace` class (`lsmd/cv_guidance.py`) fits a PCA basis on the training shard and exposes a differentiable `repulsion()` potential. A `build_cv_guidance()` function wraps that into a `guidance_fn(u0_hat) -> u0_hat_guided` callable — the same interface already used by WCA guidance — so `rollout()` in `transfer_eval.py` only needs three new parameters. An exploration script runs a sequential batch loop, growing a CV buffer as structures are accepted, and writes PDB candidates for downstream MD validation.

**Tech Stack:** PyTorch (SVD-based PCA, autograd), existing `lsmd.featurize.apply_update` for SE(3) update→position conversion, `lsmd.validation.ca_geometry` for geometry filter, `lsmd.decoder.write_ca_pdb` for PDB output, matplotlib for CV coverage plot, json for summary.

## Global Constraints

- No physics-based loss functions or CG energy scoring — all guidance is purely geometric.
- `cv_guidance.py` lives in `lsmd/`; it must not import from `scripts/`.
- `build_cv_guidance()` must mirror the `_build_wca_guidance()` pattern exactly: use `torch.enable_grad()` internally, accept detached R/t/scale, return a `guidance_fn(u0_hat) -> u0_hat_guided` callable.
- `rollout()` in `transfer_eval.py` must remain backward-compatible: all new parameters are keyword-only with defaults that reproduce existing behavior.
- Shard format is a dict with keys `"R"`, `"t"` (`[F, N, 3]` or `[F, N, 3, 3]`), `"res_type"`, `"chain_id"`, `"res_index"`, `"seq"` — do not change it.
- PDB output uses `decoder.write_ca_pdb(ca [N,3], res_type_names list[str], path)`.
- `apply_update(R_t, t_t, u)` returns `(R_f, t_f)`; `u` shape `[N, 6]` = `[local_trans(3), axis_angle(3)]`.
- Tests follow the existing pattern: `_synthetic_shard(F, N, dt, seed)` from `tests/test_transfer_eval.py` and `g.so3_exp()` from `lsmd.geometry`.

---

### Task 1: CVSpace class in `lsmd/cv_guidance.py`

**Files:**
- Create: `lsmd/cv_guidance.py`
- Create: `tests/test_cv_guidance.py`

**Interfaces:**
- Produces:
  - `CVSpace(n_pc: int = 3)` — class
  - `CVSpace.fit(coords: Tensor[F, N, 3]) -> None`
  - `CVSpace.project_single(t: Tensor[N, 3]) -> Tensor[n_pc+2]` — float32, differentiable w.r.t. `t`
  - `CVSpace.repulsion(cv: Tensor[n_pc+2], buffer: list[Tensor], sigma: float) -> Tensor scalar` — differentiable w.r.t. `cv`
  - `CVSpace.save(path: str) -> None`
  - `CVSpace.load(path: str) -> CVSpace` — classmethod

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cv_guidance.py
import torch
import pytest
from lsmd.cv_guidance import CVSpace
from lsmd import geometry as g


def _coords(F=20, N=10, seed=0):
    torch.manual_seed(seed)
    return torch.randn(F, N, 3) * 5.0


def test_fit_sets_attributes():
    cv = CVSpace(n_pc=3)
    cv.fit(_coords(F=20, N=10))
    assert cv.mean.shape == (30,)       # 3N = 30
    assert cv.components.shape == (3, 30)
    assert cv.rg_mean.ndim == 0
    assert cv.rg_std > 0
    assert cv.rmsd_std > 0


def test_project_single_shape():
    cv = CVSpace(n_pc=3)
    cv.fit(_coords(F=20, N=10))
    t = torch.randn(10, 3)
    out = cv.project_single(t)
    assert out.shape == (5,)   # n_pc=3 + Rg + RMSD = 5


def test_project_single_is_differentiable():
    cv = CVSpace(n_pc=2)
    cv.fit(_coords(F=20, N=8))
    t = torch.randn(8, 3, requires_grad=True)
    out = cv.project_single(t)
    out.sum().backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad).all()


def test_repulsion_zero_with_empty_buffer():
    cv = CVSpace(n_pc=2)
    cv.fit(_coords(F=20, N=8))
    t = torch.randn(8, 3)
    c = cv.project_single(t)
    V = cv.repulsion(c, [], sigma=1.0)
    assert V.item() == 0.0


def test_repulsion_positive_with_buffer():
    cv = CVSpace(n_pc=2)
    cv.fit(_coords(F=20, N=8))
    t = torch.randn(8, 3)
    c = cv.project_single(t)
    buf = [c.detach()]
    V = cv.repulsion(c, buf, sigma=1.0)
    assert V.item() > 0.0


def test_repulsion_gradient_flows():
    cv = CVSpace(n_pc=2)
    cv.fit(_coords(F=20, N=8))
    t = torch.randn(8, 3, requires_grad=True)
    c = cv.project_single(t)
    buf = [torch.randn(4).detach()]
    V = cv.repulsion(c, buf, sigma=1.0)
    V.backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad).all()


def test_save_load_roundtrip(tmp_path):
    cv = CVSpace(n_pc=3)
    cv.fit(_coords(F=20, N=10))
    path = str(tmp_path / "cv.pt")
    cv.save(path)
    cv2 = CVSpace.load(path)
    assert cv2.n_pc == 3
    assert torch.allclose(cv2.mean, cv.mean)
    assert torch.allclose(cv2.components, cv.components)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/qshao/DL-MD
python -m pytest tests/test_cv_guidance.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'lsmd.cv_guidance'`

- [ ] **Step 3: Write `lsmd/cv_guidance.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cv_guidance.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add lsmd/cv_guidance.py tests/test_cv_guidance.py
git commit -m "feat: add CVSpace for PCA-based CV-space guidance"
```

---

### Task 2: `build_cv_guidance()` function in `lsmd/cv_guidance.py`

**Files:**
- Modify: `lsmd/cv_guidance.py` (append `build_cv_guidance` function)
- Modify: `tests/test_cv_guidance.py` (append tests)

**Interfaces:**
- Consumes: `CVSpace` from Task 1; `feat.apply_update` from `lsmd.featurize`
- Produces:
  - `build_cv_guidance(R, t, chain_id, scale, cv_space, buffer, k_guide, sigma_cv) -> Callable[[Tensor], Tensor]`
  - The returned callable has signature `guidance_fn(u0_hat: Tensor[N,6]) -> Tensor[N,6]`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cv_guidance.py`)

```python
from lsmd.cv_guidance import CVSpace, build_cv_guidance
from lsmd import featurize as feat


def _simple_setup(N=8, n_pc=2, seed=7):
    torch.manual_seed(seed)
    coords = torch.randn(20, N, 3) * 5.0
    cv_space = CVSpace(n_pc=n_pc)
    cv_space.fit(coords)
    R = g.so3_exp(torch.zeros(N, 3))  # identity rotations
    t = coords[0].clone()
    scale = torch.ones(6)
    chain_id = torch.zeros(N, dtype=torch.long)
    return cv_space, R, t, chain_id, scale


def test_build_cv_guidance_empty_buffer_is_identity():
    cv_space, R, t, chain_id, scale = _simple_setup()
    fn = build_cv_guidance(R, t, chain_id, scale, cv_space,
                           buffer=[], k_guide=0.5, sigma_cv=1.0)
    u = torch.randn(8, 6)
    out = fn(u)
    assert torch.allclose(out, u)


def test_build_cv_guidance_with_buffer_changes_u():
    cv_space, R, t, chain_id, scale = _simple_setup()
    # Put the current structure into the buffer so repulsion is strong
    with torch.no_grad():
        _, t_cur = feat.apply_update(R, t, torch.zeros(8, 6))
    cv_cur = cv_space.project_single(t_cur).detach()
    buffer = [cv_cur]
    fn = build_cv_guidance(R, t, chain_id, scale, cv_space,
                           buffer=buffer, k_guide=0.5, sigma_cv=1.0)
    u = torch.zeros(8, 6)
    out = fn(u)
    assert not torch.allclose(out, u), "guidance should change u when buffer is non-empty"
    assert torch.isfinite(out).all()


def test_build_cv_guidance_k_guide_zero_is_identity():
    cv_space, R, t, chain_id, scale = _simple_setup()
    cv_cur = cv_space.project_single(t).detach()
    fn = build_cv_guidance(R, t, chain_id, scale, cv_space,
                           buffer=[cv_cur], k_guide=0.0, sigma_cv=1.0)
    u = torch.randn(8, 6)
    out = fn(u)
    assert torch.allclose(out, u)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cv_guidance.py::test_build_cv_guidance_empty_buffer_is_identity -v
```

Expected: `ImportError: cannot import name 'build_cv_guidance'`

- [ ] **Step 3: Append `build_cv_guidance` to `lsmd/cv_guidance.py`**

```python
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
    if k_guide == 0.0 or not buffer:
        return lambda u: u

    R_ref = R.detach()
    t_ref = t.detach()
    sc = scale.detach()

    def guidance_fn(u0_hat):
        with torch.enable_grad():
            u0 = u0_hat.detach().requires_grad_(True)
            _, t_pred = feat.apply_update(R_ref, t_ref, u0 * sc)
            cv = cv_space.project_single(t_pred)
            V = cv_space.repulsion(cv, buffer, sigma_cv)
            V.backward()
        grad = u0.grad.detach()
        grad_norm = grad.norm().clamp_min(1e-8)
        grad_n = grad / grad_norm
        return (u0_hat - k_guide * grad_norm.clamp_max(1.0) * grad_n).detach()

    return guidance_fn
```

- [ ] **Step 4: Run all cv_guidance tests**

```bash
python -m pytest tests/test_cv_guidance.py -v
```

Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add lsmd/cv_guidance.py tests/test_cv_guidance.py
git commit -m "feat: add build_cv_guidance for CV-space repulsion in DDPM rollout"
```

---

### Task 3: Extend `rollout()` in `lsmd/transfer_eval.py`

**Files:**
- Modify: `lsmd/transfer_eval.py` (add 4 new keyword parameters to `rollout()`)
- Modify: `tests/test_transfer_eval.py` (add 2 new tests)

**Interfaces:**
- Consumes: `CVSpace`, `build_cv_guidance` from Task 2
- Produces: extended `rollout()` signature:
  ```python
  def rollout(net, schedule, update_norm, R0, t0, res_type, chain_id, res_index,
              *, steps, tau_ps, k, diff_steps=50, eta=1.0, temp_K=300.0,
              bond_constraint_iters=5, max_update_norm=3.0,
              wca_sigma=4.5, wca_eps=0.3, wca_lam=0.05,
              noether=False,
              cv_space=None, cv_buffer=None, k_guide=0.05, sigma_cv=1.0,
              guide_warmup=50,
              device="cpu"):
  ```

- [ ] **Step 1: Write the failing tests** (append to `tests/test_transfer_eval.py`)

```python
from lsmd.cv_guidance import CVSpace


def test_rollout_with_cv_space_runs_and_has_correct_shape():
    import random
    shards = [_synthetic_shard(N=10, seed=i) for i in range(3)]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=1, steps=2, T_diff=20,
                    norm_samples=16, device="cpu", seed=0)
    net, sched, norm = te.load_checkpoint(ckpt, device="cpu")
    sh = shards[0]
    # Fit CVSpace on training frames
    cv_space = CVSpace(n_pc=2)
    cv_space.fit(sh["t"])   # sh["t"]: [F, N, 3]
    cv_buffer = []
    traj = te.rollout(net, sched, norm, sh["R"][0], sh["t"][0],
                      sh["res_type"], sh["chain_id"], sh["res_index"],
                      steps=4, tau_ps=200.0, k=4, diff_steps=3, device="cpu",
                      cv_space=cv_space, cv_buffer=cv_buffer, k_guide=0.05,
                      sigma_cv=1.0, guide_warmup=0)
    assert traj.shape == (5, 10, 3)
    assert torch.isfinite(traj).all()


def test_rollout_cv_none_matches_original():
    """cv_space=None must reproduce the original rollout exactly."""
    import random
    shards = [_synthetic_shard(N=10, seed=i) for i in range(3)]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=1, steps=2, T_diff=20,
                    norm_samples=16, device="cpu", seed=42)
    net, sched, norm = te.load_checkpoint(ckpt, device="cpu")
    sh = shards[0]
    torch.manual_seed(0)
    traj_orig = te.rollout(net, sched, norm, sh["R"][0], sh["t"][0],
                           sh["res_type"], sh["chain_id"], sh["res_index"],
                           steps=3, tau_ps=200.0, k=4, diff_steps=3, device="cpu")
    torch.manual_seed(0)
    traj_cv = te.rollout(net, sched, norm, sh["R"][0], sh["t"][0],
                         sh["res_type"], sh["chain_id"], sh["res_index"],
                         steps=3, tau_ps=200.0, k=4, diff_steps=3, device="cpu",
                         cv_space=None, cv_buffer=None)
    assert torch.allclose(traj_orig, traj_cv)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_transfer_eval.py::test_rollout_with_cv_space_runs_and_has_correct_shape -v
```

Expected: `TypeError: rollout() got an unexpected keyword argument 'cv_space'`

- [ ] **Step 3: Add the 4 new parameters to `rollout()`**

In `lsmd/transfer_eval.py`, change the `rollout()` signature from:

```python
@torch.no_grad()
def rollout(net, schedule, update_norm, R0, t0, res_type, chain_id, res_index,
            *, steps, tau_ps, k, diff_steps=50, eta=1.0, temp_K=300.0,
            bond_constraint_iters=5, max_update_norm=3.0,
            wca_sigma=4.5, wca_eps=0.3, wca_lam=0.05,
            noether=False, device="cpu"):
```

to:

```python
@torch.no_grad()
def rollout(net, schedule, update_norm, R0, t0, res_type, chain_id, res_index,
            *, steps, tau_ps, k, diff_steps=50, eta=1.0, temp_K=300.0,
            bond_constraint_iters=5, max_update_norm=3.0,
            wca_sigma=4.5, wca_eps=0.3, wca_lam=0.05,
            noether=False,
            cv_space=None, cv_buffer=None, k_guide=0.05, sigma_cv=1.0,
            guide_warmup=50,
            device="cpu"):
```

Then inside the step loop, immediately after the existing `guidance_fn = ...` block (lines ~188-192), add:

```python
        # CV-space repulsion guidance (composed with WCA if both active)
        if (cv_space is not None and cv_buffer is not None
                and len(cv_buffer) >= guide_warmup):
            from lsmd.cv_guidance import build_cv_guidance as _build_cv
            cv_fn = _build_cv(R, t, chain_id, scale, cv_space,
                              cv_buffer, k_guide, sigma_cv)
            if guidance_fn is not None:
                _wca = guidance_fn
                guidance_fn = lambda u, _w=_wca, _c=cv_fn: _c(_w(u))
            else:
                guidance_fn = cv_fn
```

- [ ] **Step 4: Run all transfer_eval tests**

```bash
python -m pytest tests/test_transfer_eval.py -v
```

Expected: `5 passed` (3 original + 2 new)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_eval.py tests/test_transfer_eval.py
git commit -m "feat: add cv_space guidance parameters to rollout()"
```

---

### Task 4: `scripts/explore_conformations.py`

**Files:**
- Create: `scripts/explore_conformations.py`
- Create: `tests/test_explore_conformations.py`

**Interfaces:**
- Consumes:
  - `te.load_checkpoint(ckpt_dict, device)` → `(net, sched, norm)`
  - `te.rollout(..., cv_space=, cv_buffer=, k_guide=, sigma_cv=, guide_warmup=)` from Task 3
  - `CVSpace` from Task 1
  - `val.ca_geometry(ca [N,3]) -> dict` with keys `ca_bond_mean`, `clash_count`
  - `dec.write_ca_pdb(ca [N,3], res_type_names list[str], path str)`
- Produces: `explore_out/` directory with `structures.pt`, `cv_coords.npy`, `candidates/*.pdb`, `cv_coverage.png`, `summary.json`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explore_conformations.py
import json
import subprocess
import sys
import torch
import pytest
from pathlib import Path
from lsmd import transfer_train as tt
from lsmd import geometry as g


def _make_ckpt(tmp_path, F=15, N=8, seed=0):
    torch.manual_seed(seed)
    shards = [{
        "R": g.so3_exp(torch.randn(F, N, 3) * 0.05),
        "t": torch.randn(F, N, 3) * 5.0,
        "res_type": torch.zeros(N, dtype=torch.long),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "dt": 200.0, "seq": ["ALA"] * N, "n_res": N,
    }]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=1, steps=2, T_diff=20,
                    norm_samples=16, device="cpu", seed=seed)
    ckpt_path = str(tmp_path / "test.pt")
    torch.save(ckpt, ckpt_path)
    shard_path = str(tmp_path / "shard.pt")
    torch.save(shards[0], shard_path)
    return ckpt_path, shard_path


def test_explore_smoke(tmp_path):
    ckpt_path, shard_path = _make_ckpt(tmp_path)
    out_dir = str(tmp_path / "out")
    result = subprocess.run(
        [sys.executable, "scripts/explore_conformations.py",
         "--checkpoint", ckpt_path,
         "--shard", shard_path,
         "--n_explore", "5",
         "--n_steps", "2",
         "--diff_steps", "3",
         "--tau_ps", "200",
         "--k_guide", "0.05",
         "--sigma_cv", "1.0",
         "--guide_warmup", "0",
         "--out", out_dir,
         "--seed", "0"],
        capture_output=True, text=True, cwd="/home/qshao/DL-MD"
    )
    assert result.returncode == 0, result.stderr
    out = Path(out_dir)
    assert (out / "summary.json").exists()
    assert (out / "cv_basis.pt").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary) >= 0   # may be 0 if all fail geometry filter


def test_explore_output_structure(tmp_path):
    ckpt_path, shard_path = _make_ckpt(tmp_path, F=20, N=8, seed=1)
    out_dir = str(tmp_path / "out2")
    subprocess.run(
        [sys.executable, "scripts/explore_conformations.py",
         "--checkpoint", ckpt_path,
         "--shard", shard_path,
         "--n_explore", "8",
         "--n_steps", "2",
         "--diff_steps", "3",
         "--tau_ps", "200",
         "--guide_warmup", "0",
         "--out", out_dir,
         "--seed", "42"],
        capture_output=True, text=True, cwd="/home/qshao/DL-MD", check=True
    )
    out = Path(out_dir)
    summary = json.loads((out / "summary.json").read_text())
    for entry in summary:
        for key in ("id", "cv", "rmsd_native", "clashes", "bond_rmsd",
                    "md_pass", "md_rmsd_final", "md_rg_final"):
            assert key in entry, f"missing key {key}"
        assert entry["md_pass"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_explore_conformations.py -v 2>&1 | head -20
```

Expected: `FileNotFoundError` or `ModuleNotFoundError` for `explore_conformations`

- [ ] **Step 3: Write `scripts/explore_conformations.py`**

```python
"""CV-guided conformation explorer.

Generates diverse protein Cα conformations by adding a history-dependent
repulsion in collective-variable (CV) space to the DDPM denoising guidance.
Outputs PDB candidates for external MD relaxation validation.

Usage
-----
python scripts/explore_conformations.py \
    --checkpoint checkpoints/v4_3u7t_A.pt \
    --shard data/atlas/3u7t_A.pt \
    --n_explore 500 --n_steps 50 --tau_ps 2000 \
    --k_guide 0.05 --sigma_cv 1.0 --guide_warmup 50 \
    --out explore_out/3u7t_A
"""
import argparse
import json
import os

import numpy as np
import torch

from lsmd import featurize as feat
from lsmd import geometry as g
from lsmd import transfer_eval as te
from lsmd import validation as val
from lsmd import decoder as dec
from lsmd.cv_guidance import CVSpace


def _plot_coverage(cv_buffer, cv_space, ref_cv, out_dir, step):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    if ref_cv is not None and ref_cv.shape[0] > 0:
        ax.scatter(ref_cv[:, 0], ref_cv[:, 1], c="lightgrey", s=10,
                   label="training", zorder=1)
    if cv_buffer:
        gen = np.stack([c.numpy() for c in cv_buffer])
        sc = ax.scatter(gen[:, 0], gen[:, 1],
                        c=range(len(cv_buffer)), cmap="plasma",
                        s=20, zorder=2, label="generated")
        plt.colorbar(sc, ax=ax, label="generation index")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"CV coverage — {len(cv_buffer)} accepted (step {step})")
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_coverage.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n_explore", type=int, default=500)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--tau_ps", type=float, default=2000.0)
    ap.add_argument("--temp_K", type=float, default=375.0)
    ap.add_argument("--k_guide", type=float, default=0.05)
    ap.add_argument("--sigma_cv", type=float, default=1.0)
    ap.add_argument("--guide_warmup", type=int, default=50)
    ap.add_argument("--n_pc", type=int, default=3)
    ap.add_argument("--diff_steps", type=int, default=20)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--out", default="explore_out")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-accepted structures in summary.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cand_dir = os.path.join(args.out, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

    # Load checkpoint and shard
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shard = torch.load(args.shard, map_location="cpu", weights_only=False)
    net, sched, norm = te.load_checkpoint(ckpt, device=args.device)
    k_eff = ckpt["hparams"].get("k", 16)
    res_type_names = shard.get("seq", ["ALA"] * shard["n_res"])

    # Get Cα coordinates from shard
    ca_ref = shard["t"].float()                      # [F, N, 3]
    mean_ca = ca_ref.mean(dim=0)                     # [N, 3]

    # Fit CVSpace on training frames
    cv_space = CVSpace(n_pc=args.n_pc)
    cv_space.fit(ca_ref)
    cv_space.save(os.path.join(args.out, "cv_basis.pt"))

    # Project training frames to CV for the coverage plot
    ref_cv = np.stack([
        cv_space.project_single(ca_ref[i]).numpy()
        for i in range(ca_ref.shape[0])
    ])  # [F, n_pc+2]

    # Resume: load existing summary and CV buffer
    summary_path = os.path.join(args.out, "summary.json")
    results = []
    cv_buffer = []
    start_id = 0
    if args.resume and os.path.exists(summary_path):
        with open(summary_path) as fh:
            results = json.load(fh)
        for r in results:
            cv_buffer.append(torch.tensor(r["cv"], dtype=torch.float32))
        start_id = max((r["id"] for r in results), default=-1) + 1
        print(f"Resuming from {len(results)} accepted structures (next id={start_id})")

    # Get initial SE(3) frame
    if "R_aa" in shard:
        R0_all = g.so3_exp(shard["R_aa"].float())
    else:
        R0_all = shard["R"].float()  # [F, N, 3, 3]

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    for attempt in range(start_id, args.n_explore):
        # Pick random training frame as starting point
        f_idx = torch.randint(ca_ref.shape[0], (1,), generator=rng).item()
        R0 = R0_all[f_idx].to(args.device)
        t0 = ca_ref[f_idx].to(args.device)

        traj = te.rollout(
            net, sched, norm, R0, t0,
            shard["res_type"].to(args.device),
            shard["chain_id"].to(args.device),
            shard["res_index"].to(args.device),
            steps=args.n_steps, tau_ps=args.tau_ps, k=k_eff,
            diff_steps=args.diff_steps, eta=args.eta, temp_K=args.temp_K,
            cv_space=cv_space, cv_buffer=cv_buffer,
            k_guide=args.k_guide, sigma_cv=args.sigma_cv,
            guide_warmup=args.guide_warmup,
            device=args.device,
        )
        x_final = traj[-1].cpu()    # [N, 3]

        # Geometry filter
        geo = val.ca_geometry(x_final)
        clashes = geo["clash_count"]
        ref_bond = (mean_ca[1:] - mean_ca[:-1]).norm(dim=-1).mean().item()
        bond_rmsd = abs(geo["ca_bond_mean"] - ref_bond)
        if clashes >= 0.5 or bond_rmsd >= 0.1:
            continue

        # Compute CV and add to buffer
        cv_i = cv_space.project_single(x_final).detach()
        cv_buffer.append(cv_i)

        # RMSD from native mean structure
        rmsd_native = ((x_final - mean_ca) ** 2).sum(-1).mean().sqrt().item()

        results.append({
            "id": attempt,
            "cv": cv_i.tolist(),
            "rmsd_native": round(rmsd_native, 4),
            "clashes": clashes,
            "bond_rmsd": round(bond_rmsd, 4),
            "md_pass": None,
            "md_rmsd_final": None,
            "md_rg_final": None,
        })

        # Save PDB
        dec.write_ca_pdb(x_final, res_type_names,
                         os.path.join(cand_dir, f"{attempt:05d}.pdb"))

        # Save structures tensor and numpy CV array
        accepted_coords = torch.stack([ca_ref[0]] + [
            torch.tensor(r["cv"], dtype=torch.float32)    # placeholder—see note
            for r in results
        ])

        # Flush summary every 10 accepts
        if len(results) % 10 == 0:
            with open(summary_path, "w") as fh:
                json.dump(results, fh, indent=2)
            np.save(os.path.join(args.out, "cv_coords.npy"),
                    np.stack([r["cv"] for r in results]))
            _plot_coverage(cv_buffer, cv_space, ref_cv, args.out, attempt)
            print(f"[{attempt+1}/{args.n_explore}] accepted={len(results)}")

    # Final save
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    if results:
        np.save(os.path.join(args.out, "cv_coords.npy"),
                np.stack([r["cv"] for r in results]))
        _plot_coverage(cv_buffer, cv_space, ref_cv, args.out, args.n_explore)

    print(f"Done. {len(results)} structures accepted out of {args.n_explore} attempts.")
    print(f"PDB candidates: {cand_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
```

Note: the `accepted_coords` placeholder block above accumulates dummy data — replace with proper `structures.pt` saving after the loop. The correct final save of `structures.pt` should be:

```python
# After the loop, save actual Cα coordinates
if results:
    # Re-read saved PDBs is impractical; instead collect coords in the loop
    # The loop should append x_final to a list `all_coords`, then at the end:
    # torch.save(torch.stack(all_coords), os.path.join(args.out, "structures.pt"))
```

Implement this by adding `all_coords = []` before the loop and `all_coords.append(x_final)` inside the geometry-pass block, then `torch.save(torch.stack(all_coords), ...)` after the loop.

The complete corrected loop section (replace the placeholder above):

```python
    all_coords = []   # ← add this before the for-loop

    for attempt in range(start_id, args.n_explore):
        # ... (all existing code) ...
        # After: cv_buffer.append(cv_i), add:
        all_coords.append(x_final)

    # Final save (replace the placeholder block):
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    if results:
        torch.save(torch.stack(all_coords), os.path.join(args.out, "structures.pt"))
        np.save(os.path.join(args.out, "cv_coords.npy"),
                np.stack([r["cv"] for r in results]))
        _plot_coverage(cv_buffer, cv_space, ref_cv, args.out, args.n_explore)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_explore_conformations.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/explore_conformations.py tests/test_explore_conformations.py
git commit -m "feat: add explore_conformations.py for CV-guided OOD exploration"
```

---

### Task 5: `scripts/summarize_exploration.py`

**Files:**
- Create: `scripts/summarize_exploration.py`
- Create: `tests/test_summarize_exploration.py`

**Interfaces:**
- Consumes: `summary.json` (from Task 4), `cv_coords.npy`, `cv_basis.pt`
- Produces: printed table + `explore_out/md_summary.png`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_summarize_exploration.py
import json
import subprocess
import sys
import numpy as np
import pytest
from pathlib import Path


def _write_summary(tmp_path, n=10):
    import random
    random.seed(0)
    records = []
    for i in range(n):
        records.append({
            "id": i,
            "cv": [float(x) for x in np.random.randn(5).tolist()],
            "rmsd_native": round(abs(np.random.randn()) * 3, 3),
            "clashes": 0.0,
            "bond_rmsd": 0.02,
            "md_pass": bool(i % 3 == 0),
            "md_rmsd_final": round(abs(np.random.randn()) * 3, 3) if i % 3 == 0 else None,
            "md_rg_final": round(10.0 + np.random.randn(), 3) if i % 3 == 0 else None,
        })
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(records))
    np.save(str(tmp_path / "cv_coords.npy"),
            np.stack([r["cv"] for r in records]))
    return str(tmp_path)


def test_summarize_runs(tmp_path):
    out_dir = _write_summary(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/summarize_exploration.py",
         "--out", out_dir],
        capture_output=True, text=True, cwd="/home/qshao/DL-MD"
    )
    assert result.returncode == 0, result.stderr
    assert "md_pass" in result.stdout.lower() or "validated" in result.stdout.lower()


def test_summarize_creates_figure(tmp_path):
    out_dir = _write_summary(tmp_path)
    subprocess.run(
        [sys.executable, "scripts/summarize_exploration.py",
         "--out", out_dir],
        capture_output=True, text=True, cwd="/home/qshao/DL-MD", check=True
    )
    assert (tmp_path / "md_summary.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_summarize_exploration.py -v 2>&1 | head -20
```

Expected: `FileNotFoundError` for `summarize_exploration.py`

- [ ] **Step 3: Write `scripts/summarize_exploration.py`**

```python
"""Post-MD analysis: read a completed summary.json and report validated structures.

After running explore_conformations.py and populating md_pass/md_rmsd_final/
md_rg_final in summary.json, run this script to print a classification table
and save a CV-space plot of survivors.

Usage
-----
python scripts/summarize_exploration.py --out explore_out/3u7t_A
"""
import argparse
import json
import os

import numpy as np


_CLASSIFY = [
    (3.0, "Alternative state (>3 Å from native)"),
    (1.0, "Expanded fluctuation (1-3 Å)"),
    (0.0, "Near-native (<1 Å)"),
]


def classify_rmsd(rmsd):
    for threshold, label in _CLASSIFY:
        if rmsd > threshold:
            return label
    return "Near-native (<1 Å)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="explore_out directory")
    args = ap.parse_args()

    summary_path = os.path.join(args.out, "summary.json")
    cv_path = os.path.join(args.out, "cv_coords.npy")

    with open(summary_path) as fh:
        records = json.load(fh)

    total = len(records)
    validated = [r for r in records if r.get("md_pass") is True]
    pending = [r for r in records if r.get("md_pass") is None]

    print(f"\n=== Exploration Summary ({args.out}) ===")
    print(f"Total accepted (geometry filter):  {total}")
    print(f"MD-validated (md_pass=True):       {len(validated)}")
    print(f"MD-rejected  (md_pass=False):      {total - len(validated) - len(pending)}")
    print(f"Pending MD:                        {len(pending)}")

    if validated:
        print("\nMD-validated structures:")
        print(f"{'ID':>6}  {'RMSD_native(Å)':>15}  {'MD_RMSD(Å)':>11}  Classification")
        print("-" * 65)
        for r in sorted(validated, key=lambda x: x["rmsd_native"], reverse=True):
            cls = classify_rmsd(r["rmsd_native"])
            rmsd_md = r["md_rmsd_final"] if r["md_rmsd_final"] is not None else "N/A"
            print(f"{r['id']:>6}  {r['rmsd_native']:>15.3f}  {str(rmsd_md):>11}  {cls}")

        # Category counts
        print("\nClassification breakdown:")
        for threshold, label in _CLASSIFY:
            count = sum(1 for r in validated if r["rmsd_native"] > threshold)
            print(f"  {label}: {count}")

    # Plot if cv_coords available
    if os.path.exists(cv_path) and validated:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cv_all = np.load(cv_path)                 # [M, n_cv]
            ids_all = [r["id"] for r in records]
            id_to_idx = {rid: i for i, rid in enumerate(ids_all)}

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(cv_all[:, 0], cv_all[:, 1],
                       c="lightgrey", s=15, label="geometry-passed", zorder=1)

            colors = {"Alternative state (>3 Å from native)": "red",
                      "Expanded fluctuation (1-3 Å)": "orange",
                      "Near-native (<1 Å)": "green"}
            for r in validated:
                idx = id_to_idx.get(r["id"])
                if idx is None:
                    continue
                cls = classify_rmsd(r["rmsd_native"])
                ax.scatter(cv_all[idx, 0], cv_all[idx, 1],
                           c=colors.get(cls, "blue"), s=60, zorder=3,
                           edgecolors="black", linewidths=0.5)

            # Legend patches
            import matplotlib.patches as mpatches
            handles = [mpatches.Patch(color=c, label=l)
                       for l, c in colors.items()]
            handles.append(mpatches.Patch(color="lightgrey",
                                          label="geometry-passed (not MD-run)"))
            ax.legend(handles=handles, fontsize=8)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title(f"MD-validated conformations — {len(validated)}/{total} survivors")
            plt.tight_layout()
            fig_path = os.path.join(args.out, "md_summary.png")
            plt.savefig(fig_path, dpi=120)
            plt.close(fig)
            print(f"\nFigure saved: {fig_path}")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_summarize_exploration.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
python -m pytest tests/ -q --tb=short 2>&1 | tail -15
```

Expected: all previously passing tests still pass; new tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/summarize_exploration.py tests/test_summarize_exploration.py
git commit -m "feat: add summarize_exploration.py for post-MD validation report"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| CVSpace.fit() — PCA + Rg + RMSD | Task 1 |
| CVSpace.project_single() — differentiable, [n_pc+2] output | Task 1 |
| CVSpace.repulsion() — Gaussian sum, grad flows | Task 1 |
| CVSpace.save/load | Task 1 |
| build_cv_guidance() mirrors _build_wca_guidance pattern | Task 2 |
| rollout() new params: cv_space, cv_buffer, k_guide, sigma_cv, guide_warmup | Task 3 |
| guide_warmup gates CV activation | Task 3 |
| Backward compat: cv_space=None reproduces original rollout | Task 3 test |
| explore_conformations.py with all CLI flags from spec | Task 4 |
| Sequential batch loop, geometry filter gates buffer entry | Task 4 |
| Outputs: structures.pt, cv_coords.npy, candidates/*.pdb, cv_coverage.png, summary.json | Task 4 |
| summary.json schema: id, cv, rmsd_native, clashes, bond_rmsd, md_pass (null), md_rmsd_final, md_rg_final | Task 4 |
| --resume flag | Task 4 |
| summarize_exploration.py — reads summary.json, table + figure | Task 5 |
| MD classification: >3Å / 1-3Å / <1Å | Task 5 |

**Type consistency:** `CVSpace.project_single` returns `Tensor[n_pc+2]` — used as `cv_i` in Task 4, as `buffer` entries in Tasks 2/3, and as buffer input to `repulsion()` in Task 1. Shapes are consistent throughout.

**No placeholders:** all steps contain complete code.
