# Phase 3 — Conservative-Energy Retrain + FDT Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `v2_256h_90k` with two soft physics losses — an asymmetric energy-consistency loss against a data-fitted conservative energy, and a fluctuation-dissipation (step-variance) loss — so the propagator's stationary distribution approaches the MD Boltzmann ensemble and its relaxation timescale matches MD, without caging exploration.

**Architecture:** Two decoupled stages with a hard gate. Stage 1 fits a small learnable structured energy `U_θ` to corpus-pooled local CA statistics by denoising score-matching, then gates on whether short Langevin sampling from `U_θ` reproduces the MD free-energy surface. Stage 2 freezes `U_θ` and fine-tunes the propagator from `v2_256h_90k`, adding `energy_match_loss` (thermo) and `fdt_loss` (kinetics) on top of the unchanged DDPM loss, both warmed up from zero.

**Tech Stack:** PyTorch, the existing `lsmd` package (`cg_energy.py`, `physics_loss.py`, `transfer_train.py`, `transfer_validate.py`, `featurize.py`), pytest.

## Global Constraints

- **No hand-specified energy values** — all energetic parameters are learned; literature values (M&J matrix) are initialization only.
- **Fine-tune, not from scratch** — Stage 2 initializes from `checkpoints/v2_256h_90k.pt`.
- **Soft losses only** — no `drift = −∇U_θ` reparameterization.
- **Exploration over strict matching** — energy used softly and asymmetrically (hinge-dominant); DDPM sampler stochasticity kept intact.
- **Default `kT = 0.593` kcal/mol** (300 K) everywhere a temperature is needed, matching Phase 2.
- **`scripts/` has no `__init__.py`** — tests load scripts with `importlib.util.spec_from_file_location` (see `tests/test_validate_physics_cli.py`); CLI behavior is tested via `subprocess`.
- **WT/ data must never be pushed to GitHub** (repo is public).
- **The 6-D update layout is `u = [local_trans(3), axis_angle(3)]`** and `apply_update(R, t, u)` returns `t_f = R @ local_trans + t`; CA displacement norm equals `‖u[:, :3]‖`.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `lsmd/cg_energy.py` | Owns `_wca_energy` (moved in from `transfer_eval.py`) | 1 |
| `lsmd/transfer_eval.py` | Re-export shim for `_wca_energy` | 1 |
| `lsmd/learned_energy.py` | **New** — `LearnedCGEnergy`, `score_matching_loss`, `inverse_density_weights`, `langevin_sample`, `frame_energy_cut`, `md_step_cov` | 2,3,4,6 |
| `scripts/fit_energy.py` | **New** — Stage-1 fit CLI + `--gate` | 5 |
| `lsmd/physics_loss.py` | Add `energy_match_loss`, `fdt_loss` | 7,8 |
| `lsmd/transfer_train.py` | Wire both losses into `train()`; extend `collate_physics`, `sample_example` | 9 |
| `scripts/train_transfer.py` | New CLI flags for energy/FDT | 10 |

---

## Task 1: Relocate `_wca_energy` into `cg_energy.py`

**Files:**
- Modify: `lsmd/cg_energy.py` (add `_wca_energy`, drop the local import in `total_cg_energy`)
- Modify: `lsmd/transfer_eval.py:32-62` (replace body with a re-export)
- Test: `tests/test_cg_energy.py` (append)

**Interfaces:**
- Produces: `lsmd.cg_energy._wca_energy(t_pred, chain_id, sigma=4.5, eps=0.3) -> scalar tensor`, also importable from `lsmd.transfer_eval` for backward compatibility.

- [ ] **Step 1: Write the failing test** — append to `tests/test_cg_energy.py`:

```python
def test_wca_energy_importable_from_both_modules():
    from lsmd.cg_energy import _wca_energy as wca_cge
    from lsmd.transfer_eval import _wca_energy as wca_te
    assert wca_cge is wca_te          # same object (re-export, not a copy)
    t = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    chain_id = torch.zeros(3, dtype=torch.long)
    e = wca_cge(t, chain_id)
    assert torch.isfinite(e) and e.ndim == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cg_energy.py::test_wca_energy_importable_from_both_modules -v`
Expected: FAIL (`_wca_energy` not in `lsmd.cg_energy`).

- [ ] **Step 3: Move the function.** Cut the entire `_wca_energy` function (currently `lsmd/transfer_eval.py:32-62`) and paste it into `lsmd/cg_energy.py` just below `import torch` (top of file). In `lsmd/transfer_eval.py`, replace the removed function with:

```python
from lsmd.cg_energy import _wca_energy  # moved to cg_energy in Phase 3; re-export
```

In `lsmd/cg_energy.py`, inside `total_cg_energy`, delete the line `from lsmd.transfer_eval import _wca_energy` (it is now a module-level symbol).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cg_energy.py -v`
Expected: PASS (all existing 16 tests + the new one). Also run `pytest tests/ -q` to confirm `transfer_eval` consumers (rollout WCA guidance) still import cleanly.

- [ ] **Step 5: Commit**

```bash
git add lsmd/cg_energy.py lsmd/transfer_eval.py tests/test_cg_energy.py
git commit -m "refactor: move _wca_energy into cg_energy.py with re-export shim"
```

---

## Task 2: `LearnedCGEnergy` module

**Files:**
- Create: `lsmd/learned_energy.py`
- Test: `tests/test_learned_energy.py`

**Interfaces:**
- Consumes: `cg_energy._wca_energy`, `cg_energy.angle_energy`, `cg_energy.mj_contact_energy` (from Task 1).
- Produces:
  - `LearnedCGEnergy()` — `nn.Module`; `forward(t, res_type, chain_id) -> scalar tensor`.
  - `LearnedCGEnergy.save(path)` and `LearnedCGEnergy.load(path, map_location="cpu") -> LearnedCGEnergy`.

- [ ] **Step 1: Write the failing test** — create `tests/test_learned_energy.py`:

```python
import math
import torch
from lsmd.learned_energy import LearnedCGEnergy
from lsmd import cg_energy as cge


def _toy_protein(seed=0):
    g = torch.Generator().manual_seed(seed)
    N = 12
    t = torch.randn(N, 3, generator=g) * 5.0
    res_type = torch.randint(0, 20, (N,), generator=g)
    chain_id = torch.zeros(N, dtype=torch.long)
    return t, res_type, chain_id


def test_init_matches_cg_energy_defaults():
    t, rt, cid = _toy_protein()
    e = LearnedCGEnergy()
    got = e(t, rt, cid)
    ref = cge.total_cg_energy(t, rt, cid)   # default w=1, k_angle=10, eps=0.3
    assert torch.allclose(got, ref, atol=1e-4)


def test_params_and_position_grads_flow():
    t, rt, cid = _toy_protein()
    e = LearnedCGEnergy()
    t = t.requires_grad_(True)
    out = e(t, rt, cid)
    out.backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()
    assert all(p.grad is not None for p in e.parameters())


def test_save_load_roundtrip(tmp_path):
    e = LearnedCGEnergy()
    with torch.no_grad():
        e.log_alpha_mj += 0.5
    p = tmp_path / "energy.pt"
    e.save(str(p))
    e2 = LearnedCGEnergy.load(str(p))
    assert torch.allclose(e2.log_alpha_mj, e.log_alpha_mj)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_learned_energy.py -v`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write the implementation** — create `lsmd/learned_energy.py`:

```python
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
```

Note: `cge.angle_energy` and `cge._wca_energy` multiply by their `k_angle`/`eps`
arguments internally, so passing the tensor coefficients keeps the energy
differentiable w.r.t. both positions and parameters.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_learned_energy.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add lsmd/learned_energy.py tests/test_learned_energy.py
git commit -m "feat: LearnedCGEnergy module (learnable structured conservative energy)"
```

---

## Task 3: Fit ingredients — `score_matching_loss` and `inverse_density_weights`

**Files:**
- Modify: `lsmd/learned_energy.py` (append both functions)
- Test: `tests/test_learned_energy.py` (append)

**Interfaces:**
- Produces:
  - `score_matching_loss(energy, t, res_type, chain_id, *, sigma=0.5, kT=0.593) -> scalar tensor` — denoising score-matching loss for one frame.
  - `inverse_density_weights(cv, *, bins=30, clip=10.0) -> [F] tensor` — per-frame weights, mean ≈ 1, clipped to `[1/clip, clip]`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_learned_energy.py`:

```python
from lsmd.learned_energy import score_matching_loss, inverse_density_weights


def test_score_matching_loss_finite_and_differentiable():
    t, rt, cid = _toy_protein(seed=1)
    e = LearnedCGEnergy()
    torch.manual_seed(0)
    loss = score_matching_loss(e, t, rt, cid, sigma=0.5)
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in e.parameters())


def test_score_matching_reduces_on_harmonic_toy():
    # A 1-param harmonic energy U = 0.5*c*||t||^2 ; score-matching should drive
    # c toward the value implied by the (zero-centred) data + noise.
    torch.manual_seed(0)
    data = torch.randn(200, 4, 3)            # zero-mean cloud

    class Harmonic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_c = torch.nn.Parameter(torch.tensor(2.0))   # start too stiff
        def forward(self, t, res_type, chain_id):
            return 0.5 * self.log_c.exp() * (t ** 2).sum()

    h = Harmonic()
    opt = torch.optim.Adam(h.parameters(), lr=0.05)
    rt = torch.zeros(4, dtype=torch.long); cid = torch.zeros(4, dtype=torch.long)
    first = None
    for step in range(300):
        opt.zero_grad()
        i = torch.randint(0, data.shape[0], ()).item()
        loss = score_matching_loss(h, data[i], rt, cid, sigma=0.5, kT=1.0)
        loss.backward(); opt.step()
        if step == 0:
            first = float(loss)
    # averaged loss should be well below the initial mis-specified loss
    avg_last = sum(float(score_matching_loss(h, data[j], rt, cid, sigma=0.5, kT=1.0))
                   for j in range(20)) / 20
    assert avg_last < first


def test_inverse_density_weights_upweight_sparse():
    # 100 points in a dense cluster + 5 sparse outliers
    dense = torch.zeros(100, 2)
    sparse = torch.tensor([[10.0, 10.0]]).repeat(5, 1) + torch.randn(5, 2) * 0.01
    cv = torch.cat([dense, sparse], dim=0)
    w = inverse_density_weights(cv, bins=20, clip=50.0)
    assert w.shape == (105,)
    assert w[100:].mean() > w[:100].mean()      # sparse outliers up-weighted
    assert (w >= 1.0 / 50.0).all() and (w <= 50.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_learned_energy.py -k "score_matching or inverse_density" -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Write the implementation** — append to `lsmd/learned_energy.py`:

```python
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
    F = cv.shape[0]
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_learned_energy.py -v`
Expected: PASS (6 tests total).

- [ ] **Step 5: Commit**

```bash
git add lsmd/learned_energy.py tests/test_learned_energy.py
git commit -m "feat: score-matching loss and inverse-density weights for energy fit"
```

---

## Task 4: `langevin_sample`

**Files:**
- Modify: `lsmd/learned_energy.py` (append)
- Test: `tests/test_learned_energy.py` (append)

**Interfaces:**
- Produces: `langevin_sample(energy, t0, res_type, chain_id, *, n_steps=2000, dt=1e-3, kT=0.593, stride=10) -> [S, N, 3] tensor` — overdamped Langevin samples (γ=1 reduced units) from `p ∝ exp(−U/kT)`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_learned_energy.py`:

```python
from lsmd.learned_energy import langevin_sample


def test_langevin_recovers_harmonic_statistics():
    # U = 0.5*k*||t - c||^2  → stationary p(t) is Gaussian, mean c, var kT/k
    k, kT = 2.0, 1.0
    c = torch.tensor([3.0, -1.0, 0.0])

    class Harmonic(torch.nn.Module):
        def forward(self, t, res_type, chain_id):
            return 0.5 * k * ((t - c) ** 2).sum()

    torch.manual_seed(0)
    t0 = torch.zeros(1, 3)
    rt = torch.zeros(1, dtype=torch.long); cid = torch.zeros(1, dtype=torch.long)
    samples = langevin_sample(Harmonic(), t0, rt, cid,
                              n_steps=20000, dt=5e-3, kT=kT, stride=5)
    flat = samples.reshape(-1, 3)
    assert torch.allclose(flat.mean(0), c, atol=0.2)
    assert abs(float(flat.var(0).mean()) - kT / k) < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_learned_energy.py::test_langevin_recovers_harmonic_statistics -v`
Expected: FAIL (function not defined).

- [ ] **Step 3: Write the implementation** — append to `lsmd/learned_energy.py`:

```python
def langevin_sample(energy, t0, res_type, chain_id, *,
                    n_steps=2000, dt=1e-3, kT=0.593, stride=10):
    """Overdamped Langevin sampling from p ∝ exp(-U/kT) (γ = 1, reduced units).

    Update: x ← x - dt·∇U(x) + sqrt(2·kT·dt)·N(0, I).
    Returns the collected samples [S, N, 3] (one every `stride` steps).
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_learned_energy.py -v`
Expected: PASS (7 tests total).

- [ ] **Step 5: Commit**

```bash
git add lsmd/learned_energy.py tests/test_learned_energy.py
git commit -m "feat: overdamped Langevin sampler for the learned energy"
```

---

## Task 5: `scripts/fit_energy.py` — Stage-1 fit CLI + `--gate`

**Files:**
- Create: `scripts/fit_energy.py`
- Test: `tests/test_fit_energy_cli.py`

**Interfaces:**
- Consumes: `LearnedCGEnergy`, `score_matching_loss`, `inverse_density_weights`, `langevin_sample` (Tasks 2–4); `transfer_validate.shared_pca`, `transfer_validate.project_cv`, `transfer_validate.fes_comparison`.
- Produces: a CLI writing `checkpoints/energy_theta.pt`; `--gate` prints `GATE: PASS` / `GATE: FAIL` plus `fes_js` and the energy–population Spearman ρ.

- [ ] **Step 1: Write the failing test** — create `tests/test_fit_energy_cli.py`:

```python
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "fit_energy.py")


def test_fit_energy_help_lists_flags():
    out = subprocess.run([sys.executable, SCRIPT, "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0
    for flag in ["--shard", "--sigma", "--kT", "--out", "--gate", "--gate_threshold"]:
        assert flag in out.stdout


SHARD = os.path.join(REPO, "data", "atlas", "3u7t_A.pt")


import pytest


@pytest.mark.skipif(not os.path.exists(SHARD), reason="atlas shard absent")
def test_fit_energy_runs_and_writes_checkpoint(tmp_path):
    out_path = tmp_path / "energy_theta.pt"
    out = subprocess.run(
        [sys.executable, SCRIPT, "--shard", SHARD, "--steps", "20",
         "--out", str(out_path)],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert out_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_energy_cli.py -v`
Expected: FAIL (script does not exist).

- [ ] **Step 3: Write the implementation** — create `scripts/fit_energy.py`:

```python
"""Stage 1 of Phase 3: fit the learnable conservative energy to MD CA frames.

Fits LearnedCGEnergy by denoising score-matching on corpus-pooled frames with
inverse-density weighting, then (optionally) gates on whether short Langevin
sampling from the fitted energy reproduces the MD free-energy surface.

Usage
-----
python scripts/fit_energy.py --shard data/atlas/3u7t_A.pt \\
    --steps 5000 --sigma 0.5 --kT 0.593 --out checkpoints/energy_theta.pt --gate
"""
import argparse
import torch

from lsmd.learned_energy import (LearnedCGEnergy, score_matching_loss,
                                 inverse_density_weights, langevin_sample)
from lsmd import transfer_validate as tv


def _load_frames(shard_paths):
    """Return a list of (t[F,N,3], res_type[N], chain_id[N]) per shard."""
    proteins = []
    for p in shard_paths:
        s = torch.load(p, map_location="cpu", weights_only=False)
        proteins.append((s["t"].float(), s["res_type"].long(), s["chain_id"].long()))
    return proteins


def fit(proteins, *, steps, sigma, kT, lr, bins=30, clip=10.0, seed=0):
    torch.manual_seed(seed)
    energy = LearnedCGEnergy()
    opt = torch.optim.Adam(energy.parameters(), lr=lr)
    # Precompute per-protein inverse-density weights over shared-PCA CV space.
    weights = []
    for t, _rt, _cid in proteins:
        mean, comps = tv.shared_pca(t[0], n_components=2)
        cv = tv.project_cv(t, mean, comps)
        weights.append(inverse_density_weights(cv, bins=bins, clip=clip))
    rng = torch.Generator().manual_seed(seed)
    for step in range(steps):
        pi = torch.randint(0, len(proteins), (), generator=rng).item()
        t, rt, cid = proteins[pi]
        fi = torch.randint(0, t.shape[0], (), generator=rng).item()
        opt.zero_grad()
        loss = weights[pi][fi] * score_matching_loss(
            energy, t[fi], rt, cid, sigma=sigma, kT=kT)
        loss.backward()
        opt.step()
        if step % max(1, steps // 10) == 0:
            print(f"step {step}  loss={float(loss):.4f}")
    return energy


def gate(energy, proteins, *, kT, threshold, n_steps=4000):
    """Return (passed, fes_js, rho). Uses the first protein as the reference."""
    from scipy.stats import spearmanr  # optional; fallback below if absent
    t, rt, cid = proteins[0]
    mean, comps = tv.shared_pca(t[0], n_components=2)
    samples = langevin_sample(energy, t[0].clone(), rt, cid,
                              n_steps=n_steps, dt=5e-3, kT=kT, stride=5)
    cv_model = tv.project_cv(samples, mean, comps)
    cv_md = tv.project_cv(t, mean, comps)
    fes = tv.fes_comparison(cv_model, cv_md)["fes_js"]
    # energy–population correlation: per-MD-frame energy vs that frame's basin count
    with torch.no_grad():
        e_per = torch.tensor([float(energy(t[i], rt, cid)) for i in
                              range(0, t.shape[0], max(1, t.shape[0] // 200))])
    rho = 0.0
    try:
        cv_sub = cv_md[::max(1, t.shape[0] // 200)][: e_per.shape[0]]
        # population proxy: negative distance density (denser = more populated)
        from lsmd.learned_energy import inverse_density_weights as idw
        pop = 1.0 / idw(cv_sub, bins=20, clip=1e6)
        rho = float(spearmanr(e_per.numpy(), pop.numpy()).correlation)
    except Exception:
        rho = float("nan")
    passed = fes < threshold
    return passed, fes, rho


def main():
    ap = argparse.ArgumentParser(description="Phase 3 Stage 1: fit conservative energy")
    ap.add_argument("--shard", action="append", required=True, dest="shards")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--kT", type=float, default=0.593)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="checkpoints/energy_theta.pt")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    args = ap.parse_args()

    proteins = _load_frames(args.shards)
    energy = fit(proteins, steps=args.steps, sigma=args.sigma, kT=args.kT, lr=args.lr)
    energy.save(args.out)
    print(f"saved energy to {args.out}")

    if args.gate:
        passed, fes, rho = gate(energy, proteins, kT=args.kT,
                                threshold=args.gate_threshold)
        print(f"GATE: {'PASS' if passed else 'FAIL'}  fes_js={fes:.3f}  rho={rho:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_energy_cli.py -v`
Expected: PASS (help test passes; the integration test passes if the shard is present, else SKIP).

- [ ] **Step 5: Commit**

```bash
git add scripts/fit_energy.py tests/test_fit_energy_cli.py
git commit -m "feat: fit_energy.py Stage-1 fit CLI with FES-match gate"
```

---

## Task 6: Precompute helpers — `frame_energy_cut` and `md_step_cov`

**Files:**
- Modify: `lsmd/learned_energy.py` (append)
- Test: `tests/test_learned_energy.py` (append)

**Interfaces:**
- Produces:
  - `frame_energy_cut(energy, t, res_type, chain_id, *, pct=95.0) -> float` — the `pct`-percentile of per-residue energy `U_θ(frame)/N` over MD frames.
  - `md_step_cov(t, dt_md_ps, tau_ps) -> float` — mean per-atom squared CA displacement at lag `round(tau_ps/dt_md_ps)`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_learned_energy.py`:

```python
from lsmd.learned_energy import frame_energy_cut, md_step_cov


def test_frame_energy_cut_is_percentile_per_residue():
    torch.manual_seed(0)
    t = torch.randn(50, 8, 3) * 5.0
    rt = torch.randint(0, 20, (8,)); cid = torch.zeros(8, dtype=torch.long)
    e = LearnedCGEnergy()
    cut95 = frame_energy_cut(e, t, rt, cid, pct=95.0)
    cut50 = frame_energy_cut(e, t, rt, cid, pct=50.0)
    assert isinstance(cut95, float) and cut95 >= cut50


def test_md_step_cov_matches_known_random_walk():
    # Brownian frames: x_{i+1} = x_i + step, step ~ N(0, s^2 I). One-step (lag=1)
    # mean squared displacement per coordinate ≈ s^2.
    torch.manual_seed(0)
    s = 0.3
    F, N = 4000, 5
    steps = torch.randn(F, N, 3) * s
    t = torch.cumsum(steps, dim=0)
    var = md_step_cov(t, dt_md_ps=1.0, tau_ps=1.0)   # lag = 1 frame
    assert abs(var - s * s) < 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_learned_energy.py -k "frame_energy_cut or md_step_cov" -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Write the implementation** — append to `lsmd/learned_energy.py`:

```python
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
        scalar float: mean over atoms/coords of Var(x_{i+lag} - x_i).
    """
    lag = max(1, int(round(float(tau_ps) / float(dt_md_ps))))
    disp = t[lag:] - t[:-lag]                  # [F-lag, N, 3]
    return float((disp ** 2).mean())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_learned_energy.py -v`
Expected: PASS (9 tests total).

- [ ] **Step 5: Commit**

```bash
git add lsmd/learned_energy.py tests/test_learned_energy.py
git commit -m "feat: frame_energy_cut and md_step_cov precompute helpers"
```

---

## Task 7: `energy_match_loss`

**Files:**
- Modify: `lsmd/physics_loss.py` (append)
- Test: `tests/test_physics_loss_phase3.py`

**Interfaces:**
- Consumes: `featurize.apply_update`; a frozen energy with signature `energy(t, res_type, chain_id)`.
- Produces:
  `energy_match_loss(R_cur, t_cur, u_denorm, res_type, protein_id, chain_id, energy, *, u_cut, u_denorm_target=None, w_hi=1.0, w_lo=0.05) -> scalar tensor`.

- [ ] **Step 1: Write the failing test** — create `tests/test_physics_loss_phase3.py`:

```python
import torch
from lsmd.physics_loss import energy_match_loss
from lsmd.learned_energy import LearnedCGEnergy
from lsmd import geometry as g


def _identity_frames(N):
    R = torch.eye(3).expand(N, 3, 3).contiguous()
    t = torch.zeros(N, 3)
    # spread CA along x at 3.8 Å so the reference geometry is physical
    t[:, 0] = torch.arange(N).float() * 3.8
    return R, t


def test_energy_match_zero_when_prediction_is_physical():
    N = 10
    R, t = _identity_frames(N)
    res_type = torch.zeros(N, dtype=torch.long)
    chain_id = torch.zeros(N, dtype=torch.long)
    protein_id = torch.zeros(N, dtype=torch.long)
    energy = LearnedCGEnergy()
    # zero update → predicted frame == current physical frame → low energy
    u = torch.zeros(N, 6)
    u_cut = 1e6                          # ceiling far above any physical energy
    loss = energy_match_loss(R, t, u, res_type, protein_id, chain_id, energy,
                             u_cut=u_cut)
    assert float(loss) == 0.0


def test_energy_match_positive_for_clashing_prediction():
    N = 10
    R, t = _identity_frames(N)
    res_type = torch.zeros(N, dtype=torch.long)
    chain_id = torch.zeros(N, dtype=torch.long)
    protein_id = torch.zeros(N, dtype=torch.long)
    energy = LearnedCGEnergy()
    # collapse all residues toward the origin via a large negative-x translation
    u = torch.zeros(N, 6)
    u[:, 0] = -t[:, 0]                   # local_trans cancels the x spread → clash
    u_cut = -10.0                        # low ceiling so the hinge activates
    loss = energy_match_loss(R, t, u, res_type, protein_id, chain_id, energy,
                             u_cut=u_cut)
    assert float(loss) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_physics_loss_phase3.py::test_energy_match_zero_when_prediction_is_physical -v`
Expected: FAIL (`energy_match_loss` not defined).

- [ ] **Step 3: Write the implementation** — append to `lsmd/physics_loss.py`:

```python
def energy_match_loss(R_cur, t_cur, u_denorm, res_type, protein_id, chain_id,
                      energy, *, u_cut, u_denorm_target=None, w_hi=1.0, w_lo=0.05):
    """Soft, mostly one-sided energy-consistency loss (Phase 3 Stage 2).

    For each protein in the union batch:
      - hinge term  w_hi · relu(U_θ(x_pred)/N - u_cut)   penalizes ONLY
        high-energy / unphysical predicted frames (zero below the ceiling, so
        novel low-energy basins are free), and
      - weak term   w_lo · relu(U_θ(x_pred)/N - U_θ(x_true)/N)  gently
        discourages predictions higher-energy than the real MD transition.

    Args:
        R_cur, t_cur:    [ΣN,3,3], [ΣN,3] current frames.
        u_denorm:        [ΣN,6] de-normalized predicted update.
        res_type:        [ΣN] CANONICAL residue indices.
        protein_id:      [ΣN] per-protein id (groups atoms; energy is per-protein).
        chain_id:        [ΣN] local chain id within each protein.
        energy:          frozen LearnedCGEnergy (no grad on its params).
        u_cut:           per-residue energy ceiling (scalar; from frame_energy_cut).
        u_denorm_target: [ΣN,6] de-normalized TRUE update (enables the weak term).
    Returns:
        scalar tensor (mean over proteins).
    """
    R_pred, t_pred = feat.apply_update(R_cur, t_cur, u_denorm)
    if u_denorm_target is not None:
        _, t_true = feat.apply_update(R_cur, t_cur, u_denorm_target)
    total = t_pred.new_zeros(())
    pids = protein_id.unique()
    for pid in pids:
        m = protein_id == pid
        n = int(m.sum())
        rt = res_type[m]
        cid = chain_id[m]
        u_pred = energy(t_pred[m], rt, cid) / max(n, 1)
        total = total + w_hi * torch.relu(u_pred - u_cut)
        if u_denorm_target is not None:
            u_tru = energy(t_true[m], rt, cid).detach() / max(n, 1)
            total = total + w_lo * torch.relu(u_pred - u_tru)
    return total / max(pids.numel(), 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_physics_loss_phase3.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add lsmd/physics_loss.py tests/test_physics_loss_phase3.py
git commit -m "feat: energy_match_loss (soft asymmetric energy consistency)"
```

---

## Task 8: `fdt_loss`

**Files:**
- Modify: `lsmd/physics_loss.py` (append)
- Test: `tests/test_physics_loss_phase3.py` (append)

**Interfaces:**
- Produces: `fdt_loss(u_denorm, protein_id, sigma_md_tau) -> scalar tensor`, where `sigma_md_tau` is a `[G]` tensor of per-protein MD target variances aligned to `protein_id.unique()` sorted order.

- [ ] **Step 1: Write the failing test** — append to `tests/test_physics_loss_phase3.py`:

```python
from lsmd.physics_loss import fdt_loss


def test_fdt_loss_zero_when_variance_matches_target():
    N = 200
    protein_id = torch.zeros(N, dtype=torch.long)
    torch.manual_seed(0)
    s2 = 0.09
    u = torch.zeros(N, 6)
    u[:, :3] = torch.randn(N, 3) * (s2 ** 0.5)   # translational variance ≈ s2
    target = torch.tensor([u[:, :3].pow(2).mean()])   # exact per-protein target
    loss = fdt_loss(u, protein_id, target)
    assert float(loss) < 1e-6


def test_fdt_loss_positive_when_diffusion_too_fast():
    N = 200
    protein_id = torch.zeros(N, dtype=torch.long)
    torch.manual_seed(0)
    u = torch.zeros(N, 6)
    u[:, :3] = torch.randn(N, 3) * 1.0            # large step variance
    target = torch.tensor([0.01])                  # MD is much slower
    loss = fdt_loss(u, protein_id, target)
    assert float(loss) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_physics_loss_phase3.py -k fdt -v`
Expected: FAIL (`fdt_loss` not defined).

- [ ] **Step 3: Write the implementation** — append to `lsmd/physics_loss.py`:

```python
def fdt_loss(u_denorm, protein_id, sigma_md_tau):
    """Fluctuation-dissipation step-variance matching (Phase 3 Stage 2).

    Matches the per-protein translational step variance of the predicted update
    to the MD one-step displacement variance at the same lag τ. Because
    apply_update gives t_f = R·local_trans + t and R is orthogonal, the CA
    displacement variance equals Var(u_denorm[:, :3]).

    Args:
        u_denorm:     [ΣN,6] de-normalized predicted update.
        protein_id:   [ΣN] per-protein id.
        sigma_md_tau: [G] MD target variances, aligned to sorted unique pids.
    Returns:
        scalar tensor (mean squared error over proteins).
    """
    pids = protein_id.unique()                      # sorted ascending
    total = u_denorm.new_zeros(())
    for gi, pid in enumerate(pids):
        m = protein_id == pid
        var_model = u_denorm[m][:, :3].pow(2).mean()
        var_target = sigma_md_tau[gi].to(var_model.dtype).to(var_model.device)
        total = total + (var_model - var_target) ** 2
    return total / max(pids.numel(), 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_physics_loss_phase3.py -v`
Expected: PASS (4 tests total).

- [ ] **Step 5: Commit**

```bash
git add lsmd/physics_loss.py tests/test_physics_loss_phase3.py
git commit -m "feat: fdt_loss (step-variance matching at lag tau)"
```

---

## Task 9: Wire energy + FDT losses into training

**Files:**
- Modify: `lsmd/transfer_train.py` (`collate_physics`, `sample_example`, `train`)
- Test: `tests/test_transfer_train_phase3.py`

**Interfaces:**
- Consumes: `energy_match_loss`, `fdt_loss` (Tasks 7–8); `LearnedCGEnergy.load`, `frame_energy_cut`, `md_step_cov` (Tasks 2,6); existing `lambda_schedule`, `ddpm_physics_loss`.
- Produces: `train(..., energy_ckpt=None, lam_energy=0.0, lam_fdt=0.0, phys_warmup=500, w_hi=1.0, w_lo=0.05)`; `collate_physics` additionally returns `res_type`, `chain_id` (local), `u_cut` `[G]`, `sigma_md_tau` `[G]`.

- [ ] **Step 1: Write the failing test** — create `tests/test_transfer_train_phase3.py`:

```python
import inspect
import torch
from lsmd import transfer_train as tt


def test_train_exposes_phase3_kwargs():
    params = inspect.signature(tt.train).parameters
    for name in ["energy_ckpt", "lam_energy", "lam_fdt", "phys_warmup", "w_hi", "w_lo"]:
        assert name in params


def test_collate_physics_carries_res_type_and_targets():
    # Two tiny examples; verify the new keys are present and shaped correctly.
    def ex(N, gi_res):
        return {
            "R_cur": torch.eye(3).expand(N, 3, 3).contiguous(),
            "t_cur": torch.randn(N, 3),
            "chain_id": torch.zeros(N, dtype=torch.long),
            "res_type": gi_res,
            "u_cut": 1.5,
            "sigma_md_tau": 0.04,
        }
    from lsmd.physics_loss import collate_physics
    examples = [ex(4, torch.zeros(4, dtype=torch.long)),
                ex(3, torch.ones(3, dtype=torch.long))]
    out = collate_physics(examples)
    assert out["res_type"].shape == (7,)
    assert out["chain_id"].shape == (7,)
    assert out["u_cut"].shape == (2,) and out["sigma_md_tau"].shape == (2,)
    assert torch.allclose(out["u_cut"], torch.tensor([1.5, 1.5]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_train_phase3.py -v`
Expected: FAIL (`collate_physics` lacks the new keys; `train` lacks the kwargs).

- [ ] **Step 3a: Extend `collate_physics`** in `lsmd/physics_loss.py`. Add `res_type`, local `chain_id`, and the per-protein `u_cut`/`sigma_md_tau` to the collected lists and the returned dict:

```python
def collate_physics(examples):
    R_cur, t_cur, chains, pids = [], [], [], []
    res_types, local_chains, u_cuts, sig_taus = [], [], [], []
    for gi, ex in enumerate(examples):
        R_cur.append(ex["R_cur"])
        t_cur.append(ex["t_cur"])
        cid = ex["chain_id"].long()
        if cid.numel() > 0 and int(cid.max()) >= 10_000:
            raise ValueError(
                f"chain_id values must be < 10_000 for global_chain encoding; "
                f"got max={int(cid.max())} in example {gi}")
        chains.append(gi * 10_000 + cid)
        local_chains.append(cid)
        pids.append(torch.full((cid.shape[0],), gi, dtype=torch.long))
        res_types.append(ex["res_type"].long())
        u_cuts.append(float(ex.get("u_cut", 0.0)))
        sig_taus.append(float(ex.get("sigma_md_tau", 0.0)))
    return {
        "R_cur": torch.cat(R_cur, dim=0),
        "t_cur": torch.cat(t_cur, dim=0),
        "global_chain": torch.cat(chains, dim=0),
        "protein_id": torch.cat(pids, dim=0),
        "res_type": torch.cat(res_types, dim=0),
        "chain_id": torch.cat(local_chains, dim=0),
        "u_cut": torch.tensor(u_cuts),
        "sigma_md_tau": torch.tensor(sig_taus),
    }
```

- [ ] **Step 3b: Attach per-shard targets in `sample_example`** (`lsmd/transfer_train.py`). After the example is built, copy the precomputed scalars from the shard (defaulting to 0.0 when absent, so non-Phase-3 training is unaffected):

```python
    ex = data.build_training_example(shard, start, tau_frames, k,
                                     temp_K=temp_K, reverse=reverse)
    ex["u_cut"] = float(shard.get("_u_cut", 0.0))
    ex["sigma_md_tau"] = float(shard.get("_sigma_md_tau", 0.0))
    return ex
```

(Confirm `build_training_example` already includes `"res_type"` and `"chain_id"`; the union node features rely on them, so they are present. If `"res_type"` is absent, add `ex["res_type"] = shard["res_type"]` here.)

- [ ] **Step 3c: Load energy + precompute targets, then add the losses in `train`.** Add the kwargs to the signature:

```python
def train(shards, *, lags_ps, k=12, hidden=128, layers=4, lr=1e-3,
          max_union_nodes=2000, accum=4, steps=1000, T_diff=200,
          norm_samples=256, device="cpu", seed=0, lam=0.0, lam_warmup=500,
          log_every=100, grad_clip=1.0, norm_shards=None,
          frame_weighted=True, compile_model=False, temp_schedule=None,
          temp_emb_dim=8, reverse_prob=0.0, resume_from=None,
          checkpoint_every=0, checkpoint_path=None,
          energy_ckpt=None, lam_energy=0.0, lam_fdt=0.0, phys_warmup=500,
          w_hi=1.0, w_lo=0.05):
```

After `scale`/`schedule` are set up and `shards` are loaded, before the training loop, precompute per-shard targets when an energy is provided:

```python
    energy = None
    if energy_ckpt is not None:
        from lsmd.learned_energy import (LearnedCGEnergy, frame_energy_cut,
                                         md_step_cov)
        energy = LearnedCGEnergy.load(energy_ckpt, map_location=device)
        for p in energy.parameters():
            p.requires_grad_(False)
        tau_ps0 = float(min(lags_ps))
        for s in shards:
            s["_u_cut"] = frame_energy_cut(
                energy, s["t"].float(), s["res_type"].long(),
                s["chain_id"].long(), pct=95.0)
            s["_sigma_md_tau"] = md_step_cov(
                s["t"].float(), float(s["dt"]), tau_ps0)
```

In the loss block, after the existing `lam_t` schedule, add the two physics λ and assemble the total. Replace the loss assignment so that when `energy` is enabled the physics terms are added (the existing `ddpm_physics_loss(..., lam=lam_t)` path is preserved for the geometric C1 penalty):

```python
        if i % accum == 0:
            lam_t = pl.lambda_schedule(i // accum, lam_warmup, lam)
            lam_e = pl.lambda_schedule(i // accum, phys_warmup, lam_energy)
            lam_f = pl.lambda_schedule(i // accum, phys_warmup, lam_fdt)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            temp_K = b_dev.get("temp_K")
            if lam_t == 0.0 and energy is None:
                loss = ddpm_loss_union(net, b_dev["u_target"] / scale, node_feats,
                                       edge_index, edge_feats, tau, batch,
                                       schedule, temp_K=temp_K) / accum
            else:
                phys = {kk: vv.to(device)
                        for kk, vv in pl.collate_physics(group).items()}
                loss = pl.ddpm_physics_loss(net, b_dev, phys, scale,
                                            schedule, lam=lam_t)
                if energy is not None and (lam_e > 0.0 or lam_f > 0.0):
                    u0_hat, u_denorm = pl.recover_u_denorm(
                        net, b_dev, scale, schedule)
                    u_tgt_denorm = b_dev["u_target"]
                    loss = loss + lam_e * pl.energy_match_loss(
                        phys["R_cur"], phys["t_cur"], u_denorm,
                        phys["res_type"], phys["protein_id"], phys["chain_id"],
                        energy, u_cut=float(phys["u_cut"].mean()),
                        u_denorm_target=u_tgt_denorm, w_hi=w_hi, w_lo=w_lo)
                    loss = loss + lam_f * pl.fdt_loss(
                        u_denorm, phys["protein_id"], phys["sigma_md_tau"])
                loss = loss / accum
```

- [ ] **Step 3d: Add the `recover_u_denorm` helper** to `lsmd/physics_loss.py` (factor the clean-estimate recovery already inside `ddpm_physics_loss` so the energy/FDT terms reuse identical noising). It must use the **same** RNG order as `ddpm_physics_loss`; simplest is to call it once and have `ddpm_physics_loss` reuse it in a follow-up refactor, but for this task expose a standalone that recomputes with its own fresh noise (acceptable — the physics terms act on the clean estimate, not the score):

```python
def recover_u_denorm(net, union, scale, schedule):
    """Return (u0_hat, u_denorm): the model's clean-update estimate and its
    de-normalized form, for use by the Phase 3 energy/FDT terms."""
    scale_dev = scale.to(device=union["u_target"].device)
    u_target = union["u_target"] / scale_dev
    batch = union["batch"].to(u_target.device)
    G = union["tau"].shape[0]
    T = schedule.T
    t_min = max(1, T // 20)
    t_idx = torch.randint(t_min, T + 1, (G,), device=u_target.device)
    t_nodes = t_idx[batch]
    eps = torch.randn_like(u_target)
    sqrt_ab = schedule.sqrt_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    sqrt_1mab = schedule.sqrt_one_minus_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    noisy = sqrt_ab * u_target + sqrt_1mab * eps
    s = (t_idx.float() / T).to(u_target.dtype)
    pred = net(noisy, s, union["node_feats"], union["edge_index"],
               union["edge_feats"], union["tau"], batch,
               temp_K=union.get("temp_K"))
    u0_hat = (noisy - sqrt_1mab * pred) / sqrt_ab.clamp_min(1e-8)
    return u0_hat, u0_hat * scale_dev
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_transfer_train_phase3.py tests/test_physics_loss_phase3.py -v`
Expected: PASS. Then run the full suite: `pytest tests/ -q` (expect all green; existing training tests unaffected because the new kwargs default to disabled).

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_train.py lsmd/physics_loss.py tests/test_transfer_train_phase3.py
git commit -m "feat: wire energy_match and fdt losses into train() (Phase 3 Stage 2)"
```

---

## Task 10: CLI flags in `scripts/train_transfer.py`

**Files:**
- Modify: `scripts/train_transfer.py` (argparse + the `tt.train(...)` call)
- Test: `tests/test_train_transfer_cli.py`

**Interfaces:**
- Consumes: `train(..., energy_ckpt, lam_energy, lam_fdt, phys_warmup, w_hi, w_lo)` (Task 9).
- Produces: CLI flags `--energy_ckpt`, `--lam_energy`, `--lam_fdt`, `--phys_warmup`, `--w_hi`, `--w_lo`.

- [ ] **Step 1: Write the failing test** — create `tests/test_train_transfer_cli.py`:

```python
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "train_transfer.py")


def test_train_transfer_help_lists_phase3_flags():
    out = subprocess.run([sys.executable, SCRIPT, "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0
    for flag in ["--energy_ckpt", "--lam_energy", "--lam_fdt", "--phys_warmup",
                 "--w_hi", "--w_lo"]:
        assert flag in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_transfer_cli.py -v`
Expected: FAIL (flags absent).

- [ ] **Step 3: Add the flags and pass them through.** In `scripts/train_transfer.py`, after the existing `--lam_warmup` argument (line ~43), add:

```python
    ap.add_argument("--energy_ckpt", default=None,
                    help="Frozen LearnedCGEnergy checkpoint (Phase 3 Stage 2).")
    ap.add_argument("--lam_energy", type=float, default=0.0,
                    help="Max energy-match loss weight (0 = disabled).")
    ap.add_argument("--lam_fdt", type=float, default=0.0,
                    help="Max FDT step-variance loss weight (0 = disabled).")
    ap.add_argument("--phys_warmup", type=int, default=500,
                    help="Gradient steps to ramp lam_energy/lam_fdt from 0.")
    ap.add_argument("--w_hi", type=float, default=1.0,
                    help="Energy-match hinge weight (unphysical excursions).")
    ap.add_argument("--w_lo", type=float, default=0.05,
                    help="Energy-match weak Boltzmann-consistency weight.")
```

In the `tt.train(...)` call (line ~139), add the new keyword arguments:

```python
                    energy_ckpt=args.energy_ckpt, lam_energy=args.lam_energy,
                    lam_fdt=args.lam_fdt, phys_warmup=args.phys_warmup,
                    w_hi=args.w_hi, w_lo=args.w_lo,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_transfer_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_transfer.py tests/test_train_transfer_cli.py
git commit -m "feat: Phase 3 CLI flags for energy/FDT fine-tuning"
```

---

## Task 11: Run Stage 1 (fit + gate), Stage 2 (fine-tune), and validate

**Files:**
- Create (data artifacts): `checkpoints/energy_theta.pt`, `checkpoints/v3_phase3.pt`, `validation_phase3.json`

This task produces artifacts and the go/no-go decision; it has a hard gate.

- [ ] **Step 1: Fit the energy and run the gate**

```bash
python scripts/fit_energy.py \
    --shard data/atlas/3u7t_A.pt --shard data/atlas/4p3a_B.pt \
    --shard data/atlas/1b2s_F.pt --shard data/atlas/2y4x_B.pt \
    --shard data/atlas/1z0b_A.pt --shard data/atlas/6ovk_R.pt \
    --steps 8000 --sigma 0.5 --kT 0.593 \
    --out checkpoints/energy_theta.pt --gate --gate_threshold 0.5
```

Expected: prints decreasing fit loss and a final `GATE: PASS fes_js=… rho=…` line.

**HARD GATE:** if the line reads `GATE: FAIL`, STOP. Do not run Stage 2. Escalate to the spec's documented fallback (expand the contact term to a learnable 20×20 matrix or a small neural energy head in `LearnedCGEnergy`, refit, re-gate). Record the failure in the progress ledger.

- [ ] **Step 2: Fine-tune the propagator (Stage 2), ~compute-heavy**

```bash
python scripts/train_transfer.py \
    --shards_dir data/atlas --split train \
    --resume checkpoints/v2_256h_90k.pt \
    --energy_ckpt checkpoints/energy_theta.pt \
    --lam_energy 0.1 --lam_fdt 0.1 --phys_warmup 1000 \
    --steps 20000 --out checkpoints/v3_phase3.pt
```

Expected: training runs to completion; logged loss stays finite (energy/FDT terms ramp in over the first 1000 steps).

- [ ] **Step 3: Validate on the 6 held-out proteins**

```bash
python scripts/validate_physics.py \
    --checkpoint checkpoints/v3_phase3.pt \
    --shard data/atlas/3u7t_A.pt --shard data/atlas/4p3a_B.pt \
    --shard data/atlas/1b2s_F.pt --shard data/atlas/2y4x_B.pt \
    --shard data/atlas/1z0b_A.pt --shard data/atlas/6ovk_R.pt \
    --steps 300 --tau_ps 2000 --diff_steps 20 --eta 1.0 \
    --noether \
    --out validation_phase3.json
```

Expected: completes; prints summary JSON.

- [ ] **Step 4: Compare against baseline and Mode A**

```bash
python scripts/compare_modes.py \
    validation_baseline.json validation_modeA.json validation_phase3.json
```

Expected: a delta table. **Success** = `relax_ratio` < 5, `fes_js` < 0.5, `pop_tv` < 0.35 (mean across proteins). Record whether each target is met; note any MD-unvisited model density and confirm (via `U_θ`) that it is low-energy/physical rather than a regression.

- [ ] **Step 5: Run the full test suite, then commit results**

```bash
pytest tests/ -q
git add validation_phase3.json
git commit -m "data: Phase 3 conservative-energy retrain validation results"
```

(`checkpoints/*.pt` are large — follow the repo's existing checkpoint-tracking convention; do not push WT/ data.)

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Move `_wca_energy` into `cg_energy.py` | 1 |
| `LearnedCGEnergy` (~6 learnable scalars, init from cg_energy defaults) | 2 |
| Denoising score-matching fit objective | 3 |
| Inverse-density weighting | 3 |
| Langevin sampler on `U_θ` (for the gate) | 4 |
| `scripts/fit_energy.py` fit + FES-match gate (energy–population ρ + Langevin fes_js) | 5 |
| `frame_energy_cut` (`u_cut`) precompute | 6 |
| `md_step_cov` (`sigma_md_tau`) precompute | 6 |
| `energy_match_loss` (hinge + weak Boltzmann, one-sided) | 7 |
| `fdt_loss` (step-variance at lag τ) | 8 |
| Init from `v2_256h_90k`, frozen `U_θ`, warmed-up λ | 9 |
| Extend `collate_physics` (res_type, targets) | 9 |
| CLI flags | 10 |
| Run fit→gate→fine-tune→validate; success criteria | 11 |
| Fallback energy form on gate failure | 11 (escalation) |
| `compare_modes.py` Phase 3 column | 11 (no code change — it already accepts N files) |

**Placeholder scan:** none — every code step shows complete code; thresholds (`gate_threshold`, `lam_energy`, `pct`) are explicit tunable values, not gaps.

**Type consistency:**
- `LearnedCGEnergy(t, res_type, chain_id) -> scalar` defined in Task 2, consumed identically in Tasks 3,4,5,6,7.
- `frame_energy_cut(...) -> float` (Task 6) → `u_cut` consumed by `energy_match_loss` (Task 7) and produced per-shard in Task 9.
- `md_step_cov(...) -> float` (Task 6) → `sigma_md_tau` `[G]` consumed by `fdt_loss` (Task 8) and assembled in Task 9.
- `energy_match_loss(R_cur, t_cur, u_denorm, res_type, protein_id, chain_id, energy, *, u_cut, u_denorm_target, w_hi, w_lo)` — same signature in Task 7 definition and Task 9 call.
- `fdt_loss(u_denorm, protein_id, sigma_md_tau)` — same in Task 8 and Task 9.
- `collate_physics` returns `res_type`, `chain_id`, `u_cut`, `sigma_md_tau` (Task 9) — consumed by the Task 9 loss block.
- `recover_u_denorm(net, union, scale, schedule) -> (u0_hat, u_denorm)` defined and consumed within Task 9.
