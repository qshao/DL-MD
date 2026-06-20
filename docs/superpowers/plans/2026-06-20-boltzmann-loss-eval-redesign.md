# Boltzmann-Correct Loss and Evaluation Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CFM loss with DDPM score matching, add inverse-density training correction and target augmentation, and add thermodynamic evaluation metrics (Ramachandran JS, PCA coverage, ensemble recall/novelty).

**Architecture:** `NoiseSchedule`, `ddpm_loss`, and `sample_ddpm` are additive to `model.py` — `FlowNet` is unchanged, only what the network predicts changes (noise ε instead of velocity). Density reweighting lives in `data.py`. Five new validation functions in `validation.py`. `demo.py` rewires `train()` and `run_demo()` to use the new functions.

**Tech Stack:** PyTorch 2.12, mdtraj, pytest. No new dependencies.

## Global Constraints

- Python 3.10+; PyTorch ≥ 2.0.
- All tensors default to `torch.float32`; device must be respected (no silent CPU fallback).
- `cfm_loss` and `sample` remain in `model.py` and must continue to pass existing tests — new functions are additive, not replacements in the API.
- `ensemble_overlap` is removed from `validation.py`; its test is removed.
- `diversity()` function signature is unchanged; only the report key in `demo.py` is renamed to `"diversity_rmsd"`.
- Run `pytest tests/ -q` after each task to confirm no regressions.

---

### Task 1: DDPM core — `NoiseSchedule`, `ddpm_loss`, `sample_ddpm`

**Files:**
- Modify: `lsmd/model.py` (append after line 224)
- Modify: `tests/test_model.py` (append after existing tests)

**Interfaces:**
- Produces: `NoiseSchedule(T=200)` — `nn.Module` with buffers `alphas_bar`, `sqrt_alphas_bar`, `sqrt_one_minus_alphas_bar`, `betas`, `posterior_variance`, each shape `[T+1]`
- Produces: `ddpm_loss(net, u_target, node_feats, edge_index, edge_feats, tau, schedule, pair_weights=None, sigma_aug=0.0)` → scalar
- Produces: `sample_ddpm(net, node_feats, edge_index, edge_feats, K, tau, schedule, steps=50, eta=1.0, sigma_init=1.0)` → `[K, N, 6]`
- Task 4 consumes all three.

- [ ] **Step 1: Write failing tests for `NoiseSchedule`**

Append to `tests/test_model.py`:

```python
# ── NoiseSchedule ──────────────────────────────────────────────────────────────

def test_noise_schedule_shape():
    sched = m.NoiseSchedule(T=100)
    for attr in ("alphas_bar", "sqrt_alphas_bar",
                 "sqrt_one_minus_alphas_bar", "posterior_variance"):
        assert getattr(sched, attr).shape == (101,), attr


def test_noise_schedule_values():
    sched = m.NoiseSchedule(T=100)
    # index 0 = ᾱ_0 = 1 (clean); index T = ᾱ_T ≈ 0 (fully noisy)
    assert abs(sched.alphas_bar[0].item() - 1.0) < 1e-5
    assert sched.alphas_bar[-1].item() < 0.01
    # sqrt_alphas_bar must be monotone decreasing
    assert (sched.sqrt_alphas_bar[1:] <= sched.sqrt_alphas_bar[:-1]).all()
    # all values non-negative
    assert (sched.posterior_variance >= 0).all()
```

- [ ] **Step 2: Run — expect `AttributeError: module 'lsmd.model' has no attribute 'NoiseSchedule'`**

```bash
pytest tests/test_model.py::test_noise_schedule_shape tests/test_model.py::test_noise_schedule_values -v
```

- [ ] **Step 3: Implement `NoiseSchedule`**

Append to `lsmd/model.py` (after the existing `sample` function):

```python
class NoiseSchedule(nn.Module):
    """Cosine DDPM noise schedule.

    Buffers are indexed 0..T: alphas_bar[t] = ᾱ_t where ᾱ_0 = 1 (clean),
    ᾱ_T ≈ 0 (fully noisy).  All buffers move with .to(device).
    """

    def __init__(self, T=200):
        super().__init__()
        self.T = T
        t = torch.arange(T + 1, dtype=torch.float32)
        s = 0.008  # offset prevents singularity at t=0
        f = torch.cos((t / T + s) / (1 + s) * (torch.pi / 2)) ** 2
        ab = f / f[0]                                          # [T+1]
        betas = torch.zeros(T + 1)
        betas[1:] = (1 - ab[1:] / ab[:-1]).clamp(0, 0.999)   # β_t
        post_var = torch.zeros(T + 1)
        post_var[1:] = (
            betas[1:] * (1 - ab[:-1]) / (1 - ab[1:]).clamp_min(1e-8)
        ).clamp_min(0)
        self.register_buffer("alphas_bar", ab)
        self.register_buffer("sqrt_alphas_bar", ab.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_bar",
                             (1 - ab).clamp_min(0).sqrt())
        self.register_buffer("betas", betas)
        self.register_buffer("posterior_variance", post_var)
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_model.py::test_noise_schedule_shape tests/test_model.py::test_noise_schedule_values -v
```

Expected: 2 passed.

- [ ] **Step 5: Write failing tests for `ddpm_loss`**

Append to `tests/test_model.py`:

```python
# ── ddpm_loss ──────────────────────────────────────────────────────────────────

def test_ddpm_loss_unbatched():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    u_target = torch.randn(6, 6)
    loss = m.ddpm_loss(net, u_target, nf, ei, ef, tau=50, schedule=sched)
    assert loss.shape == ()
    assert loss.item() > 0


def test_ddpm_loss_batched():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    B = 4
    u_target = torch.randn(B, 6, 6)
    tau_b = torch.tensor([10.0, 25.0, 50.0, 100.0])
    loss = m.ddpm_loss(net, u_target, nf, ei, ef, tau=tau_b, schedule=sched)
    assert loss.shape == ()
    assert loss.item() > 0


def test_ddpm_loss_weighted():
    torch.manual_seed(7)
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    u_target = torch.randn(4, 6, 6)
    tau_b = torch.tensor([10.0, 25.0, 50.0, 100.0])

    # Uniform weights == no weights
    torch.manual_seed(0)
    loss_no_w = m.ddpm_loss(net, u_target, nf, ei, ef, tau=tau_b, schedule=sched)
    torch.manual_seed(0)
    loss_unif = m.ddpm_loss(net, u_target, nf, ei, ef, tau=tau_b, schedule=sched,
                             pair_weights=torch.ones(4))
    assert torch.isclose(loss_no_w, loss_unif, atol=1e-5)

    # Non-uniform weights give different result
    torch.manual_seed(0)
    loss_w = m.ddpm_loss(net, u_target, nf, ei, ef, tau=tau_b, schedule=sched,
                          pair_weights=torch.tensor([0.0, 0.0, 2.0, 2.0]))
    assert not torch.isclose(loss_no_w, loss_w, atol=1e-3)


def test_ddpm_can_overfit_constant_target():
    torch.manual_seed(0)
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=64, layers=2)
    sched = m.NoiseSchedule(T=20)   # small T → fewer noise levels → easier to overfit
    u_target = torch.randn(6, 6)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        m.ddpm_loss(net, u_target, nf, ei, ef, tau=50, schedule=sched).backward()
        opt.step()
    samples = m.sample_ddpm(net, nf, ei, ef, K=8, tau=50, schedule=sched,
                             steps=20, eta=1.0)
    assert samples.shape == (8, 6, 6)
    assert (samples.mean(0) - u_target).abs().mean() < 0.5
```

- [ ] **Step 6: Run — expect `AttributeError: module 'lsmd.model' has no attribute 'ddpm_loss'`**

```bash
pytest tests/test_model.py::test_ddpm_loss_unbatched tests/test_model.py::test_ddpm_loss_batched tests/test_model.py::test_ddpm_loss_weighted tests/test_model.py::test_ddpm_can_overfit_constant_target -v
```

- [ ] **Step 7: Implement `ddpm_loss`**

Append to `lsmd/model.py` (after `NoiseSchedule`):

```python
def ddpm_loss(net, u_target, node_feats, edge_index, edge_feats, tau, schedule,
              pair_weights=None, sigma_aug=0.0):
    """DDPM ε-prediction loss.

    Supports [N, 6] (single pair) and [B, N, 6] (mini-batch) targets.
    Per-batch-item noise level t is sampled from [t_min, T] so the network
    learns the score at every noise level simultaneously.

    Args:
        net:          FlowNet instance.
        u_target:     [N, 6] or [B, N, 6] — clean target updates.
        node_feats:   [N, node_dim]
        edge_index:   [2, E]
        edge_feats:   [E, edge_dim]
        tau:          Scalar or [B] tensor — lag time(s) in frames.
        schedule:     NoiseSchedule instance (on same device as u_target).
        pair_weights: Optional [B] per-sample loss weights (density correction).
        sigma_aug:    Target augmentation noise scale (0 to disable).

    Returns:
        Scalar loss.
    """
    batched = u_target.dim() == 3
    if not batched:
        u_target = u_target.unsqueeze(0)    # [1, N, 6]
    B = u_target.shape[0]
    T = schedule.T
    t_min = max(1, T // 20)

    if sigma_aug > 0.0:
        u_target = u_target + sigma_aug * torch.randn_like(u_target)

    t_idx = torch.randint(t_min, T + 1, (B,), device=u_target.device)   # [B]
    eps = torch.randn_like(u_target)

    sqrt_ab   = schedule.sqrt_alphas_bar[t_idx].to(u_target.dtype)        # [B]
    sqrt_1mab = schedule.sqrt_one_minus_alphas_bar[t_idx].to(u_target.dtype)
    noisy_u = sqrt_ab[:, None, None] * u_target + sqrt_1mab[:, None, None] * eps

    s = (t_idx.float() / T).to(u_target.dtype)                            # [B]
    pred_eps = net(noisy_u, s, node_feats, edge_index, edge_feats, tau)

    per_sample = ((pred_eps - eps) ** 2).mean(dim=(-2, -1))               # [B]
    if pair_weights is not None:
        w = pair_weights.to(device=u_target.device, dtype=u_target.dtype)
        return (w * per_sample).mean()
    return per_sample.mean()
```

- [ ] **Step 8: Run — expect PASS (overfit test may take ~10 s)**

```bash
pytest tests/test_model.py::test_ddpm_loss_unbatched tests/test_model.py::test_ddpm_loss_batched tests/test_model.py::test_ddpm_loss_weighted tests/test_model.py::test_ddpm_can_overfit_constant_target -v
```

Expected: 4 passed.

- [ ] **Step 9: Write failing tests for `sample_ddpm`**

Append to `tests/test_model.py`:

```python
# ── sample_ddpm ────────────────────────────────────────────────────────────────

def test_sample_ddpm_shape():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    samples = m.sample_ddpm(net, nf, ei, ef, K=8, tau=50, schedule=sched, steps=10)
    assert samples.shape == (8, 6, 6)


def test_sample_ddpm_diverse():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    # eta=1 → stochastic reverse → samples must differ
    samples = m.sample_ddpm(net, nf, ei, ef, K=8, tau=50, schedule=sched,
                             steps=10, eta=1.0)
    assert samples.std(0).mean() > 0.0


def test_sample_ddpm_eta0_deterministic():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    sched = m.NoiseSchedule(T=50)
    torch.manual_seed(42)
    s1 = m.sample_ddpm(net, nf, ei, ef, K=4, tau=50, schedule=sched,
                        steps=10, eta=0.0)
    torch.manual_seed(42)
    s2 = m.sample_ddpm(net, nf, ei, ef, K=4, tau=50, schedule=sched,
                        steps=10, eta=0.0)
    assert torch.allclose(s1, s2)
```

- [ ] **Step 10: Run — expect `AttributeError: module 'lsmd.model' has no attribute 'sample_ddpm'`**

```bash
pytest tests/test_model.py::test_sample_ddpm_shape tests/test_model.py::test_sample_ddpm_diverse tests/test_model.py::test_sample_ddpm_eta0_deterministic -v
```

- [ ] **Step 11: Implement `sample_ddpm`**

Append to `lsmd/model.py` (after `ddpm_loss`):

```python
@torch.no_grad()
def sample_ddpm(net, node_feats, edge_index, edge_feats, K, tau, schedule,
                steps=50, eta=1.0, sigma_init=1.0):
    """DDPM/DDIM unified reverse-process sampler.

    Runs `steps` uniformly-strided denoising steps from t=T-1 down to t=0.
    eta=1.0 → full DDPM (stochastic, diverse, Boltzmann-stationary).
    eta=0.0 → DDIM (deterministic, faster, less diverse).
    sigma_init > 1.0 → broader prior for exploration beyond training data.

    Args:
        net:         FlowNet instance (in eval mode recommended).
        node_feats:  [N, node_dim]
        edge_index:  [2, E]
        edge_feats:  [E, edge_dim]
        K:           Number of samples to draw in parallel.
        tau:         Scalar lag time in frames.
        schedule:    NoiseSchedule (on same device as node_feats).
        steps:       Number of denoising steps.
        eta:         Stochasticity scale (1=DDPM, 0=DDIM).
        sigma_init:  Scale of the initial noise.

    Returns:
        samples: [K, N, 6]
    """
    T = schedule.T
    N = node_feats.shape[0]
    device = node_feats.device
    dtype = node_feats.dtype

    u = torch.randn(K, N, 6, device=device, dtype=dtype) * sigma_init

    # Strided timesteps T-1..0, steps+1 values
    t_full = torch.round(
        torch.linspace(T - 1, 0, steps + 1, device=device)
    ).long().clamp(0, T - 1)   # [steps+1]

    for i in range(steps):
        t = t_full[i].item()
        t_prev = t_full[i + 1].item()

        s = torch.tensor(t / T, dtype=dtype, device=device)
        eps_pred = net(u, s, node_feats, edge_index, edge_feats, tau)   # [K, N, 6]

        sqrt_ab_t   = schedule.sqrt_alphas_bar[t].to(dtype)
        sqrt_1mab_t = schedule.sqrt_one_minus_alphas_bar[t].to(dtype)
        ab_prev     = schedule.alphas_bar[t_prev].to(dtype)

        # Predicted clean update
        u0_hat = (u - sqrt_1mab_t * eps_pred) / sqrt_ab_t.clamp_min(1e-8)

        # DDPM/DDIM variance
        pv = schedule.posterior_variance[t].to(dtype) if t > 0 \
             else torch.tensor(0.0, dtype=dtype, device=device)
        sigma_t   = eta * pv.sqrt()
        dir_coeff = (1 - ab_prev - sigma_t ** 2).clamp_min(0.0).sqrt()

        z = torch.randn_like(u) if t > 0 else torch.zeros_like(u)
        u = ab_prev.sqrt() * u0_hat + dir_coeff * eps_pred + sigma_t * z

    return u
```

- [ ] **Step 12: Run all new model tests — expect PASS**

```bash
pytest tests/test_model.py -v
```

Expected: all tests pass (existing CFM tests still pass, 9 new tests added).

- [ ] **Step 13: Commit**

```bash
git add lsmd/model.py tests/test_model.py
git commit -m "feat: DDPM score matching — NoiseSchedule, ddpm_loss, sample_ddpm"
```

---

### Task 2: Frame density weights — `compute_frame_weights`

**Files:**
- Modify: `lsmd/data.py` (append after `time_split`)
- Modify: `tests/test_data.py` (append after existing tests)

**Interfaces:**
- Consumes: `frames` dict from `data.load_frames` — uses `frames["t"]` [F, N, 3]
- Produces: `compute_frame_weights(frames, n_pca=3, bins=30, density_clip=10.0)` → `[F]` float32 tensor with mean≈1
- Task 4 consumes this to build `pair_weights_all`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_data.py`:

```python
def test_compute_frame_weights_shape_and_mean(tmp_path):
    path = _tiny_traj(tmp_path)
    frames = d.load_frames(path, path)
    weights = d.compute_frame_weights(frames)
    F = frames["R"].shape[0]
    assert weights.shape == (F,)
    assert abs(weights.mean().item() - 1.0) < 0.01
    assert (weights > 0).all()


def test_compute_frame_weights_uniform():
    """Identical CA positions → uniform density → all weights equal."""
    F, N = 10, 4
    # All frames have the same CA positions
    t_identical = torch.zeros(F, N, 3)
    for i in range(N):
        t_identical[:, i, 0] = i * 3.8
    frames = {
        "R": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(F, N, 3, 3).clone(),
        "t": t_identical,
    }
    weights = d.compute_frame_weights(frames)
    assert weights.std().item() < 1e-3
    assert abs(weights.mean().item() - 1.0) < 0.01
```

- [ ] **Step 2: Run — expect `AttributeError: module 'lsmd.data' has no attribute 'compute_frame_weights'`**

```bash
pytest tests/test_data.py::test_compute_frame_weights_shape_and_mean tests/test_data.py::test_compute_frame_weights_uniform -v
```

- [ ] **Step 3: Implement `compute_frame_weights`**

Append to `lsmd/data.py`:

```python
def compute_frame_weights(frames, n_pca=3, bins=30, density_clip=10.0):
    """Inverse-density weights for training pairs (source-frame correction).

    Projects all CA frames to a 2D PCA space, bins into a density histogram,
    and weights each frame by 1/count so rare conformations are upweighted.
    Corrects for the over-representation of dominant MD basins.

    Args:
        frames:       dict from load_frames — uses frames["t"] [F, N, 3].
        n_pca:        Number of PCA components to compute (only PC1-PC2 used).
        bins:         Grid resolution for the 2D density histogram.
        density_clip: Max weight relative to mean (prevents extreme upweighting).

    Returns:
        weights: [F] float32 tensor, mean = 1.0.
    """
    ca = frames["t"].float()          # [F, N, 3]
    F, N, _ = ca.shape
    ca_flat = ca.reshape(F, -1)       # [F, N*3]
    ca_flat = ca_flat - ca_flat.mean(0, keepdim=True)

    _, _, Vt = torch.linalg.svd(ca_flat, full_matrices=False)   # Vt: [min(F,N*3), N*3]
    n_comp = min(n_pca, Vt.shape[0])
    pc = ca_flat @ Vt[:n_comp].T      # [F, n_comp]

    # 2D histogram in PC1-PC2
    lo = pc[:, :2].min(0).values      # [2]
    hi = pc[:, :2].max(0).values      # [2]
    span = (hi - lo).clamp_min(1e-8)

    x_bin = ((pc[:, 0] - lo[0]) / span[0] * bins).long().clamp(0, bins - 1)
    y_bin = ((pc[:, 1] - lo[1]) / span[1] * bins).long().clamp(0, bins - 1)
    bin_idx = x_bin * bins + y_bin    # [F]

    counts = torch.zeros(bins * bins)
    counts.scatter_add_(0, bin_idx, torch.ones(F))

    frame_counts = counts[bin_idx].clamp_min(1.0)   # [F]
    weights = 1.0 / frame_counts
    mean_w = weights.mean()
    weights = weights.clamp(max=mean_w * density_clip)
    weights = weights / weights.mean()
    return weights
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_data.py::test_compute_frame_weights_shape_and_mean tests/test_data.py::test_compute_frame_weights_uniform -v
```

Expected: 2 passed.

- [ ] **Step 5: Run full test suite — no regressions**

```bash
pytest tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add lsmd/data.py tests/test_data.py
git commit -m "feat: inverse-density frame weights for training pair reweighting"
```

---

### Task 3: Distributional validation metrics

**Files:**
- Modify: `lsmd/validation.py` — remove `ensemble_overlap`, add 5 new functions
- Modify: `tests/test_validation.py` — remove `test_ensemble_overlap_identical_is_high`, add 8 new tests

**Interfaces:**
- Consumes: `atoms [N, 4, 3]` (N, CA, C, O at axis 1) from `decoder.build_structure`
- Consumes: `atoms_model [K, N, 4, 3]`, `atoms_md [M, N, 4, 3]` for ensemble comparisons
- Produces: `backbone_torsions(atoms)` → `(phi [N-2], psi [N-2])`
- Produces: `ramachandran_js(atoms_model, atoms_md, bins=36)` → float ∈ [0, 1]
- Produces: `pca_js(atoms_model, atoms_md, n_components=2, bins=20)` → dict with keys `"js"` (float) and `"var_explained"` ([float, float])
- Produces: `ensemble_recall(atoms_model, atoms_md, r_ang=2.0)` → float ∈ [0, 1]
- Produces: `ensemble_novelty(atoms_model, atoms_md, r_ang=2.0)` → float ∈ [0, 1]
- Task 4 calls all five in `run_demo()`.

- [ ] **Step 1: Write failing tests**

Replace `tests/test_validation.py` entirely:

```python
import torch
import pytest
from lsmd import geometry as g
from lsmd import decoder as dec
from lsmd import validation as val


def _atoms(n_structs, n_res, t_base, rot_scale, seed=0):
    """Build [n_structs, n_res, 4, 3] atom tensor with reproducible noise."""
    torch.manual_seed(seed)
    R = g.so3_exp(torch.randn(n_structs, n_res, 3) * rot_scale)
    t = t_base.unsqueeze(0).expand(n_structs, -1, -1).clone()
    return torch.stack([dec.build_structure(R[k], t[k]) for k in range(n_structs)])


def _t_base(n_res=6):
    t = torch.zeros(n_res, 3)
    for i in range(n_res):
        t[i, 0] = i * 3.8
    return t


# ── existing tests (unchanged) ─────────────────────────────────────────────────

def test_geometry_metrics_keys():
    R = g.so3_exp(torch.randn(5, 3) * 0.1)
    t = torch.arange(5).float().unsqueeze(-1).repeat(1, 3) * 3.8
    atoms = dec.build_structure(R, t)
    mt = val.geometry_metrics(atoms)
    assert {"ca_bond_mean", "peptide_violation", "clash_count"} <= set(mt)


def test_diversity_zero_for_identical():
    R = g.so3_exp(torch.randn(4, 3) * 0.1)
    t = torch.randn(4, 3)
    atoms = dec.build_structure(R, t)
    stacked = atoms.unsqueeze(0).repeat(5, 1, 1, 1)
    assert val.diversity(stacked) < 1e-5


def test_baselines_shapes():
    R = g.so3_exp(torch.randn(6, 3) * 0.1)
    t = torch.randn(6, 3)
    uc = val.baseline_copy(R, t, K=4)
    un = val.baseline_noise(R, t, K=4, sigma=0.2)
    assert uc.shape == (4, 6, 6) and un.shape == (4, 6, 6)
    assert uc.abs().sum() < 1e-5


# ── backbone_torsions ──────────────────────────────────────────────────────────

def test_backbone_torsions_shape():
    t = _t_base(n_res=6)
    atoms = _atoms(1, 6, t, 0.1)[0]    # [6, 4, 3]
    phi, psi = val.backbone_torsions(atoms)
    assert phi.shape == (4,)            # N - 2 = 4
    assert psi.shape == (4,)


def test_backbone_torsions_range():
    t = _t_base(n_res=6)
    atoms = _atoms(1, 6, t, 0.5)[0]
    phi, psi = val.backbone_torsions(atoms)
    assert (phi >= -torch.pi).all() and (phi <= torch.pi).all()
    assert (psi >= -torch.pi).all() and (psi <= torch.pi).all()


# ── ramachandran_js ────────────────────────────────────────────────────────────

def test_ramachandran_js_identical():
    t = _t_base()
    atoms = _atoms(5, 6, t, 0.1)
    js = val.ramachandran_js(atoms, atoms.clone())
    assert js < 1e-5


def test_ramachandran_js_bounded():
    t = _t_base()
    atoms_model = _atoms(5, 6, t, 0.1, seed=0)
    atoms_md    = _atoms(5, 6, t, 1.5, seed=1)  # large rotations → different angles
    js = val.ramachandran_js(atoms_model, atoms_md)
    assert 0.0 <= js <= 1.0


# ── pca_js ─────────────────────────────────────────────────────────────────────

def test_pca_js_returns_dict():
    t = _t_base()
    atoms_model = _atoms(5, 6, t, 0.1, seed=0)
    atoms_md    = _atoms(5, 6, t, 0.1, seed=2)
    result = val.pca_js(atoms_model, atoms_md)
    assert set(result.keys()) == {"js", "var_explained"}
    assert 0.0 <= result["js"] <= 1.0
    assert len(result["var_explained"]) == 2
    assert all(0.0 <= v <= 1.0 for v in result["var_explained"])


# ── ensemble_recall / novelty ─────────────────────────────────────────────────

def test_ensemble_recall_perfect():
    t = _t_base()
    atoms = _atoms(5, 6, t, 0.01, seed=0)
    # Model = MD → every MD frame is within r of itself
    recall = val.ensemble_recall(atoms, atoms.clone(), r_ang=0.01)
    assert recall == 1.0


def test_ensemble_recall_zero():
    t_near = _t_base()
    t_far = t_near.clone(); t_far[:, 0] += 200.0   # shift x by 200 Å
    atoms_model = _atoms(5, 6, t_near, 0.01, seed=0)
    atoms_md    = _atoms(5, 6, t_far,  0.01, seed=1)
    recall = val.ensemble_recall(atoms_model, atoms_md, r_ang=2.0)
    assert recall == 0.0


def test_ensemble_novelty_zero():
    t = _t_base()
    atoms = _atoms(5, 6, t, 0.01, seed=0)
    # Model = MD clone → no sample is novel
    novelty = val.ensemble_novelty(atoms, atoms.clone(), r_ang=2.0)
    assert novelty == 0.0
```

- [ ] **Step 2: Run — expect multiple failures on new functions**

```bash
pytest tests/test_validation.py -v
```

Expected: `test_geometry_metrics_keys`, `test_diversity_zero_for_identical`, `test_baselines_shapes` PASS. All new tests FAIL with `AttributeError`.

- [ ] **Step 3: Remove `ensemble_overlap` from `lsmd/validation.py` and add the five new functions**

Edit `lsmd/validation.py` — delete lines 48-68 (`ensemble_overlap` function), then append:

```python
def backbone_torsions(atoms):
    """Compute backbone dihedral angles for interior residues.

    Args:
        atoms: [N, 4, 3] — atom order per residue: N, CA, C, O

    Returns:
        phi: [N-2] tensor in (-π, π]
        psi: [N-2] tensor in (-π, π]
    """
    def _dihedral(a, b, c, d):
        b1 = b - a; b2 = c - b; b3 = d - c
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        b2n = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        m1 = torch.cross(n1, b2n, dim=-1)
        return torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    # phi_i = dihedral(C(i-1), N(i), CA(i), C(i))  for i = 1..N-2
    phi = _dihedral(
        atoms[:-2, 2, :], atoms[1:-1, 0, :],
        atoms[1:-1, 1, :], atoms[1:-1, 2, :]
    )
    # psi_i = dihedral(N(i), CA(i), C(i), N(i+1))  for i = 1..N-2
    psi = _dihedral(
        atoms[1:-1, 0, :], atoms[1:-1, 1, :],
        atoms[1:-1, 2, :], atoms[2:, 0, :]
    )
    return phi, psi


def _batch_torsions(atoms_batch):
    """Vectorised backbone torsions over a batch of structures.

    Args:
        atoms_batch: [K, N, 4, 3]

    Returns:
        (phi, psi) each [K*(N-2)]
    """
    def _dihedral(a, b, c, d):  # each [K, N-2, 3]
        b1 = b - a; b2 = c - b; b3 = d - c
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        b2n = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        m1 = torch.cross(n1, b2n, dim=-1)
        return torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    phi = _dihedral(
        atoms_batch[:, :-2, 2, :], atoms_batch[:, 1:-1, 0, :],
        atoms_batch[:, 1:-1, 1, :], atoms_batch[:, 1:-1, 2, :]
    ).reshape(-1)
    psi = _dihedral(
        atoms_batch[:, 1:-1, 0, :], atoms_batch[:, 1:-1, 1, :],
        atoms_batch[:, 1:-1, 2, :], atoms_batch[:, 2:,   0, :]
    ).reshape(-1)
    return phi, psi


def ramachandran_js(atoms_model, atoms_md, bins=36):
    """Jensen-Shannon divergence between Ramachandran distributions.

    Pools φ,ψ from all K×(N-2) model angles and M×(N-2) MD angles, builds
    36×36 histograms over [-π, π]², and computes JS divergence.

    Args:
        atoms_model: [K, N, 4, 3]
        atoms_md:    [M, N, 4, 3]
        bins:        Grid resolution (10° at bins=36).

    Returns:
        JS divergence in [0, 1].  0 = identical, 1 = disjoint.
    """
    def _hist2d(phi, psi, bins):
        lo, hi = -torch.pi, torch.pi
        phi_b = ((phi - lo) / (hi - lo) * bins).long().clamp(0, bins - 1)
        psi_b = ((psi - lo) / (hi - lo) * bins).long().clamp(0, bins - 1)
        idx = phi_b * bins + psi_b
        h = torch.zeros(bins * bins, device=phi.device)
        h.scatter_add_(0, idx, torch.ones(len(phi), device=phi.device))
        h = h + 1e-8
        return h / h.sum()

    phi_m, psi_m = _batch_torsions(atoms_model)
    phi_d, psi_d = _batch_torsions(atoms_md)
    p = _hist2d(phi_m, psi_m, bins)
    q = _hist2d(phi_d, psi_d, bins)
    mix = 0.5 * (p + q)
    js = 0.5 * (p * torch.log(p / mix)).sum() + \
         0.5 * (q * torch.log(q / mix)).sum()
    return js.clamp(0.0, 1.0).item()


def pca_js(atoms_model, atoms_md, n_components=2, bins=20):
    """Jensen-Shannon divergence of 2D PCA density between two ensembles.

    Fits PCA on the MD CA ensemble, projects both ensembles, and computes
    JS divergence of the 2D density.

    Args:
        atoms_model:  [K, N, 4, 3]
        atoms_md:     [M, N, 4, 3]
        n_components: Number of PCA components (only first 2 used for JS).
        bins:         Histogram bins per axis.

    Returns:
        dict with keys:
            js:            JS divergence ∈ [0, 1]
            var_explained: [float, float] — per-component variance fraction
    """
    ca_model = atoms_model[:, :, 1, :].float()   # [K, N, 3]
    ca_md    = atoms_md[:,    :, 1, :].float()   # [M, N, 3]
    K, M = ca_model.shape[0], ca_md.shape[0]

    mu = ca_md.mean(0)                            # [N, 3]
    cm = (ca_model - mu).reshape(K, -1)
    cd = (ca_md    - mu).reshape(M, -1)

    _, s_vals, Vt = torch.linalg.svd(cd, full_matrices=False)
    total_var = (s_vals ** 2).sum().clamp_min(1e-8)
    var_explained = [(s_vals[i] ** 2 / total_var).item()
                     for i in range(min(n_components, len(s_vals)))]

    V = Vt[:n_components].T                      # [N*3, n_components]
    pm = cm @ V                                   # [K, n_components]
    pd = cd @ V                                   # [M, n_components]

    lo = pd[:, :2].min(0).values
    hi = pd[:, :2].max(0).values
    span = (hi - lo).clamp_min(1e-8)

    def _idx(proj):
        xb = ((proj[:, 0] - lo[0]) / span[0] * bins).long().clamp(0, bins - 1)
        yb = ((proj[:, 1] - lo[1]) / span[1] * bins).long().clamp(0, bins - 1)
        return (xb * bins + yb).cpu()

    def _hist(idx, n):
        h = torch.zeros(bins * bins)
        h.scatter_add_(0, idx, torch.ones(n))
        h = h + 1e-8
        return h / h.sum()

    p = _hist(_idx(pm), K)
    q = _hist(_idx(pd), M)
    mix = 0.5 * (p + q)
    js = (0.5 * (p * torch.log(p / mix)).sum() +
          0.5 * (q * torch.log(q / mix)).sum()).clamp(0.0, 1.0).item()
    return {"js": js, "var_explained": var_explained}


def ensemble_recall(atoms_model, atoms_md, r_ang=2.0):
    """Fraction of MD frames covered by at least one model sample within r_ang Å.

    Measures whether the model reproduces all conformational states the MD visits.
    recall = 1.0 → no mode collapse; recall < 0.8 → model missing states.

    Args:
        atoms_model: [K, N, 4, 3]
        atoms_md:    [M, N, 4, 3]
        r_ang:       CA-RMSD coverage radius in Angstrom.

    Returns:
        float in [0, 1].
    """
    ca_model = atoms_model[:, :, 1, :]   # [K, N, 3]
    ca_md    = atoms_md[:,    :, 1, :]   # [M, N, 3]
    M = ca_md.shape[0]
    covered = 0
    for m_idx in range(M):
        diff = ca_model - ca_md[m_idx].unsqueeze(0)    # [K, N, 3]
        rmsd = diff.norm(dim=-1).pow(2).mean(dim=-1).sqrt()   # [K]
        if rmsd.min().item() < r_ang:
            covered += 1
    return covered / M


def ensemble_novelty(atoms_model, atoms_md, r_ang=2.0):
    """Fraction of model samples with no MD neighbor within r_ang Å.

    Measures generalization beyond the training trajectory.
    High novelty + good geometry = beneficial extrapolation.
    High novelty + bad geometry  = hallucination.

    Args:
        atoms_model: [K, N, 4, 3]
        atoms_md:    [M, N, 4, 3]
        r_ang:       CA-RMSD novelty radius in Angstrom.

    Returns:
        float in [0, 1].
    """
    ca_model = atoms_model[:, :, 1, :]   # [K, N, 3]
    ca_md    = atoms_md[:,    :, 1, :]   # [M, N, 3]
    K = ca_model.shape[0]
    novel = 0
    for k_idx in range(K):
        diff = ca_md - ca_model[k_idx].unsqueeze(0)    # [M, N, 3]
        rmsd = diff.norm(dim=-1).pow(2).mean(dim=-1).sqrt()   # [M]
        if rmsd.min().item() >= r_ang:
            novel += 1
    return novel / K
```

- [ ] **Step 4: Run — expect all tests PASS**

```bash
pytest tests/test_validation.py -v
```

Expected: 11 tests pass (3 old + 8 new). `test_ensemble_overlap_identical_is_high` no longer exists.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -q
```

Expected: all pass. If `demo.py` imports `ensemble_overlap`, it will fail at import — check and remove that call in the next task.

- [ ] **Step 6: Commit**

```bash
git add lsmd/validation.py tests/test_validation.py
git commit -m "feat: distributional validation metrics — Ramachandran JS, PCA JS, recall, novelty"
```

---

### Task 4: Wire into `demo.py` and update smoke test

**Files:**
- Modify: `lsmd/demo.py` — rewrite `train()`, `run_demo()`, and CLI section
- Modify: `tests/test_demo.py` — update assertions in `test_run_demo_smoke`

**Interfaces:**
- Consumes: `m.NoiseSchedule`, `m.ddpm_loss`, `m.sample_ddpm` from Task 1
- Consumes: `data.compute_frame_weights` from Task 2
- Consumes: `val.ramachandran_js`, `val.pca_js`, `val.ensemble_recall`, `val.ensemble_novelty`, `val.backbone_torsions` from Task 3
- Produces: `train(frames, taus, epochs, k, hidden, layers, lr, clip, batch_size, T_diff, sigma_aug, density_clip, device)` → `(net, schedule, ctx)`
- Produces: `run_demo(...)` report with keys `model_geometry`, `diversity_rmsd`, `ramachandran_js`, `pca_js`, `pca_var_explained`, `ensemble_recall`, `ensemble_novelty`, `n_residues`, `n_md_reference`, `taus`, `infer_tau`

- [ ] **Step 1: Update `test_demo.py` with new assertions**

Replace `tests/test_demo.py` entirely:

```python
import numpy as np
import mdtraj as md
from lsmd import demo


def _tiny_traj(tmp_path, n_res=6, n_frames=60):
    top = md.Topology()
    chain = top.add_chain()
    for i in range(n_res):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        for name, elem in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]:
            top.add_atom(name, md.element.get_by_symbol(elem), res)
    rs = np.random.RandomState(0)
    xyz = np.zeros((n_frames, n_res * 4, 3), np.float32)
    for i in range(n_res):
        base = np.array([i * 0.38, 0, 0], np.float32)
        offs = rs.randn(4, 3).astype(np.float32) * 0.02
        for fr in range(n_frames):
            wobble = rs.randn(4, 3).astype(np.float32) * 0.01
            xyz[fr, i * 4:(i + 1) * 4] = base + offs + wobble
    p = tmp_path / "tiny.pdb"
    md.Trajectory(xyz, top).save_pdb(str(p))
    return str(p)


def test_run_demo_smoke(tmp_path):
    path = _tiny_traj(tmp_path)
    out = tmp_path / "out"
    report = demo.run_demo(
        path, path,
        taus=[3, 5, 8], infer_tau=5,
        out_dir=str(out), K=4, epochs=30, batch_size=8,
        T_diff=20,          # small T for speed
        diff_steps=5,       # few inference steps for speed
    )
    # Structure keys
    assert "model_geometry" in report
    assert "diversity_rmsd" in report
    assert "ramachandran_js" in report
    assert "pca_js" in report
    assert "ensemble_recall" in report
    assert "ensemble_novelty" in report
    assert "n_md_reference" in report
    assert report["taus"] == [3, 5, 8]
    assert report["infer_tau"] == 5
    # PDB files written
    pdbs = list(out.glob("future_*.pdb"))
    assert len(pdbs) == 4
    # Metrics in valid ranges
    assert 0.0 <= report["ramachandran_js"] <= 1.0
    assert 0.0 <= report["ensemble_recall"]  <= 1.0
    assert 0.0 <= report["ensemble_novelty"] <= 1.0
    assert report["diversity_rmsd"] >= 0.0
```

- [ ] **Step 2: Run — expect failures on missing keys in report**

```bash
pytest tests/test_demo.py::test_run_demo_smoke -v
```

Expected: FAIL — `"diversity_rmsd"` not in report (old key is `"diversity"`), `"ramachandran_js"` not found, etc.

- [ ] **Step 3: Rewrite `lsmd/demo.py`**

Replace the entire file:

```python
"""End-to-end demo: train FlowNet on a trajectory, sample futures, write PDBs, report metrics."""
import os
import time
import torch
from lsmd import data, featurize as f, model as m, decoder as dec, validation as val


def _build_ctx(frames, k):
    R0, t0 = frames["R"][0], frames["t"][0]
    edge_index = f.knn_graph(t0, k=k)
    edge_feats = f.edge_features(R0, t0, edge_index)
    node_feats = f.node_features(
        frames["res_type"], frames["chain_id"],
        frames["res_index"], frames["n_types"],
    )
    return node_feats, edge_index, edge_feats


def train(frames, taus, epochs, k, hidden, layers, lr,
          clip=1.0, batch_size=32, T_diff=200, sigma_aug=0.05,
          density_clip=10.0, device=None):
    """Train FlowNet with DDPM score matching on a loaded trajectory.

    Uses multi-lag mini-batch training with inverse-density reweighting
    (upweights rare conformations) and target augmentation (smooths discrete
    MD frames into a continuous distribution).

    Args:
        frames:       dict from data.load_frames
        taus:         list of integer lag values (frames)
        epochs:       number of training epochs
        k:            KNN neighbours for graph construction
        hidden:       FlowNet hidden dimension
        layers:       number of message-passing layers
        lr:           Adam learning rate
        clip:         max gradient norm (0 to disable)
        batch_size:   pairs per optimizer step
        T_diff:       DDPM noise schedule length
        sigma_aug:    target augmentation noise scale (0 to disable)
        density_clip: max density weight relative to mean
        device:       torch device; auto-selects CUDA if None

    Returns:
        (net, schedule, ctx) where ctx = (node_feats, edge_index, edge_feats)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}  taus={taus}  batch_size={batch_size}  T_diff={T_diff}")

    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)
    train_pairs, _ = data.time_split(pairs, val_frac=0.2)
    if train_pairs.shape[0] == 0:
        raise ValueError("No training pairs: taus too large relative to trajectory length.")

    # Inverse-density reweighting: upweight rare conformations
    frame_weights = data.compute_frame_weights(frames, density_clip=density_clip)  # [F]
    pair_weights_all = frame_weights[train_pairs[:, 0]]                             # [P]

    n_batches = (train_pairs.shape[0] + batch_size - 1) // batch_size
    print(f"  {train_pairs.shape[0]} training pairs → {n_batches} steps/epoch")

    node_feats, edge_index, edge_feats = _build_ctx(frames, k)
    node_feats  = node_feats.to(device)
    edge_index  = edge_index.to(device)
    edge_feats  = edge_feats.to(device)
    R_all = frames["R"].to(device)
    t_all = frames["t"].to(device)

    schedule = m.NoiseSchedule(T=T_diff).to(device)
    net = m.FlowNet(
        node_dim=node_feats.shape[1],
        edge_dim=edge_feats.shape[1],
        hidden=hidden,
        layers=layers,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        perm_idx = torch.randperm(train_pairs.shape[0])
        perm = train_pairs[perm_idx]
        perm_w = pair_weights_all[perm_idx]
        epoch_loss, n_steps, t0 = 0.0, 0, time.time()

        for start in range(0, perm.shape[0], batch_size):
            batch   = perm[start:start + batch_size]              # [B, 3] CPU
            batch_w = perm_w[start:start + batch_size].to(device) # [B]
            i_idx   = batch[:, 0]
            j_idx   = batch[:, 1]
            tau_b   = batch[:, 2].to(device=device, dtype=R_all.dtype)

            u_batch = f.relative_update(
                R_all[i_idx], t_all[i_idx], R_all[j_idx], t_all[j_idx]
            )   # [B, N, 6]

            opt.zero_grad()
            loss = m.ddpm_loss(
                net, u_batch, node_feats, edge_index, edge_feats,
                tau_b, schedule, pair_weights=batch_w, sigma_aug=sigma_aug,
            )
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1

        print(f"Epoch {epoch+1}/{epochs}  loss={epoch_loss/n_steps:.4f}  "
              f"t={time.time()-t0:.1f}s")

    return net, schedule, (node_feats, edge_index, edge_feats)


def run_demo(traj_path, top_path, taus, infer_tau, out_dir, K=8, epochs=50,
             k=8, hidden=64, layers=3, lr=1e-3, clip=1.0, batch_size=32,
             T_diff=200, diff_steps=50, eta=1.0, sigma_init=1.0,
             sigma_aug=0.05, density_clip=10.0, device=None):
    """Full demo: load trajectory, train DDPM on multi-lag pairs, sample K futures,
    write PDBs, compute distributional metrics.

    Args:
        traj_path:    path to trajectory file
        top_path:     path to topology file
        taus:         list of training lag values (frames)
        infer_tau:    lag time for inference (any value)
        out_dir:      output directory for PDB files
        K:            number of future structures to sample
        epochs:       number of training epochs
        k:            KNN neighbours
        hidden:       FlowNet hidden dimension
        layers:       number of message-passing layers
        lr:           Adam learning rate
        clip:         gradient clip norm
        batch_size:   pairs per optimizer step
        T_diff:       DDPM noise schedule length
        diff_steps:   reverse-process denoising steps
        eta:          DDPM stochasticity (1=full DDPM, 0=DDIM)
        sigma_init:   prior scale for reverse-process start
        sigma_aug:    target augmentation noise scale
        density_clip: max density weight relative to mean
        device:       torch device (auto-detected if None)

    Returns:
        dict with keys: model_geometry, diversity_rmsd, ramachandran_js,
                        pca_js, pca_var_explained, ensemble_recall,
                        ensemble_novelty, n_residues, n_md_reference,
                        taus, infer_tau
    """
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = data.load_frames(traj_path, top_path)
    net, schedule, (node_feats, edge_index, edge_feats) = train(
        frames, taus, epochs, k, hidden, layers, lr,
        clip=clip, batch_size=batch_size, T_diff=T_diff,
        sigma_aug=sigma_aug, density_clip=density_clip, device=device,
    )

    # Pick a source frame from val set with matching tau
    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)
    _, val_pairs = data.time_split(pairs, val_frac=0.2)
    matching = val_pairs[val_pairs[:, 2] == infer_tau]
    ref_pair = matching[0] if matching.shape[0] > 0 else val_pairs[0]
    i0 = int(ref_pair[0])

    R_t = frames["R"][i0].to(device)
    t_t = frames["t"][i0].to(device)

    # Sample K futures
    u = m.sample_ddpm(net, node_feats, edge_index, edge_feats, K=K,
                      tau=infer_tau, schedule=schedule,
                      steps=diff_steps, eta=eta, sigma_init=sigma_init)
    R_f, t_f = dec.decode_frames(R_t, t_t, u)

    # Build and write each sample
    res_names = ["ALA"] * frames["R"].shape[1]
    atoms_K = []
    for kk in range(K):
        atoms = dec.idealize(dec.build_structure(R_f[kk].cpu(), t_f[kk].cpu()))
        atoms_K.append(atoms)
        dec.write_pdb(atoms, res_names, os.path.join(out_dir, f"future_{kk}.pdb"))
    atoms_K = torch.stack(atoms_K, 0)   # [K, N, 4, 3]

    # Assemble MD reference ensemble from val end-frames matching infer_tau
    ref_end_frames = matching[:, 1][:128] if matching.shape[0] > 0 \
                     else val_pairs[:, 1][:128]
    md_atoms = torch.stack([
        dec.build_structure(frames["R"][int(j)].cpu(), frames["t"][int(j)].cpu())
        for j in ref_end_frames
    ])   # [M, N, 4, 3]

    pca_result = val.pca_js(atoms_K, md_atoms)
    report = {
        "model_geometry":    val.geometry_metrics(atoms_K[0]),
        "diversity_rmsd":    val.diversity(atoms_K),
        "ramachandran_js":   val.ramachandran_js(atoms_K, md_atoms),
        "pca_js":            pca_result["js"],
        "pca_var_explained": pca_result["var_explained"],
        "ensemble_recall":   val.ensemble_recall(atoms_K, md_atoms),
        "ensemble_novelty":  val.ensemble_novelty(atoms_K, md_atoms),
        "n_residues":        frames["R"].shape[1],
        "n_md_reference":    md_atoms.shape[0],
        "taus":              taus,
        "infer_tau":         infer_tau,
    }
    return report


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Long-stride protein MD demo CLI")
    ap.add_argument("--traj",        required=True,  help="Trajectory file path")
    ap.add_argument("--top",         required=True,  help="Topology file path")
    ap.add_argument("--taus",        type=int, nargs="+", default=[10, 25, 50, 100, 200],
                    help="Training lag schedule (frames).")
    ap.add_argument("--infer_tau",   type=int, default=None,
                    help="Lag time for inference. Defaults to max(taus).")
    ap.add_argument("--out",         default="demo_out", help="Output directory")
    ap.add_argument("--K",           type=int,   default=8)
    ap.add_argument("--epochs",      type=int,   default=50)
    ap.add_argument("--k",           type=int,   default=8,    help="KNN neighbours")
    ap.add_argument("--hidden",      type=int,   default=64)
    ap.add_argument("--layers",      type=int,   default=3)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--clip",        type=float, default=1.0,  help="Gradient clip norm")
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--T_diff",      type=int,   default=200,  help="DDPM noise levels")
    ap.add_argument("--diff_steps",  type=int,   default=50,   help="Reverse-process steps")
    ap.add_argument("--eta",         type=float, default=1.0,  help="DDPM stochasticity")
    ap.add_argument("--sigma_init",  type=float, default=1.0,  help="Prior scale")
    ap.add_argument("--sigma_aug",   type=float, default=0.05, help="Target augmentation noise")
    ap.add_argument("--density_clip",type=float, default=10.0, help="Max density weight")
    ap.add_argument("--device",      default=None,  help="cuda / cpu (auto if omitted)")
    args = ap.parse_args()

    infer_tau = args.infer_tau if args.infer_tau is not None else max(args.taus)
    dev = torch.device(args.device) if args.device else None
    rep = run_demo(
        args.traj, args.top, args.taus, infer_tau, args.out,
        K=args.K, epochs=args.epochs, k=args.k,
        hidden=args.hidden, layers=args.layers,
        lr=args.lr, clip=args.clip, batch_size=args.batch_size,
        T_diff=args.T_diff, diff_steps=args.diff_steps,
        eta=args.eta, sigma_init=args.sigma_init,
        sigma_aug=args.sigma_aug, density_clip=args.density_clip,
        device=dev,
    )
    print(json.dumps(rep, indent=2))
```

- [ ] **Step 4: Run smoke test — expect PASS**

```bash
pytest tests/test_demo.py::test_run_demo_smoke -v
```

Expected: PASS. May take ~60 s (30 epochs on CPU with a tiny trajectory).

- [ ] **Step 5: Run full test suite — expect all pass**

```bash
pytest tests/ -q
```

Expected: All tests pass. If any CFM-related test fails because `demo.py` no longer exports `sigma`, check that test_demo.py does not pass `sigma=` anywhere.

- [ ] **Step 6: Commit**

```bash
git add lsmd/demo.py tests/test_demo.py
git commit -m "feat: wire DDPM training, density reweighting, and distributional eval into demo"
```
