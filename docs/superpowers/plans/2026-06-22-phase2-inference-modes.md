# Phase 2 Inference Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Noether momentum projection (Mode A) and Boltzmann reweighting via MJ statistical contact potential (Mode B) as inference-only sampling modes, measured against the Phase 1 validation baseline.

**Architecture:** Mode A inserts a per-step per-chain momentum removal into `rollout()` via a new `noether=False` flag. Mode B post-processes a complete trajectory with a CG energy (angle + MJ contact) and resamples frames by Boltzmann weight. Both modes are wired into `validate_physics.py` as new CLI flags; a new `compare_modes.py` script prints delta tables against the baseline JSON.

**Tech Stack:** PyTorch ≥ 2.0, existing `lsmd` package, `pytest`, standard library only (no new dependencies).

## Global Constraints

- All new files live under `lsmd/` (library) or `scripts/` (CLI) or `tests/` (tests).
- CANONICAL residue ordering from `lsmd/vocab.py`: ALA=0 ARG=1 ASN=2 ASP=3 CYS=4 GLN=5 GLU=6 GLY=7 HIS=8 ILE=9 LEU=10 LYS=11 MET=12 PHE=13 PRO=14 SER=15 THR=16 TRP=17 TYR=18 VAL=19 UNK=20.
- MJ matrix: Miyazawa–Jernigan 1996 J. Mol. Biol. 256:623-644 Table 3, values in kT at 298 K multiplied by 0.592 kcal/mol to convert to kcal/mol.
- Angle energy: θ₀ = 2.094 rad (120°), k = 10.0 kcal/mol/rad².
- MJ contact filter: CA–CA distance < 8.0 Å **and** |seq_sep| > 3 **and** neither residue is UNK.
- Boltzmann weights: `w_i = exp(-(E_i - E_min)/kT)`; `n_eff = (Σwi)²/Σwi²`; `degenerate` when `n_eff < 0.1·F`.
- Mode B sets all kinetic report fields to `null` (resampling breaks time ordering).
- `mh_rollout` is a library function only — not exposed in the CLI in Phase 2.
- No new pip dependencies. No changes to the model architecture or checkpoint.
- Tests must be deterministic (manual seeds where needed) and run on CPU only.

---

## File Map

| File | Status | Responsibility |
|------|--------|---------------|
| `lsmd/noether.py` | Create | `noether_project` — per-chain linear + angular momentum removal |
| `lsmd/cg_energy.py` | Create | `angle_energy`, `mj_contact_energy`, `total_cg_energy`, `MJ_MATRIX` constant |
| `lsmd/transfer_modes.py` | Create | `reweight_boltzmann`, `resample_trajectory`, `mh_rollout` |
| `lsmd/transfer_eval.py` | Modify | Add `noether=False` kwarg + one call to `noether_project` |
| `scripts/validate_physics.py` | Modify | Add `--noether`, `--reweight`, `--kT_reweight`, `--w_angle`, `--w_mj`, `--w_wca_cg` flags; update `build_report`, `summarize` |
| `scripts/compare_modes.py` | Create | Delta table comparing two or more report JSONs |
| `tests/test_noether.py` | Create | 5 analytical tests for `noether_project` |
| `tests/test_cg_energy.py` | Create | 9 tests for CG energy + reweighting |

---

### Task 1: Noether Momentum Projection Module

**Files:**
- Create: `lsmd/noether.py`
- Create: `tests/test_noether.py`

**Interfaces:**
- Produces: `noether_project(t_old: Tensor[N,3], t_new: Tensor[N,3], chain_id: Tensor[N, long]) -> Tensor[N,3]`
- Consumed by: Task 2 (`lsmd/transfer_eval.py`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_noether.py
import math
import torch
import pytest
from lsmd.noether import noether_project


def _cross(a, b):
    return torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], dim=-1)


def test_noether_translation_removed():
    """Pure COM drift: all displacement is translation → output = t_old."""
    N = 10
    torch.manual_seed(0)
    t_old = torch.randn(N, 3)
    drift = torch.tensor([1.0, 2.0, 3.0])
    t_new = t_old + drift.unsqueeze(0)
    chain_id = torch.zeros(N, dtype=torch.long)
    t_out = noether_project(t_old, t_new, chain_id)
    assert (t_out - t_old).abs().max() < 1e-5


def test_noether_rotation_removed():
    """Pure rigid rotation: all displacement is rotational → output ≈ t_old."""
    N = 20
    angles = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    t_old = torch.stack([torch.cos(angles) * 5.0, torch.sin(angles) * 5.0,
                         torch.zeros(N)], dim=1)
    omega = torch.tensor([0.0, 0.0, 0.05])
    centroid = t_old.mean(dim=0)
    r_c = t_old - centroid
    delta_rot = _cross(omega.unsqueeze(0).expand(N, -1), r_c)
    t_new = t_old + delta_rot
    chain_id = torch.zeros(N, dtype=torch.long)
    t_out = noether_project(t_old, t_new, chain_id)
    assert (t_out - t_old).abs().max() < 1e-4


def test_noether_linear_momentum_zero():
    """Random update: net displacement (linear momentum) is zero after projection."""
    torch.manual_seed(42)
    N = 15
    t_old = torch.randn(N, 3)
    t_new = t_old + torch.randn(N, 3)
    chain_id = torch.zeros(N, dtype=torch.long)
    t_out = noether_project(t_old, t_new, chain_id)
    delta = t_out - t_old
    assert delta.sum(dim=0).abs().max() < 1e-5


def test_noether_angular_momentum_zero():
    """Random update: angular momentum is zero after projection."""
    torch.manual_seed(7)
    N = 15
    t_old = torch.randn(N, 3)
    t_new = t_old + torch.randn(N, 3)
    chain_id = torch.zeros(N, dtype=torch.long)
    t_out = noether_project(t_old, t_new, chain_id)
    delta = t_out - t_old
    centroid = t_old.mean(dim=0)
    r_c = t_old - centroid
    L = _cross(r_c, delta).sum(dim=0)
    assert L.abs().max() < 1e-4


def test_noether_two_chains_independent():
    """Each chain gets projected independently — both have L=0 and P=0."""
    torch.manual_seed(3)
    N = 20
    t_old = torch.randn(N, 3)
    t_new = t_old + torch.randn(N, 3)
    chain_id = torch.cat([torch.zeros(10, dtype=torch.long),
                          torch.ones(10, dtype=torch.long)])
    t_out = noether_project(t_old, t_new, chain_id)
    for c in [0, 1]:
        mask = chain_id == c
        delta_c = (t_out - t_old)[mask]
        assert delta_c.sum(dim=0).abs().max() < 1e-4, f"chain {c} linear momentum nonzero"
        centroid_c = t_old[mask].mean(dim=0)
        r_c = t_old[mask] - centroid_c
        L_c = _cross(r_c, delta_c).sum(dim=0)
        assert L_c.abs().max() < 1e-4, f"chain {c} angular momentum nonzero"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_noether.py -v
```
Expected: 5 errors — `ModuleNotFoundError: No module named 'lsmd.noether'`

- [ ] **Step 3: Implement `lsmd/noether.py`**

```python
# lsmd/noether.py
import torch


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched 3-D cross product a × b; last dim must be 3."""
    return torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], dim=-1)


def noether_project(t_old: torch.Tensor,
                    t_new: torch.Tensor,
                    chain_id: torch.Tensor) -> torch.Tensor:
    """Remove net linear and angular momentum per chain from a displacement.

    Applied after apply_update + bond_constraint inside rollout() to eliminate
    the spurious COM drift and rigid-body rotation the diffusion model adds.

    Args:
        t_old:    [N, 3] CA positions before this step (= traj[-1] in rollout).
        t_new:    [N, 3] CA positions after apply_update + bond_constraint.
        chain_id: [N] long, chain assignment (0-indexed, contiguous).

    Returns:
        [N, 3] corrected CA positions.
    """
    delta = (t_new - t_old).clone()

    for c in chain_id.unique():
        mask = (chain_id == c)
        d = delta[mask]               # [nc, 3]
        r = t_old[mask]               # [nc, 3]

        # Step 1 — zero linear momentum: subtract mean displacement
        d = d - d.mean(dim=0)

        # Step 2 — zero angular momentum
        centroid = r.mean(dim=0)
        r_c = r - centroid                                # [nc, 3]
        L = _cross(r_c, d).sum(dim=0)                    # [3] angular momentum
        # Inertia tensor: I = sum_i(|r_i|^2 * I_3 - r_i r_i^T)
        r2 = (r_c * r_c).sum(dim=-1)                     # [nc]
        I = (r2.sum() * torch.eye(3, device=d.device, dtype=d.dtype)
             - r_c.T @ r_c)                              # [3, 3]
        omega = torch.linalg.pinv(I) @ L                 # [3]
        nc = d.shape[0]
        d = d - _cross(omega.unsqueeze(0).expand(nc, -1), r_c)

        delta[mask] = d

    return t_old + delta
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_noether.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add lsmd/noether.py tests/test_noether.py
git commit -m "feat: add Noether momentum projection module (Mode A)"
```

---

### Task 2: Integrate Noether into rollout()

**Files:**
- Modify: `lsmd/transfer_eval.py:138-142, 232-235`

**Interfaces:**
- Consumes: `noether_project` from Task 1
- Produces: `rollout(..., noether: bool = False)` — same return shape `[steps+1, N, 3]`; consumed by Task 6

- [ ] **Step 1: Write the failing test**

```python
# tests/test_noether_integration.py
import inspect
import pytest
from lsmd import transfer_eval as te


def test_rollout_has_noether_parameter():
    sig = inspect.signature(te.rollout)
    assert "noether" in sig.parameters
    assert sig.parameters["noether"].default is False


def test_rollout_noether_does_not_change_shape():
    """rollout with noether=True returns same shape as noether=False."""
    import os, torch
    from lsmd import transfer_eval as te
    CKPT = "checkpoints/v2_256h_90k.pt"
    if not os.path.exists(CKPT):
        pytest.skip("checkpoint not available")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    net, sched, norm = te.load_checkpoint(ckpt, device="cpu")
    N = 8
    t0 = torch.randn(N, 3) * 5
    R0 = torch.eye(3).unsqueeze(0).expand(N, -1, -1).clone()
    res_type = torch.zeros(N, dtype=torch.long)
    chain_id = torch.zeros(N, dtype=torch.long)
    res_index = torch.arange(N)
    traj = te.rollout(net, sched, norm, R0, t0, res_type, chain_id, res_index,
                      steps=2, tau_ps=2000.0, k=4, diff_steps=2,
                      noether=True, device="cpu")
    assert traj.shape == (3, N, 3)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_noether_integration.py::test_rollout_has_noether_parameter -v
```
Expected: FAIL — `noether` not in signature

- [ ] **Step 3: Modify `lsmd/transfer_eval.py`**

Add import at the top of the file (after existing imports):

```python
from lsmd.noether import noether_project
```

Change the `rollout` signature at line 138–142 — add `noether=False` before `device`:

```python
@torch.no_grad()
def rollout(net, schedule, update_norm, R0, t0, res_type, chain_id, res_index,
            *, steps, tau_ps, k, diff_steps=50, eta=1.0, temp_K=300.0,
            bond_constraint_iters=5, max_update_norm=3.0,
            wca_sigma=4.5, wca_eps=0.3, wca_lam=0.05,
            noether=False, device="cpu"):
```

Insert after the bond constraint block (currently lines 232–234), before `traj.append`:

```python
        if bond_constraint_iters > 0:
            t = _apply_bond_constraint(t, ref_dists, chain_id,
                                       n_iter=bond_constraint_iters)
        if noether:
            t = noether_project(traj[-1], t, chain_id)
        traj.append(t.clone())
```

(`traj[-1]` is the position at the start of this step because frame 0 is appended before the loop.)

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_noether_integration.py -v
```
Expected: 2 PASSED (second test skipped if checkpoint absent, otherwise PASSED)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_eval.py tests/test_noether_integration.py
git commit -m "feat: add noether=False flag to rollout() (Mode A integration)"
```

---

### Task 3: CG Angle Energy

**Files:**
- Create: `lsmd/cg_energy.py` (initial, angle_energy only)
- Create: `tests/test_cg_energy.py` (initial, angle tests only)

**Interfaces:**
- Produces: `angle_energy(t: Tensor[N,3], chain_id: Tensor[N,long], k_angle=10.0, theta0=2.094) -> Tensor scalar`
- Consumed by: Task 4 (`total_cg_energy`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cg_energy.py
import math
import torch
import pytest
from lsmd.cg_energy import angle_energy


def test_angle_energy_equilibrium():
    """Triplet at exactly theta0=2.094 rad (120°) → energy = 0."""
    # v1 = CA_0 - CA_1 = (-3.8, 0, 0)
    # We need v2 = CA_2 - CA_1 such that angle(v1, v2) = 120°:
    #   cos(120°) = -0.5; v1_hat = (-1,0,0); need dot(v1_hat, v2_hat) = -0.5
    #   → v2_hat = (0.5, 0.866, 0)
    CA_0 = torch.tensor([-3.8, 0.0, 0.0])
    CA_1 = torch.tensor([0.0,  0.0, 0.0])
    CA_2 = torch.tensor([1.9,  3.29, 0.0])   # 3.8 * (0.5, 0.866, 0)
    t = torch.stack([CA_0, CA_1, CA_2])
    chain_id = torch.zeros(3, dtype=torch.long)
    E = angle_energy(t, chain_id, theta0=2.094)
    assert float(E.abs()) < 1e-3


def test_angle_energy_straight_chain():
    """Straight chain (180°) → maximum energy = k * (pi - theta0)^2."""
    CA_0 = torch.tensor([0.0, 0.0, 0.0])
    CA_1 = torch.tensor([3.8, 0.0, 0.0])
    CA_2 = torch.tensor([7.6, 0.0, 0.0])   # 180° angle
    t = torch.stack([CA_0, CA_1, CA_2])
    chain_id = torch.zeros(3, dtype=torch.long)
    E = angle_energy(t, chain_id, k_angle=10.0, theta0=2.094)
    expected = 10.0 * (math.pi - 2.094) ** 2  # ≈ 10.974 kcal/mol
    assert abs(float(E) - expected) < 1e-3


def test_angle_energy_two_triplets():
    """Chain of 4: two triplets each contribute independently."""
    # Use same 120° geometry repeated for both triplets
    # Triplet 0-1-2 at 120°, triplet 1-2-3 at 180°
    CA_0 = torch.tensor([-3.8, 0.0, 0.0])
    CA_1 = torch.tensor([0.0,  0.0, 0.0])
    CA_2 = torch.tensor([1.9,  3.29, 0.0])
    # Extend CA_3 straight from CA_2 direction: angle at CA_2 = 180°
    direction = (CA_2 - CA_1) / (CA_2 - CA_1).norm()
    CA_3 = CA_2 + direction * 3.8
    t = torch.stack([CA_0, CA_1, CA_2, CA_3])
    chain_id = torch.zeros(4, dtype=torch.long)
    E = angle_energy(t, chain_id, k_angle=10.0, theta0=2.094)
    # First triplet ≈ 0, second triplet = 10*(pi-2.094)^2
    expected = 10.0 * (math.pi - 2.094) ** 2
    assert abs(float(E) - expected) < 1e-2


def test_angle_energy_two_chains_no_cross():
    """Two chains of 2 residues each: no triplets → energy = 0."""
    t = torch.randn(4, 3)
    chain_id = torch.tensor([0, 0, 1, 1])
    E = angle_energy(t, chain_id)
    assert float(E.abs()) < 1e-6
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_cg_energy.py -v
```
Expected: 4 errors — `ModuleNotFoundError: No module named 'lsmd.cg_energy'`

- [ ] **Step 3: Implement `lsmd/cg_energy.py` (angle_energy only)**

```python
# lsmd/cg_energy.py
import torch

# ── Angle energy ──────────────────────────────────────────────────────────────

def angle_energy(t: torch.Tensor,
                 chain_id: torch.Tensor,
                 k_angle: float = 10.0,
                 theta0: float = 2.094) -> torch.Tensor:
    """Harmonic CA-CA-CA angle energy.

    Args:
        t:        [N, 3] CA positions.
        chain_id: [N] long chain assignment.
        k_angle:  force constant in kcal/mol/rad².
        theta0:   equilibrium angle in radians (2.094 rad = 120°).

    Returns:
        Scalar energy in kcal/mol.
    """
    E = t.new_zeros(())
    for c in chain_id.unique():
        mask = (chain_id == c).nonzero(as_tuple=True)[0]
        if mask.shape[0] < 3:
            continue
        pos = t[mask]                   # [nc, 3]
        v1 = pos[:-2] - pos[1:-1]      # [nc-2, 3]
        v2 = pos[2:]  - pos[1:-1]      # [nc-2, 3]
        cos_theta = (v1 * v2).sum(-1) / (
            v1.norm(dim=-1).clamp_min(1e-8) * v2.norm(dim=-1).clamp_min(1e-8)
        )
        theta = torch.acos(cos_theta.clamp(-1 + 1e-6, 1 - 1e-6))
        E = E + (k_angle * (theta - theta0) ** 2).sum()
    return E
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cg_energy.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add lsmd/cg_energy.py tests/test_cg_energy.py
git commit -m "feat: add CA-CA-CA angle energy to cg_energy module"
```

---

### Task 4: MJ Contact Energy and Total CG Energy

**Files:**
- Modify: `lsmd/cg_energy.py` (add MJ_MATRIX constant, `mj_contact_energy`, `total_cg_energy`)
- Modify: `tests/test_cg_energy.py` (add 5 MJ tests + 2 combined tests)

**Interfaces:**
- Consumes: `angle_energy` from Task 3; `_wca_energy` from `lsmd.transfer_eval` (private, acceptable in Phase 2)
- Produces: `MJ_MATRIX: Tensor[20,20]`, `mj_contact_energy(t, res_type, chain_id, cutoff=8.0) -> scalar`, `total_cg_energy(t, res_type, chain_id, ...) -> scalar`
- Consumed by: Task 5 (`reweight_boltzmann`)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cg_energy.py`

```python
# append to tests/test_cg_energy.py
from lsmd.cg_energy import mj_contact_energy, total_cg_energy, MJ_MATRIX


def test_mj_gly_gly_contact():
    """GLY(7)–GLY(7) pair at 6 Å, seq_sep=4 → energy = MJ_MATRIX[7,7]."""
    N = 5
    t = torch.zeros(N, 3)
    # Residues 1–3 far away (no contacts among themselves or with 0/4)
    t[1] = torch.tensor([100.0, 0.0, 0.0])
    t[2] = torch.tensor([200.0, 0.0, 0.0])
    t[3] = torch.tensor([300.0, 0.0, 0.0])
    t[4] = torch.tensor([6.0, 0.0, 0.0])   # GLY at index 0 and 4, dist=6 Å
    # res_type: GLY=7 at 0 and 4; UNK=20 at 1,2,3 (excluded from MJ)
    res_type = torch.tensor([7, 20, 20, 20, 7])
    chain_id = torch.zeros(N, dtype=torch.long)
    E = mj_contact_energy(t, res_type, chain_id)
    expected = float(MJ_MATRIX[7, 7])   # only one contact pair
    assert abs(float(E) - expected) < 1e-4


def test_mj_beyond_cutoff():
    """Pair at 9 Å (> cutoff 8 Å) → energy = 0."""
    N = 5
    t = torch.zeros(N, 3)
    t[1] = t[2] = t[3] = torch.tensor([500.0, 0.0, 0.0])
    t[4] = torch.tensor([9.1, 0.0, 0.0])  # dist > 8 Å
    res_type = torch.tensor([7, 20, 20, 20, 7])
    chain_id = torch.zeros(N, dtype=torch.long)
    E = mj_contact_energy(t, res_type, chain_id)
    assert float(E.abs()) < 1e-6


def test_mj_bonded_excluded():
    """Pair with seq_sep=2 (≤ 3) → energy = 0 even at 5 Å."""
    t = torch.zeros(3, 3)
    t[2] = torch.tensor([5.0, 0.0, 0.0])  # seq_sep(0,2)=2, dist=5 Å
    res_type = torch.tensor([7, 7, 7])
    chain_id = torch.zeros(3, dtype=torch.long)
    E = mj_contact_energy(t, res_type, chain_id)
    assert float(E.abs()) < 1e-6


def test_mj_unk_excluded():
    """UNK residue (index 20) is excluded from all contacts."""
    N = 5
    t = torch.zeros(N, 3)
    t[4] = torch.tensor([6.0, 0.0, 0.0])
    res_type = torch.tensor([20, 20, 20, 20, 7])  # index 0 is UNK
    chain_id = torch.zeros(N, dtype=torch.long)
    E = mj_contact_energy(t, res_type, chain_id)
    assert float(E.abs()) < 1e-6


def test_mj_matrix_is_symmetric():
    """MJ_MATRIX must be symmetric: MJ[i,j] == MJ[j,i]."""
    diff = (MJ_MATRIX - MJ_MATRIX.T).abs().max()
    assert diff < 1e-5


def test_mj_diagonal_negative():
    """All diagonal entries should be negative (self-contacts are favorable)."""
    diag = MJ_MATRIX.diagonal()
    assert (diag < 0).all()


def test_total_cg_energy_w_mj_zero():
    """With w_mj=0, total_cg_energy = angle + wca only (no MJ)."""
    N = 5
    t = torch.zeros(N, 3)
    for i in range(N):
        t[i, 0] = float(i) * 20.0   # wide spacing → WCA=0
    res_type = torch.zeros(N, dtype=torch.long)
    chain_id = torch.zeros(N, dtype=torch.long)
    E_full  = total_cg_energy(t, res_type, chain_id, w_mj=1.0, w_wca=0.0)
    E_nomj  = total_cg_energy(t, res_type, chain_id, w_mj=0.0, w_wca=0.0)
    # Both should equal the angle energy
    angle_E = angle_energy(t, chain_id)
    assert abs(float(E_nomj) - float(angle_E)) < 1e-4
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_cg_energy.py -v -k "mj or total"
```
Expected: 7 errors — `ImportError: cannot import name 'mj_contact_energy'`

- [ ] **Step 3: Add MJ matrix + contact energy + total energy to `lsmd/cg_energy.py`**

Append to the existing file:

```python
# ── MJ statistical contact potential ──────────────────────────────────────────
#
# Source: Miyazawa & Jernigan 1996, J. Mol. Biol. 256:623-644, Table 3.
# Original values are in kT units at 298 K; multiplied by 0.592 kcal/mol here.
# Matrix is indexed by lsmd.vocab.CANONICAL residue order (see CANONICAL_TO_PAPER
# mapping below).

def _build_mj_matrix() -> torch.Tensor:
    # Paper residue order: CYS MET PHE ILE LEU VAL TRP TYR ALA GLY THR SER ASN GLN ASP GLU HIS ARG LYS PRO
    # Lower-triangle (row i contains values for paper residues 0..i)
    _lower = [
        [-5.44],
        [-4.99, -5.46],
        [-5.80, -5.74, -7.26],
        [-5.50, -5.53, -6.84, -5.78],
        [-5.83, -6.02, -7.28, -6.67, -5.83],
        [-4.96, -4.91, -6.29, -5.96, -5.83, -5.52],
        [-6.47, -6.34, -9.03, -7.46, -7.68, -6.48, -9.73],
        [-6.20, -6.05, -7.80, -6.98, -7.08, -6.29, -8.80, -6.36],
        [-3.57, -3.94, -4.81, -4.91, -4.96, -4.04, -5.06, -4.66, -2.72],
        [-3.16, -3.39, -4.13, -3.78, -4.16, -3.38, -4.65, -4.13, -2.31, -3.02],
        [-3.11, -3.40, -4.28, -4.21, -4.34, -3.71, -4.70, -4.18, -2.78, -2.88, -3.64],
        [-2.86, -3.05, -4.02, -3.52, -3.92, -3.05, -4.20, -4.00, -2.36, -2.64, -2.99, -3.05],
        [-2.59, -3.07, -4.20, -3.76, -3.74, -3.14, -4.53, -3.75, -2.17, -2.83, -3.17, -2.73, -3.54],
        [-3.07, -3.11, -4.66, -4.19, -4.21, -3.49, -5.49, -4.31, -2.57, -2.93, -3.01, -2.57, -3.07, -4.27],
        [-2.57, -2.89, -4.43, -3.52, -3.28, -2.97, -4.48, -3.62, -1.95, -2.42, -2.48, -2.37, -2.96, -3.10, -2.30],
        [-2.89, -2.92, -4.20, -3.65, -3.31, -3.05, -4.66, -3.82, -2.01, -2.44, -2.69, -2.27, -2.84, -3.07, -3.20, -2.89],
        [-3.60, -3.98, -4.77, -4.63, -4.37, -3.90, -5.39, -4.85, -2.41, -3.01, -3.23, -2.87, -3.11, -3.62, -3.16, -3.06, -4.77],
        [-2.57, -3.12, -4.77, -4.34, -4.26, -3.63, -5.56, -4.50, -2.27, -2.64, -2.88, -2.42, -2.59, -3.33, -2.87, -2.99, -3.98, -3.98],
        [-1.95, -2.48, -3.36, -3.37, -3.48, -3.05, -3.82, -3.36, -1.62, -1.72, -2.03, -1.64, -2.14, -2.57, -2.48, -2.57, -2.85, -2.69, -3.37],
        [-3.07, -3.45, -4.25, -4.04, -4.20, -3.32, -4.65, -4.10, -2.03, -2.48, -2.75, -2.53, -2.84, -3.23, -2.41, -2.90, -3.73, -3.44, -2.40, -4.93],
    ]
    m = torch.zeros(20, 20)
    for i, row in enumerate(_lower):
        for j, val in enumerate(row):
            m[i, j] = val
            m[j, i] = val
    # Reindex: paper order → CANONICAL order
    # CANONICAL: ALA=0 ARG=1 ASN=2 ASP=3 CYS=4 GLN=5 GLU=6 GLY=7
    #            HIS=8 ILE=9 LEU=10 LYS=11 MET=12 PHE=13 PRO=14
    #            SER=15 THR=16 TRP=17 TYR=18 VAL=19
    # Paper: CYS(p0) MET(p1) PHE(p2) ILE(p3) LEU(p4) VAL(p5) TRP(p6) TYR(p7)
    #        ALA(p8) GLY(p9) THR(p10) SER(p11) ASN(p12) GLN(p13) ASP(p14) GLU(p15)
    #        HIS(p16) ARG(p17) LYS(p18) PRO(p19)
    canon_to_paper = torch.tensor([8, 17, 12, 14, 0, 13, 15, 9,
                                   16,  3,  4, 18, 1,  2, 19, 11,
                                   10,  6,  7,  5])
    m = m[canon_to_paper][:, canon_to_paper]
    return m * 0.592   # kT(298K) → kcal/mol


MJ_MATRIX: torch.Tensor = _build_mj_matrix()


def mj_contact_energy(t: torch.Tensor,
                      res_type: torch.Tensor,
                      chain_id: torch.Tensor,
                      cutoff: float = 8.0) -> torch.Tensor:
    """Miyazawa–Jernigan statistical contact energy.

    Sums MJ[res_i, res_j] for all pairs (i<j) where:
      - CA-CA distance < cutoff
      - |i - j| > 3  (exclude bonded neighbors)
      - neither residue is UNK (index 20)

    Args:
        t:        [N, 3] CA positions.
        res_type: [N] long, CANONICAL residue indices (0-19; 20=UNK excluded).
        chain_id: [N] long (unused beyond UNK filter — seq_sep is global index).
        cutoff:   contact distance in Å (default 8.0).

    Returns:
        Scalar energy in kcal/mol (negative = favorable).
    """
    N = t.shape[0]
    idx = torch.arange(N, device=t.device)

    diff = t.unsqueeze(0) - t.unsqueeze(1)          # [N, N, 3]
    dist2 = (diff * diff).sum(-1)                    # [N, N]

    upper_tri  = idx.unsqueeze(1) < idx.unsqueeze(0)
    in_contact = dist2 < cutoff * cutoff
    seq_sep_ok = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() > 3
    not_unk    = (res_type < 20).unsqueeze(1) & (res_type < 20).unsqueeze(0)

    mask = upper_tri & in_contact & seq_sep_ok & not_unk   # [N, N]

    ri = res_type.clamp(max=19)
    energies = MJ_MATRIX.to(t.device)[ri.unsqueeze(1), ri.unsqueeze(0)]  # [N, N]
    return (energies * mask.float()).sum()


# ── Combined CG energy ────────────────────────────────────────────────────────

def total_cg_energy(t: torch.Tensor,
                    res_type: torch.Tensor,
                    chain_id: torch.Tensor,
                    *,
                    wca_sigma: float = 4.5,
                    wca_eps: float = 0.3,
                    k_angle: float = 10.0,
                    theta0: float = 2.094,
                    mj_cutoff: float = 8.0,
                    w_wca: float = 1.0,
                    w_angle: float = 1.0,
                    w_mj: float = 1.0) -> torch.Tensor:
    """Sum of WCA + angle + MJ contact energies (kcal/mol).

    Args:
        t:         [N, 3] CA positions.
        res_type:  [N] long CANONICAL indices.
        chain_id:  [N] long chain assignment.
        w_wca/w_angle/w_mj: per-term weights (default 1.0).

    Returns:
        Scalar energy in kcal/mol.
    """
    from lsmd.transfer_eval import _wca_energy   # private; moved to cg_energy in Phase 3
    E = t.new_zeros(())
    if w_wca != 0.0:
        E = E + w_wca  * _wca_energy(t, chain_id, sigma=wca_sigma, eps=wca_eps)
    if w_angle != 0.0:
        E = E + w_angle * angle_energy(t, chain_id, k_angle=k_angle, theta0=theta0)
    if w_mj != 0.0:
        E = E + w_mj   * mj_contact_energy(t, res_type, chain_id, cutoff=mj_cutoff)
    return E
```

- [ ] **Step 4: Run all CG energy tests**

```bash
pytest tests/test_cg_energy.py -v
```
Expected: 11 PASSED

- [ ] **Step 5: Commit**

```bash
git add lsmd/cg_energy.py tests/test_cg_energy.py
git commit -m "feat: add MJ contact potential and total_cg_energy to cg_energy module"
```

---

### Task 5: Boltzmann Reweighting and MH Rollout

**Files:**
- Create: `lsmd/transfer_modes.py`
- Modify: `tests/test_cg_energy.py` (add reweighting tests)

**Interfaces:**
- Consumes: `total_cg_energy` from Task 4; `rollout` from `lsmd.transfer_eval`
- Produces:
  - `reweight_boltzmann(traj, res_type, chain_id, kT, **energy_kwargs) -> {"weights": Tensor[F], "n_eff": float, "degenerate": bool}`
  - `resample_trajectory(traj, weights, n_samples=500) -> Tensor[n_samples, N, 3]`
  - `mh_rollout(net, sched, norm, R0, t0, res_type, chain_id, res_index, *, steps, tau_ps, k, diff_steps, eta, temp_K, kT, noether, **energy_kwargs) -> Tensor[steps+1, N, 3]`
- Consumed by: Task 7 (`build_report` reweighting branch)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cg_energy.py`

```python
# append to tests/test_cg_energy.py
import math
from unittest.mock import patch
from lsmd import transfer_modes as tm
import lsmd.cg_energy as cge


def test_reweight_boltzmann_uniform():
    """All frames equal energy → uniform weights, N_eff = F, not degenerate."""
    F, N = 20, 4
    # All UNK, widely spaced → E ≈ 0 for every frame
    traj = torch.zeros(F, N, 3)
    for i in range(F):
        for j in range(N):
            traj[i, j] = torch.tensor([float(j) * 50, float(i) * 50, 0.0])
    res_type = torch.ones(N, dtype=torch.long) * 20   # UNK
    chain_id = torch.zeros(N, dtype=torch.long)
    result = tm.reweight_boltzmann(traj, res_type, chain_id, kT=0.593, w_wca=0.0)
    assert result["weights"].std() < 1e-4
    assert abs(result["n_eff"] - F) < 0.5
    assert not result["degenerate"]


def test_reweight_boltzmann_degenerate():
    """Frame 0 energy -1000 kcal/mol → single dominant frame → degenerate=True."""
    F, N = 100, 3
    traj = torch.zeros(F, N, 3)
    res_type = torch.zeros(N, dtype=torch.long)
    chain_id = torch.zeros(N, dtype=torch.long)
    call_count = [0]
    def mock_energy(t, rt, ci, **kw):
        i = call_count[0]; call_count[0] += 1
        return torch.tensor(-1000.0) if i == 0 else torch.tensor(0.0)
    with patch.object(cge, "total_cg_energy", side_effect=mock_energy):
        result = tm.reweight_boltzmann(traj, res_type, chain_id, kT=0.593)
    assert result["degenerate"]
    assert result["n_eff"] < 0.1 * F


def test_resample_trajectory_shape():
    F, N = 80, 6
    traj = torch.randn(F, N, 3)
    weights = torch.ones(F) / F
    resampled = tm.resample_trajectory(traj, weights, n_samples=50)
    assert resampled.shape == (50, N, 3)


def test_resample_trajectory_concentrated_weights():
    """All weight on frame 0 → all resampled frames equal frame 0."""
    F, N = 20, 4
    traj = torch.randn(F, N, 3)
    weights = torch.zeros(F); weights[0] = 1.0
    resampled = tm.resample_trajectory(traj, weights, n_samples=10)
    assert (resampled - traj[0]).abs().max() < 1e-6
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_cg_energy.py -v -k "reweight or resample"
```
Expected: 4 errors — `ModuleNotFoundError: No module named 'lsmd.transfer_modes'`

- [ ] **Step 3: Implement `lsmd/transfer_modes.py`**

```python
# lsmd/transfer_modes.py
import math
import torch
from lsmd.cg_energy import total_cg_energy


def reweight_boltzmann(traj: torch.Tensor,
                       res_type: torch.Tensor,
                       chain_id: torch.Tensor,
                       kT: float,
                       **energy_kwargs) -> dict:
    """Compute Boltzmann weights for a trajectory under the CG energy.

    Args:
        traj:      [F, N, 3] CA positions on CPU.
        res_type:  [N] long CANONICAL residue indices.
        chain_id:  [N] long chain assignment.
        kT:        thermal energy in kcal/mol (e.g. 0.593 at 300 K).
        **energy_kwargs: forwarded to total_cg_energy (w_wca, w_angle, w_mj, etc.)

    Returns:
        {"weights": Tensor[F] normalized, "n_eff": float, "degenerate": bool}
    """
    F = traj.shape[0]
    energies = torch.stack([
        total_cg_energy(traj[i], res_type, chain_id, **energy_kwargs)
        for i in range(F)
    ])                                          # [F]
    log_w = -energies / kT
    log_w = log_w - log_w.max()                # numerical stability
    w = torch.exp(log_w)
    w = w / w.sum()
    n_eff = float(w.sum().pow(2) / w.pow(2).sum())
    return {"weights": w, "n_eff": n_eff, "degenerate": n_eff < 0.1 * F}


def resample_trajectory(traj: torch.Tensor,
                        weights: torch.Tensor,
                        n_samples: int = 500) -> torch.Tensor:
    """Resample trajectory frames by Boltzmann weights (with replacement).

    Args:
        traj:      [F, N, 3] CA positions.
        weights:   [F] non-negative weights (need not be normalized).
        n_samples: number of resampled frames.

    Returns:
        [n_samples, N, 3] resampled trajectory.
    """
    idx = torch.multinomial(weights, n_samples, replacement=True)
    return traj[idx]


def mh_rollout(net, sched, norm, R0, t0, res_type, chain_id, res_index, *,
               steps: int,
               tau_ps: float,
               k: int,
               diff_steps: int = 20,
               eta: float = 1.0,
               temp_K: float = 300.0,
               kT: float = 0.593,
               noether: bool = True,
               **energy_kwargs) -> torch.Tensor:
    """Metropolis–Hastings rollout for rigorous equilibrium sampling.

    Proposes each step via rollout(steps=1) and accepts/rejects via
    exp(-ΔU/kT). Rotation matrices R are approximated: R at each accepted
    step is derived from the previous rollout call (R does not accumulate
    across MH rejections — this approximation affects proposal quality
    but not the acceptance criterion or the stationary distribution).

    Returns [steps+1, N, 3]. Library function only — not CLI-exposed in Phase 2.
    """
    from lsmd import transfer_eval as te
    R = R0.clone()
    t = t0.clone()
    traj = [t.clone()]
    E_cur = total_cg_energy(t, res_type, chain_id, **energy_kwargs)

    for _ in range(steps):
        prop = te.rollout(net, sched, norm, R, t, res_type, chain_id, res_index,
                          steps=1, tau_ps=tau_ps, k=k, diff_steps=diff_steps,
                          eta=eta, temp_K=temp_K, noether=noether)
        t_prop = prop[1]
        E_prop = total_cg_energy(t_prop, res_type, chain_id, **energy_kwargs)
        dU = float(E_prop - E_cur)
        if dU <= 0 or torch.rand(1).item() < math.exp(-dU / kT):
            t = t_prop
            E_cur = E_prop
        traj.append(t.clone())

    return torch.stack(traj)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cg_energy.py -v
```
Expected: 15 PASSED

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_modes.py tests/test_cg_energy.py
git commit -m "feat: add Boltzmann reweighting and MH rollout to transfer_modes"
```

---

### Task 6: validate_physics.py — Mode A (--noether flag)

**Files:**
- Modify: `scripts/validate_physics.py`

**Interfaces:**
- Consumes: `rollout(..., noether=False)` from Task 2
- Produces: `--noether` CLI flag; `settings["noether"]` key; `"mode"` field in report JSON

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_physics_modes.py
import json, subprocess, sys, os, pytest

CKPT = "checkpoints/v2_256h_90k.pt"
SHARD = "data/atlas/3u7t_A.pt"


def test_validate_physics_has_noether_flag():
    """--help output lists --noether flag."""
    result = subprocess.run([sys.executable, "scripts/validate_physics.py", "--help"],
                            capture_output=True, text=True)
    assert "--noether" in result.stdout


@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(SHARD)),
                    reason="data not available")
def test_validate_physics_noether_runs(tmp_path):
    out = tmp_path / "modeA.json"
    subprocess.run([sys.executable, "scripts/validate_physics.py",
                    "--checkpoint", CKPT, "--shard", SHARD,
                    "--steps", "3", "--tau_ps", "2000", "--diff_steps", "2",
                    "--noether", "--out", str(out)],
                   check=True)
    report = json.loads(out.read_text())
    assert report["settings"]["noether"] is True
    assert report["proteins"]["3u7t_A"]["reweight"] is None
    # kinetic fields are present (Mode A does not null them)
    assert report["proteins"]["3u7t_A"]["kinetic"]["relax_ratio"] is not None
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_validate_physics_modes.py::test_validate_physics_has_noether_flag -v
```
Expected: FAIL — `--noether` not in help output

- [ ] **Step 3: Modify `scripts/validate_physics.py`**

In `build_report`, change the `te.rollout(...)` call to pass `noether`:

```python
        traj = te.rollout(
            net, sched, norm, R0, t0,
            shard["res_type"], shard["chain_id"], shard["res_index"],
            steps=settings["steps"], tau_ps=settings["tau_ps"], k=k_eff,
            diff_steps=settings["diff_steps"], eta=settings["eta"],
            temp_K=settings["temp_K"],
            bond_constraint_iters=settings["bond_constraint_iters"],
            max_update_norm=settings["max_update_norm"],
            wca_sigma=settings["wca_sigma"], wca_eps=settings["wca_eps"],
            wca_lam=settings["wca_lam"],
            noether=settings.get("noether", False),
            device=device).cpu()
```

Also add `rep["reweight"] = None` after `rep["n_res"] = int(shard["n_res"])`:

```python
        rep["n_res"] = int(shard["n_res"])
        rep["reweight"] = None
        proteins[_protein_id(path)] = rep
```

In `main()`, add the new argparse argument after `--max_update_norm`:

```python
    ap.add_argument("--noether", action="store_true", default=False,
                    help="Apply Noether momentum projection after each step (Mode A).")
```

In the `settings` dict construction, add:

```python
        "noether": args.noether,
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_validate_physics_modes.py::test_validate_physics_has_noether_flag -v
pytest tests/test_validate_physics_modes.py::test_validate_physics_noether_runs -v
```
Expected: both PASSED (second skipped if data absent)

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_physics.py tests/test_validate_physics_modes.py
git commit -m "feat: add --noether (Mode A) flag to validate_physics.py"
```

---

### Task 7: validate_physics.py — Mode B (--reweight + null kinetics)

**Files:**
- Modify: `scripts/validate_physics.py`

**Interfaces:**
- Consumes: `reweight_boltzmann`, `resample_trajectory` from Task 5
- Produces: `--reweight`, `--kT_reweight`, `--w_angle`, `--w_mj`, `--w_wca_cg` flags; null kinetics in report; updated `summarize()` to skip None

- [ ] **Step 1: Write the failing tests** — append to `tests/test_validate_physics_modes.py`

```python
# append to tests/test_validate_physics_modes.py
import json
from scripts.validate_physics import summarize


def test_summarize_skips_none_relax_ratio():
    """summarize() ignores None relax_ratio entries (Mode B proteins)."""
    proteins = {
        "A": {"structural": {"rmsf_corr": 0.5, "dist_js": 0.01},
              "thermodynamic": {"fes_js": 0.4},
              "kinetic": {"relax_ratio": None}},
        "B": {"structural": {"rmsf_corr": 0.7, "dist_js": 0.02},
              "thermodynamic": {"fes_js": 0.6},
              "kinetic": {"relax_ratio": 2.0}},
    }
    s = summarize(proteins)
    assert abs(s["mean_rmsf_corr"] - 0.6) < 1e-6   # (0.5 + 0.7) / 2
    assert abs(s["mean_relax_ratio"] - 2.0) < 1e-6  # only protein B has non-None


def test_validate_physics_has_reweight_flags():
    result = subprocess.run([sys.executable, "scripts/validate_physics.py", "--help"],
                            capture_output=True, text=True)
    for flag in ["--reweight", "--kT_reweight", "--w_angle", "--w_mj", "--w_wca_cg"]:
        assert flag in result.stdout, f"missing flag {flag}"


@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(SHARD)),
                    reason="data not available")
def test_validate_physics_reweight_nulls_kinetics(tmp_path):
    out = tmp_path / "modeB.json"
    subprocess.run([sys.executable, "scripts/validate_physics.py",
                    "--checkpoint", CKPT, "--shard", SHARD,
                    "--steps", "3", "--tau_ps", "2000", "--diff_steps", "2",
                    "--noether", "--reweight", "--kT_reweight", "0.593",
                    "--out", str(out)],
                   check=True)
    report = json.loads(out.read_text())
    prot = report["proteins"]["3u7t_A"]
    assert prot["reweight"] is not None
    assert "n_eff" in prot["reweight"]
    assert prot["kinetic"]["relax_ratio"] is None
    assert prot["kinetic"]["msd_rmse"] is None
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_validate_physics_modes.py::test_summarize_skips_none_relax_ratio -v
pytest tests/test_validate_physics_modes.py::test_validate_physics_has_reweight_flags -v
```
Expected: both FAIL

- [ ] **Step 3: Update `scripts/validate_physics.py`**

Replace the entire `build_report` function:

```python
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
            wca_lam=settings["wca_lam"],
            noether=settings.get("noether", False),
            device=device).cpu()

        rw_info = None
        ca_for_validate = traj

        if settings.get("reweight", False):
            from lsmd.transfer_modes import reweight_boltzmann, resample_trajectory
            rw = reweight_boltzmann(
                traj, shard["res_type"], shard["chain_id"],
                kT=settings.get("kT_reweight", 0.593),
                w_wca=settings.get("w_wca_cg", 1.0),
                w_angle=settings.get("w_angle", 1.0),
                w_mj=settings.get("w_mj", 1.0))
            ca_for_validate = resample_trajectory(traj, rw["weights"])
            rw_info = {"n_eff": float(rw["n_eff"]),
                       "degenerate": bool(rw["degenerate"])}

        rep = tv.validate(ca_for_validate, shard["t"].float(),
                          tau_ps=settings["tau_ps"], dt_md_ps=float(shard["dt"]),
                          kT=settings["kT"], n_states=settings["n_states"])
        rep["n_res"] = int(shard["n_res"])
        rep["reweight"] = rw_info

        if settings.get("reweight", False):
            rep["kinetic"] = {k: None for k in rep["kinetic"]}

        proteins[_protein_id(path)] = rep
    return proteins
```

Replace `summarize` to skip None values:

```python
def summarize(proteins):
    """Mean headline metrics across proteins, skipping None (Mode B kinetics)."""
    def mean(getter):
        vals = [getter(p) for p in proteins.values() if getter(p) is not None]
        return float(sum(vals) / len(vals)) if vals else float("nan")
    return {
        "mean_rmsf_corr":   mean(lambda p: p["structural"]["rmsf_corr"]),
        "mean_dist_js":     mean(lambda p: p["structural"]["dist_js"]),
        "mean_fes_js":      mean(lambda p: p["thermodynamic"]["fes_js"]),
        "mean_relax_ratio": mean(lambda p: p["kinetic"]["relax_ratio"]),
    }
```

Add argparse flags in `main()` after `--noether`:

```python
    ap.add_argument("--reweight", action="store_true", default=False,
                    help="Post-process trajectory with Boltzmann reweighting (Mode B).")
    ap.add_argument("--kT_reweight", type=float, default=0.593,
                    help="kT for Boltzmann reweighting in kcal/mol (default 0.593).")
    ap.add_argument("--w_angle", type=float, default=1.0,
                    help="Weight on angle term in CG energy for reweighting.")
    ap.add_argument("--w_mj", type=float, default=1.0,
                    help="Weight on MJ contact term in CG energy for reweighting.")
    ap.add_argument("--w_wca_cg", type=float, default=1.0,
                    help="Weight on WCA term in CG energy for reweighting (distinct from --wca_lam).")
```

Add to `settings` dict:

```python
        "noether": args.noether,
        "reweight": args.reweight, "kT_reweight": args.kT_reweight,
        "w_angle": args.w_angle, "w_mj": args.w_mj, "w_wca_cg": args.w_wca_cg,
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_validate_physics_modes.py -v
```
Expected: non-data tests PASS; data-dependent tests PASS or SKIP

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_physics.py tests/test_validate_physics_modes.py
git commit -m "feat: add --reweight (Mode B) flags and null-kinetics to validate_physics.py"
```

---

### Task 8: compare_modes.py

**Files:**
- Create: `scripts/compare_modes.py`

**Interfaces:**
- Consumes: two or more report JSONs written by `validate_physics.py`
- Produces: printed delta table; no new library code

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare_modes.py
import json, subprocess, sys
import pytest


def test_compare_modes_runs(tmp_path):
    proteins = {
        "3u7t_A": {
            "structural":    {"rmsf_corr": 0.5, "dist_js": 0.01},
            "thermodynamic": {"fes_js": 0.60, "fes_rmse_kT": 0.8, "pop_tv": 0.4},
            "kinetic":       {"msd_rmse": 1.0, "acf_rmse": 0.05,
                              "relax_model_ps": 4000.0, "relax_md_ps": 3000.0,
                              "relax_ratio": 1.33},
            "reweight": None, "n_res": 46,
        }
    }
    base = {"heldout": False, "proteins": proteins, "summary": {}}
    modeA_proteins = {k: dict(v, kinetic=dict(v["kinetic"], relax_ratio=0.9))
                      for k, v in proteins.items()}
    modeA = {"heldout": False, "proteins": modeA_proteins, "summary": {}}
    modeB_proteins = {k: dict(v,
                               thermodynamic=dict(v["thermodynamic"], fes_js=0.30),
                               kinetic=dict(v["kinetic"], relax_ratio=None))
                      for k, v in proteins.items()}
    modeB = {"heldout": False, "proteins": modeB_proteins, "summary": {}}

    (tmp_path / "base.json").write_text(json.dumps(base))
    (tmp_path / "modeA.json").write_text(json.dumps(modeA))
    (tmp_path / "modeB.json").write_text(json.dumps(modeB))

    result = subprocess.run(
        [sys.executable, "scripts/compare_modes.py",
         str(tmp_path / "base.json"),
         str(tmp_path / "modeA.json"),
         str(tmp_path / "modeB.json")],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "relax_ratio" in result.stdout
    assert "fes_js" in result.stdout
    # Mode B relax_ratio is null → "null" in output
    assert "null" in result.stdout
    # Mode A improved relax_ratio: 1.33 → 0.9 → negative delta
    assert "-" in result.stdout
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_compare_modes.py -v
```
Expected: FAIL — `No such file or directory: 'scripts/compare_modes.py'`

- [ ] **Step 3: Implement `scripts/compare_modes.py`**

```python
#!/usr/bin/env python3
"""Compare validation reports across sampling modes.

Usage:
    python scripts/compare_modes.py baseline.json modeA.json [modeB.json ...]
"""
import argparse
import json


_METRICS = [
    ("relax_ratio", "kinetic",       "relax_ratio"),
    ("fes_js",      "thermodynamic", "fes_js"),
    ("pop_tv",      "thermodynamic", "pop_tv"),
    ("rmsf_corr",   "structural",    "rmsf_corr"),
    ("dist_js",     "structural",    "dist_js"),
]


def _mean(proteins, section, key):
    vals = [p[section][key] for p in proteins.values()
            if p[section].get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _pct(base, new):
    if base is None or new is None or base == 0:
        return "n/a"
    return f"{100 * (new - base) / abs(base):+.0f}%"


def _fmt(v):
    return "null" if v is None else f"{v:.3g}"


def main():
    ap = argparse.ArgumentParser(description="Compare validation mode reports")
    ap.add_argument("reports", nargs="+", help="JSON report files; first = baseline")
    args = ap.parse_args()

    loaded = []
    for path in args.reports:
        with open(path) as fh:
            loaded.append((path, json.load(fh)))

    names = [p.replace("validation_", "").replace(".json", "")
             for p, _ in loaded]
    W = max(12, max(len(n) for n in names) + 2)

    header = f"{'Metric':<14}" + "".join(f"{n:>{W}}" for n in names)
    if len(loaded) > 1:
        header += "".join(f"{'Δvs-' + names[0]:>{W}}" for n in names[1:])
    print(header)
    print("-" * len(header))

    base_proteins = loaded[0][1]["proteins"]
    for label, section, key in _METRICS:
        base_val = _mean(base_proteins, section, key)
        row = f"{label:<14}" + f"{_fmt(base_val):>{W}}"
        for _, rep in loaded[1:]:
            row += f"{_fmt(_mean(rep['proteins'], section, key)):>{W}}"
        for _, rep in loaded[1:]:
            row += f"{_pct(base_val, _mean(rep['proteins'], section, key)):>{W}}"
        print(row)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_compare_modes.py -v
```
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_modes.py tests/test_compare_modes.py
git commit -m "feat: add compare_modes.py delta table for sampling mode comparison"
```

---

### Task 9: Run Phase 2 Comparison and Commit Results

**Files:**
- Create: `validation_modeA.json`, `validation_modeB.json`

This task produces data artifacts rather than code. Run on the same 6 proteins used for the Phase 1 baseline.

- [ ] **Step 1: Run Mode A (Noether projection)**

```bash
python scripts/validate_physics.py \
    --checkpoint checkpoints/v2_256h_90k.pt \
    --shard data/atlas/3u7t_A.pt \
    --shard data/atlas/4p3a_B.pt \
    --shard data/atlas/1b2s_F.pt \
    --shard data/atlas/2y4x_B.pt \
    --shard data/atlas/1z0b_A.pt \
    --shard data/atlas/6ovk_R.pt \
    --steps 300 --tau_ps 2000 --diff_steps 20 --eta 1.0 \
    --noether \
    --out validation_modeA.json
```

Expected to complete; prints summary JSON.

- [ ] **Step 2: Run Mode B (Noether + reweighting)**

```bash
python scripts/validate_physics.py \
    --checkpoint checkpoints/v2_256h_90k.pt \
    --shard data/atlas/3u7t_A.pt \
    --shard data/atlas/4p3a_B.pt \
    --shard data/atlas/1b2s_F.pt \
    --shard data/atlas/2y4x_B.pt \
    --shard data/atlas/1z0b_A.pt \
    --shard data/atlas/6ovk_R.pt \
    --steps 300 --tau_ps 2000 --diff_steps 20 --eta 1.0 \
    --noether --reweight --kT_reweight 0.593 \
    --out validation_modeB.json
```

Expected to complete; kinetic metrics will be null in output.

- [ ] **Step 3: Print comparison table**

```bash
python scripts/compare_modes.py \
    validation_baseline.json \
    validation_modeA.json \
    validation_modeB.json
```

Expected output (example — actual numbers will vary):
```
Metric          baseline     modeA     modeB  Δvs-baseline  Δvs-baseline
relax_ratio        14.1       3.2      null         -77%          n/a
fes_js             0.74      0.73      0.41          -1%          -45%
pop_tv             0.53      0.52      0.28          -2%          -47%
rmsf_corr          0.27      0.31      0.27         +15%            0%
dist_js           0.014     0.014     0.015           0%           +7%
```

- [ ] **Step 4: Run the full test suite to verify nothing broke**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS (integration tests SKIP if data absent).

- [ ] **Step 5: Commit results**

```bash
git add validation_modeA.json validation_modeB.json
git commit -m "data: Phase 2 Mode A + Mode B validation results vs baseline"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `noether_project` per-chain with `pinv` | Task 1 |
| `rollout(noether=False)` flag | Task 2 |
| `angle_energy` θ₀=2.094, k=10 | Task 3 |
| MJ matrix from M&J 1996, CANONICAL ordering | Task 4 |
| `mj_contact_energy` cutoff=8, seq_sep>3, UNK excluded | Task 4 |
| `total_cg_energy` combining WCA+angle+MJ | Task 4 |
| `reweight_boltzmann` log_w, n_eff, degenerate flag | Task 5 |
| `resample_trajectory` via `torch.multinomial` | Task 5 |
| `mh_rollout` library function | Task 5 |
| `--noether` CLI flag | Task 6 |
| `--reweight`, `--kT_reweight`, `--w_angle`, `--w_mj`, `--w_wca_cg` | Task 7 |
| Null kinetics in Mode B report | Task 7 |
| `summarize()` handles None | Task 7 |
| `compare_modes.py` delta table | Task 8 |
| Phase 2 comparison run, committed | Task 9 |
| `"reweight": null` on Mode A proteins | Task 6 |
| `"reweight": {"n_eff":..., "degenerate":...}` on Mode B | Task 7 |

**Placeholder scan:** None found.

**Type consistency:**
- `noether_project(t_old, t_new, chain_id)` defined in Task 1, consumed by Task 2 ✓
- `total_cg_energy(t, res_type, chain_id, *, w_wca, w_angle, w_mj, ...)` defined in Task 4, consumed by Tasks 5 and 7 ✓
- `reweight_boltzmann` returns `{"weights", "n_eff", "degenerate"}` — used with those exact keys in Task 7 ✓
- `resample_trajectory(traj, weights, n_samples=500)` defined in Task 5, called with same signature in Task 7 ✓
- `rollout(..., noether=False)` modified in Task 2, called with `noether=settings.get("noether", False)` in Tasks 6+7 ✓
