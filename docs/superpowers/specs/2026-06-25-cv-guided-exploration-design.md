# CV-Guided Conformation Explorer — Design Spec

**Date:** 2026-06-25
**Status:** Approved for implementation

---

## Goal

Leverage the trained SE(3) PropagatorNet to generate protein conformations that lie outside the training MD trajectory distribution, then validate physical plausibility via a tiered filter (structural geometry → short MD relaxation). The result is a pool of novel, physically meaningful conformations — alternative folded states, cryptic pockets, partially displaced loops — that classical short MD would not sample.

## Background and Motivation

The ATLAS training trajectories (1001 frames × 1 replica per protein, 100 ns total at 300 K) cover only the near-native equilibrium basin. The v4 per-protein fine-tune achieves FES JS ≈ 0.33–0.57, confirming that the generated ensemble does not fully cover even the training distribution. Rare conformational transitions on μs–ms timescales are entirely absent.

Standard inference (unguided rollout) re-samples the same near-native basin. To explore beyond it, we need a mechanism that actively steers the generative process away from already-visited regions of conformational space.

**Constraint**: physics-based loss functions (FDT loss, CG energy scoring) are off-limits as filters or training objectives — prior explorations showed the ATLAS dataset is too sparse to fit them reliably. All guidance must be purely geometric.

---

## Architecture Overview

```
Shard trajectory
      │
      ▼
  CVSpace.fit()          ← PCA on Cα coords + Rg + RMSD baseline
      │
      ▼  (saved as cv_basis.pt alongside checkpoint)
      │
  explore_conformations.py
      │
      ├── for each batch i = 1..N_explore:
      │       │
      │       ▼
      │   guided DDPM rollout  (transfer_eval.py + cv_guidance.py)
      │       │  each denoising step:
      │       │    1. predict x0_hat
      │       │    2. WCA gradient (existing)
      │       │    3. CV repulsion gradient (new)  ← steers away from buffer B
      │       │    4. apply combined nudge
      │       │
      │       ▼
      │   geometry filter (clash, bond)
      │       │
      │       ├── PASS → add cv_i to B, save PDB
      │       └── FAIL → discard (do not pollute buffer)
      │
      └── output: structures.pt, cv_coords.npy, candidates/*.pdb,
                  cv_coverage.png, summary.json
                      │
                      ▼
              user runs 50 ns MD on each candidate PDB
                      │
                      ▼
              classify survivors by RMSD-from-native
```

---

## Component 1: CVSpace (`lsmd/cv_guidance.py`)

**Responsibility**: compute and manage the collective variable basis; compute guidance gradients.

```python
class CVSpace:
    def __init__(self, n_pc: int = 3, device: str = "cpu"):
        # n_pc: number of PCA components (default 3)

    def fit(self, coords: Tensor) -> None:
        # coords: [F, N, 3] Cα coordinates from training shard
        # 1. Kabsch-align all frames to mean
        # 2. sklearn PCA on flattened [F, 3N] array, keep top n_pc components
        # 3. store: self.mean [N,3], self.components [n_pc, N, 3],
        #            self.explained_variance [n_pc]
        # 4. compute self.rg_mean, self.rg_std from training coords
        # 5. compute self.rmsd_std from training coords vs mean

    def project(self, x: Tensor) -> Tensor:
        # x: [N, 3] single structure
        # returns cv: [n_pc + 2] = [PC1..PCn_pc, Rg_normalized, RMSD_normalized]

    def repulsion_grad(self, x: Tensor, buffer: list[Tensor],
                       sigma: float, k_guide: float) -> Tensor:
        # x: [N, 3] current x0_hat
        # buffer: list of cv vectors from accepted structures
        # Returns gradient dV/dx where V = sum_j exp(-||cv(x)-cv_j||^2 / 2*sigma^2)
        # Gradient is exact (PCA projection is linear; Rg/RMSD are differentiable)
        # Shape: [N, 3]

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

The gradient `dV/dx` is computed in two steps:
1. `dV/dcv` — analytic from the Gaussian kernel
2. `dcv/dx` — exact Jacobian: PCA part is `W^T` (linear); Rg part is `(x - mean) / (N * Rg)`; RMSD part is `(x - ref) / (N * RMSD)`

No autograd through the model is needed — all gradients are closed-form.

---

## Component 2: Guided Rollout (modify `lsmd/transfer_eval.py`)

Add a `cv_guidance` parameter to the existing rollout function (whichever function WCA guidance currently threads through):

```python
def rollout_guided(
    model, x0, n_steps, tau_ps, temp_K,
    wca_k=0.0,           # existing
    cv_space=None,       # CVSpace instance or None
    cv_buffer=None,      # list of cv Tensors, mutated in-place by caller
    k_guide=0.05,        # CV repulsion strength
    sigma_cv=1.0,        # Gaussian width in normalized CV units
    guide_warmup=50,     # number of buffer entries before CV repulsion activates
    **kwargs
):
```

Inside the denoising loop, after computing the WCA gradient, add:

```python
if cv_space is not None and len(cv_buffer) >= guide_warmup:
    cv_grad = cv_space.repulsion_grad(x0_hat, cv_buffer, sigma_cv, k_guide)
    # add cv_grad to the existing score nudge (same mechanism as WCA)
```

The `cv_buffer` list is managed externally by `explore_conformations.py` — the rollout function only reads it. This keeps the rollout function stateless.

---

## Component 3: Exploration Script (`scripts/explore_conformations.py`)

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to v4 per-protein checkpoint |
| `--shard` | required | Path to protein shard (.pt) |
| `--n_explore` | 500 | Number of candidate structures to attempt |
| `--n_steps` | 50 | Rollout steps per structure |
| `--tau_ps` | 2000 | Lag per step (ps) |
| `--temp_K` | 375 | Inference temperature |
| `--k_guide` | 0.05 | CV repulsion strength |
| `--sigma_cv` | 1.0 | Gaussian width (normalized CV units) |
| `--guide_warmup` | 50 | Min buffer size before repulsion activates |
| `--n_pc` | 3 | PCA components in CV |
| `--diff_steps` | 20 | DDIM denoising steps per rollout step |
| `--out` | `explore_out/` | Output directory |
| `--resume` | None | Resume from existing summary.json, skip already-done |
| `--seed` | 42 | Random seed |

**Main loop:**

```python
cv_space = CVSpace(n_pc=args.n_pc)
cv_space.fit(shard["coords"])          # [F, N, 3]
cv_buffer = []
results = []

for i in range(args.n_explore):
    x_start = random_training_frame(shard)
    x_final = rollout_guided(model, x_start, ..., cv_buffer=cv_buffer)

    # geometry filter
    clashes = count_clashes(x_final)
    bond_rmsd = bond_length_rmsd(x_final)
    if clashes >= 0.5 or bond_rmsd >= 0.1:
        continue

    cv_i = cv_space.project(x_final)
    cv_buffer.append(cv_i)

    rmsd_native = kabsch_rmsd(x_final, shard["mean_coords"])
    results.append({"id": i, "cv": cv_i, "rmsd_native": rmsd_native,
                    "clashes": clashes, "bond_rmsd": bond_rmsd})
    save_pdb(x_final, f"{args.out}/candidates/{i:05d}.pdb")

    if i % 50 == 0:
        plot_cv_coverage(cv_buffer, cv_space, shard, args.out)
```

**Outputs:**

```
explore_out/
  cv_basis.pt          ← saved CVSpace (PCA basis, norms)
  structures.pt        ← [M, N, 3] Cα coords of M accepted structures
  cv_coords.npy        ← [M, n_pc+2] CV vectors
  candidates/          ← 000000.pdb … M PDB files for MD input
  cv_coverage.png      ← PC1 vs PC2 scatter: training (grey) + generated (blue→red)
  summary.json         ← per-structure metadata; add "md_pass": true/false after MD
```

The `summary.json` schema:
```json
[
  {
    "id": 0,
    "cv": [0.12, -1.3, 0.45, 1.02, 2.1],
    "rmsd_native": 3.4,
    "clashes": 0.0,
    "bond_rmsd": 0.03,
    "md_pass": null,
    "md_rmsd_final": null,
    "md_rg_final": null
  }
]
```

User populates `md_pass`, `md_rmsd_final`, `md_rg_final` after running MD. A companion script `scripts/summarize_exploration.py` reads the completed `summary.json` and produces a final report (table + figures).

---

## Physical Validity Pipeline (Stage 2 — User-run MD)

For each PDB in `candidates/`:

1. Run 50 ns unbiased MD (OpenMM or GROMACS; user provides topology/forcefield)
2. Sample Cα RMSD every 1 ns relative to the *generated* structure
3. **Pass criteria** (both must hold):
   - Mean Cα RMSD over the last 10 ns < 5 Å from the generated structure
   - No complete unfolding: final Rg < 1.5 × native Rg
4. **Classification of MD survivors:**

| RMSD from native (Å) | Classification |
|---|---|
| > 3 | Genuine alternative state (primary target) |
| 1–3 | Expanded equilibrium fluctuation |
| < 1 | Near-native (valid but less novel) |

---

## Success Criteria for the Exploration Run

A run is considered successful if it produces at least:
- 10 geometry-passing structures with PC1 or PC2 > 2σ beyond the training distribution
- ≥ 1 MD-validated structure with RMSD-from-native > 3 Å

The `cv_coverage.png` plot is the primary diagnostic: generated structures should visibly extend beyond the grey training cloud in PC1-PC2 space.

---

## Files Created / Modified

| File | Action | Description |
|---|---|---|
| `lsmd/cv_guidance.py` | Create | CVSpace class: PCA fit, project, repulsion_grad, save/load |
| `lsmd/transfer_eval.py` | Modify | Add cv_guidance, cv_buffer, k_guide, sigma_cv, guide_warmup params to rollout |
| `scripts/explore_conformations.py` | Create | Exploration loop, geometry filter, output |
| `scripts/summarize_exploration.py` | Create | Reads completed summary.json, produces report table + figures |

No changes to `train_transfer.py`, `validate_physics.py`, or any training code.

---

## Non-Goals

- No physics-based loss functions or CG energy scoring (known to be unreliable on ATLAS scale)
- No automatic MD running — the MD step remains a user-driven external step
- No retraining on validated structures in this phase (that is Approach 3, a future iteration)
- No multi-protein joint exploration in a single run
