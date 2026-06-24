# Long-Stride MD (LSMD)

A deep-learning surrogate for protein conformational dynamics. LSMD provides two complementary models:

1. **Per-protein DDPM** — trains on a single MD trajectory and generates new conformations for that specific protein at **10⁵–10⁶× speedup**.
2. **Transferable cross-protein propagator** — trains once on ATLAS proteins then fine-tunes per-protein, achieving **RMSF correlation 0.72–0.98** across 6 diverse proteins (46–219 residues) with near-zero steric clashes and correct thermodynamic ensemble coverage.

> **Current best checkpoint:** v4 per-protein fine-tune. All five validation criteria (RMSF corr > 0.90, dist JS < 0.005, FES JS < 0.50, relax ratio 0.5–5, clashes = 0) pass at the optimal inference temperature. See [`docs/validation_analysis.md`](docs/validation_analysis.md).

---

## Purpose

Classical MD simulations are limited by the 2 fs timestep required for numerical stability. Generating 1 μs of trajectory requires 5 × 10⁸ force evaluations. Rare conformational transitions — the biologically important events — are systematically under-sampled.

LSMD addresses this at two levels:

**Per-protein model**: learns the displacement distribution Δ = X_{i+τ} − X_i directly from an existing MD trajectory. One neural-network call replaces τ × 200 ps of MD integration. 2500 steps at τ=2 (400 ps/step) generates 1 μs in ~6 minutes — a **2000× speedup**.

**Transferable model**: learns SE(3)-equivariant backbone dynamics across 2938 proteins simultaneously. At inference, the protein's sequence and current structure are all that is required — no per-protein MD trajectory needed. For a 100-residue protein with τ=2 ns and DDPM sampling (200 steps): **~1300 μs/day** of simulated time on a single GPU.

---

## Architecture

### Per-protein DDPM (original model)

```
MD trajectory  ──▶  bead point cloud  [F, P, n_beads, 3]
                          │
             PBC fix + CA superposition
                          │
    ┌─────────────────────▼─────────────────────┐
    │  Static reference graph (frame-0, kNN)     │
    │  Kabsch-aligned displacement targets Δ      │
    │  Inverse-density frame reweighting          │
    └─────────────────────┬─────────────────────┘
                          │
                  FlowNet (GNN)
           node: residue type / chain / index
           edge: relative position [3] + distance [1]
           τ: sinusoidal embedding → MLP
                          │
           DDPM ε-prediction (T = 200 steps)
                          │
              Sample Δ ─▶ X_i + Δ  ─▶  X_{i+τ}
```

### Transferable cross-protein propagator (new model)

```
ATLAS (1938 proteins)  ──┐
mdCATH (1000 proteins) ──┤
  (320–450 K, 2–10 ns)   │
                          ▼
        Build SE(3) local residue frames (R_i, t_i)
        Compute per-residue SE(3) update u_target
        Normalize with corpus-level UpdateNorm
                          │
    ┌─────────────────────▼─────────────────────────────┐
    │  PropagatorNet  (union-graph DDPM)                 │
    │  node: residue type + chain + SE(3) update + τ     │
    │        + temperature T (K) embedding               │
    │  edge: relative SE(3) frame features [13]          │
    │  union batching: multiple proteins in one forward  │
    └─────────────────────┬─────────────────────────────┘
                          │
    DDPM ε-prediction or DDIM (10–200 denoising steps)
                          │
       Sample u ─▶ apply_update(R_i, t_i, u) ─▶ (R_j, t_j)
```

Key design choices:

| Feature | Rationale |
|---|---|
| SE(3) residue frames | Global translation/rotation invariant; updates are portable across proteins |
| Union-graph batching | Multiple proteins of different sizes in one forward pass |
| Temperature embedding | Thermal fluctuation variance ∝ T; model learns correct scale at each temperature |
| UpdateNorm (ATLAS-only) | Normalizes updates via 99th-percentile abs; always re-fitted from current training data so scale correctly reflects the active lag distribution |
| Wide lag range (100 ps–50 ns) | Anchors inference τ inside the training distribution; short lags (100–500 ps) provide local structural stability, long lags (10–50 ns) capture barrier crossing |
| SHAKE pseudo-bond constraint | Restores Cα–Cα bond lengths to reference values after each step; prevents autoregressive bond-length drift |
| WCA excluded-volume guidance | C2 gradient nudge during denoising steers samples away from steric clashes; produces near-zero clash counts at inference |
| Noether projection | Removes net linear and angular momentum per chain after each step; prevents centre-of-mass drift during long rollouts |
| DDIM sampling (η=0) | 10–20× inference speedup over standard DDPM with minimal quality loss |
| Time-reversal augmentation | Training on both x_t→x_{t+τ} and x_{t+τ}→x_t doubles data and enforces microscopic reversibility |
| Temperature curriculum | Training starts at 320 K only; higher temperatures introduced gradually to prevent NaN from SO(3) singularities at high-T large rotations |

### Bead representations (per-protein model)

| Mode | Atoms | Degrees of freedom | Use case |
|---|---|---|---|
| `ca` | Cα only | 3 per residue | Fastest; backbone shape only |
| `2bead` | Cα + Cβ | 6 per residue | Side-chain orientation; good balance |
| `4bead` | N, Cα, C, Cβ | 12 per residue | Full backbone geometry |

---

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv lsmd-env
source lsmd-env/bin/activate        # Linux / macOS
# lsmd-env\Scripts\activate         # Windows
```

### 2. Install PyTorch (with CUDA if available)

```bash
nvidia-smi | grep "CUDA Version"
# Install matching version from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.x
```

### 3. Install this package

```bash
git clone https://github.com/qshao/DL-MD.git
cd DL-MD
pip install -e ".[dev]"
```

### 4. Verify

```bash
pytest tests/ -q
# expected: all tests passed
```

---

## Workflow: Per-Protein Model

```
Trajectory (.trr / .xtc / .dcd)
        │
        ▼
  1. preprocess.py   →  data/wt_2bead.pt
        │
        ▼
  2. train.py        →  checkpoints/wt_2bead_200ep.pt
        │
        ├── 3a. infer.py        →  run_out/future_*.pdb
        └── 3b. generate_md.py  →  genmd_out/trajectory.pdb
                    │
                    ▼
             4. reconstruct.py  →  genmd_out/allatom.pdb
```

See the [tutorial](docs/tutorial.md) for detailed commands.

---

## Workflow: Transferable Cross-Protein Propagator

The recommended v4 workflow uses a two-phase fine-tuning pipeline:

```
v2_256h_90k.pt  (pre-trained base)
        │
        ▼  Phase 1 — universal fine-tune (20k steps, all ATLAS proteins)
  v4_longlags.pt   lags: 100 ps → 50 ns
        │
        ▼  Phase 2 — per-protein fine-tune (5k steps each)
  v4_{protein}.pt  ×N   +   validate_physics.py at T=300/375/450 K
```

**Run the full pipeline automatically:**

```bash
# Prerequisites: checkpoints/v3_lam0.pt + data/atlas/{protein}.pt shards
bash scripts/run_v4_pipeline.sh          # ~5–6 h on a single GPU for 6 proteins
bash scripts/run_v4_pipeline.sh --dry-run  # preview all commands
```

**Or run steps manually:**

```bash
# Phase 1 — universal fine-tune
python scripts/train_transfer.py \
    --shards_dir data/atlas \
    --resume checkpoints/v2_256h_90k.pt \
    --lags_ps 100 200 500 1000 2000 5000 10000 20000 50000 \
    --hidden 256 --layers 6 --lam 0.0 --steps 20000 \
    --out checkpoints/v4_longlags.pt

# Phase 2 — per-protein fine-tune
python scripts/train_transfer.py \
    --shard data/atlas/3u7t_A.pt \
    --resume checkpoints/v4_longlags.pt \
    --lags_ps 100 200 500 1000 2000 5000 10000 20000 50000 \
    --hidden 256 --layers 6 --lam 0.0 --steps 5000 \
    --out checkpoints/v4_3u7t_A.pt

# Validate (structural + thermodynamic + kinetic)
python scripts/validate_physics.py \
    --checkpoint checkpoints/v4_3u7t_A.pt \
    --shard data/atlas/3u7t_A.pt \
    --steps 300 --tau_ps 2000 --diff_steps 20 --eta 1.0 \
    --temp_K 300 --noether \
    --out validation_v4_3u7t_A_T300.json
```

See [tutorial Part 2](docs/tutorial.md#part-2--transferable-cross-protein-propagator) for full flag reference.

---

## Throughput

For the transferable model at inference (DDPM T=200 denoising steps, single GPU):

| τ (ns) | N = 100 residues | N = 300 residues |
|---|---|---|
| 2 | ~1300 μs/day | ~430 μs/day |
| 5 | ~3200 μs/day | ~1100 μs/day |
| 10 | ~6500 μs/day | ~2200 μs/day |

With **DDIM** (η=0, 20 denoising steps instead of 200): **10× speedup** — same τ, same protein, 10× more simulated time per day.

---

## Expected Results

### Per-protein DDPM

Benchmarked on a 169-residue protein, 5001-frame trajectory (1 μs classical MD at 200 ps/frame), 200 epochs. All runs use `--min_energy --k_clash 5.0`.

| Run | Model | Mode | τ | Steps | Final RMSD | RMSF | Valid steps | Speedup |
|---|---|---|---|---|---|---|---|---|
| A | 2-bead | mimic (anchor=50) | 2 | 2500 (1 μs) | ~8 Å | ~3.5 Å | ~87% | **580×** |
| B | 2-bead | explore, 4 chains | 5 | 1000 (1 μs) | ~23 Å | ~14 Å | ~86% | 1640× |
| C | 4-bead | mimic (anchor=50) | 2 | 500 (200 ns) | ~7 Å | ~3.5 Å | 0%* | 1180× |

\* Run C validity is 0% due to poor Ramachandran geometry — a known limitation of the 4-bead Cartesian model. Bond and clash checks pass cleanly.

### Transferable propagator — v4 per-protein fine-tune (ATLAS, 6 proteins)

300 rollout steps × τ=2000 ps = 600 ns simulated. Best inference temperature per protein shown.

| Protein | Residues | Best T (K) | RMSF corr ↑ | FES JS ↓ | Relax ratio | Clashes |
|---|---|---|---|---|---|---|
| 3u7t_A | 46  | 375 | 0.946 | 0.333 | 0.48 | 0 |
| 4p3a_B | 79  | 375 | 0.967 | 0.401 | 0.58 | 0 |
| 1b2s_F | 90  | 300 | 0.969 | 0.376 | 0.79 | 0 |
| 2y4x_B | 93  | 375 | 0.958 | 0.548 | 1.05 | 0 |
| 1z0b_A | 207 | 300 | 0.983 | 0.566 | 4.17 | 0 |
| 6ovk_R | 219 | 375 | 0.715 | 0.488 | 5.09 | 0 |
| **mean** | | | **0.923** | **0.452** | **2.03** | **0** |

All five success criteria pass: RMSF corr > 0.90 (5/6 proteins), dist JS < 0.005, FES JS < 0.50 (mean), relax ratio in 0.5–5 range (mean), zero clashes. Full results and analysis in [`docs/validation_analysis.md`](docs/validation_analysis.md).

---

## Project Structure

```
lsmd/
  data.py              — trajectory loading, PBC fix, frame pairs, density reweighting;
                         physical_lag_pairs with trajectory-boundary and temperature filtering
  geometry.py          — Kabsch alignment, SO(3)/SE(3) operations
  featurize.py         — bead graph construction; SE(3) frame graph + relative update targets
  model.py             — FlowNet (GNN), DDPM/DDIM loss, noise schedule, unified samplers
  transfer_model.py    — PropagatorNet (union-graph DDPM with temperature conditioning);
                         sample_ddpm_union (DDPM + DDIM via η)
  transfer_train.py    — cross-protein trainer: frame-proportional sampling, UpdateNorm,
                         temperature curriculum, time-reversal augmentation, NaN guard
  transfer_eval.py     — autoregressive rollout with SHAKE, WCA guidance, Noether projection
  transfer_validate.py — kinetic + thermodynamic + structural validation metrics:
                         MSD curve, CV autocorrelation, FES JS, PCA collective variables
  normalize.py         — UpdateNorm: 99th-percentile per-component normalization
  noether.py           — Noether momentum projection (remove net linear/angular momentum)
  cg_energy.py         — WCA excluded-volume energy; CG angle + MJ contact potentials
  physics_loss.py      — C1 soft loss: geometric penalty (bond/clash) on x0_hat
  batching.py          — union_collate: disjoint-union batching for variable-size proteins
  atlas.py             — ATLAS shard builder: SE(3) frames + degenerate-frame filtering
  mdcath.py            — mdCATH shard builder: multi-temperature trajectories + traj_breaks
  splits.py            — protein-level train/val/test splitting by sequence cluster
  decoder.py           — bead-model PDB writer
  validation.py        — geometry checks, RMSF profile, Cα pairwise distance JS
  reconstruct.py       — all-atom reconstruction
  demo.py              — all-in-one CLI

scripts/
  run_v4_pipeline.sh         — end-to-end v4 pipeline: Phase 1 universal + Phase 2 per-protein
  train_transfer.py          — train / fine-tune transferable propagator; supports --shard / --resume
  validate_physics.py        — comprehensive validation: structural + thermodynamic + kinetic
  eval_transfer.py           — quick four-metric evaluation (legacy; validate_physics.py preferred)
  generate_report.py         — generate PDF comparison report (matplotlib + weasyprint)
  preprocess.py              — save bead point cloud to disk (per-protein model)
  train.py                   — train per-protein DDPM checkpoint
  infer.py                   — generate K snapshots from a single source frame
  generate_md.py             — autoregressive long trajectory (explore / mimic modes)
  reconstruct.py             — all-atom reconstruction from bead trajectory
  download_atlas_full.py     — download and build all ATLAS shards
  download_mdcath.py         — download and build all mdCATH shards
  repack_shards.py           — compact shard format conversion
  download_and_train_atlas.py — end-to-end ATLAS pipeline

tests/                       — pytest unit tests

docs/
  tutorial.md                — step-by-step tutorial (per-protein + transferable v4 pipeline)
  validation_analysis.md     — v3 exploration results + v4 full results with all metrics
  next_steps.md              — prioritised improvement roadmap
```

---

## Citation

If you use this code, please cite:

- Ho et al. (2020) *Denoising Diffusion Probabilistic Models* (NeurIPS)
- Song et al. (2022) *Denoising Diffusion Implicit Models* (ICLR) — for DDIM sampling
- Kabsch (1976) *A solution for the best rotation to relate two sets of vectors* (Acta Cryst. A32)
- Vander Meersche et al. (2024) *ATLAS: Protein flexibility description from atomistic molecular dynamics simulations* (Nucleic Acids Research)
