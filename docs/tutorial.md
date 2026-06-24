# LSMD Tutorial

This tutorial covers both the original per-protein DDPM and the transferable cross-protein propagator.

- **Part 1** — per-protein DDPM: train on a single MD trajectory, generate trajectories for that protein.
- **Part 2** — transferable propagator: train once on ATLAS + mdCATH, then zero-shot rollout on any new protein from its sequence alone.

**Pre-generated demo output** is in `demo_2bead/` and `demo_4bead/`. Visualize immediately after cloning without running any steps yourself.

---

## Prerequisites

### 1. Create a virtual environment

```bash
python -m venv lsmd-env
source lsmd-env/bin/activate        # Linux / macOS
# lsmd-env\Scripts\activate         # Windows
```

### 2. Install PyTorch with CUDA

```bash
nvidia-smi | grep "CUDA Version"
# Install from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.x
pip install torch   # CPU-only fallback
```

### 3. Install LSMD

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

# Part 1 — Per-Protein DDPM

The per-protein model trains on a single MD trajectory and learns to generate new conformations for that specific protein. No cross-protein generalization.

## Demo data

Two ready-to-visualize demos are bundled with the repository:

```
demo_2bead/
  trajectory.pdb  — 51-frame 2-bead (Cα+Cβ) trajectory, 20 ns
  allatom.pdb     — full heavy-atom reconstruction, 51 frames
  metrics.json    — RMSD, RMSF, validity
  timing_report.txt

demo_4bead/
  trajectory.pdb  — 51-frame 4-bead (N/Cα/C/Cβ) trajectory, 20 ns
  allatom.pdb     — full heavy-atom reconstruction, 51 frames
  metrics.json
  timing_report.txt
```

```bash
pymol demo_2bead/allatom.pdb
pymol demo_4bead/allatom.pdb
```

Both demos use mimic mode (τ=2, anchor_every=50) on a 169-residue protein.

## Input data

```
WT/WT-sol6.trr   — 5001-frame GROMACS trajectory (1 μs, 200 ps/frame)
WT/WT-sol6.gro   — GROMACS topology (169 protein residues + solvent)
```

> The `WT/` directory is not distributed. You need the original trajectory files to reproduce the demos.

---

## Step 1 — Preprocess

Convert the MD trajectory to bead point clouds. Runs once, ~30 seconds.

```bash
# 2-bead (Cα + Cβ): recommended for most use cases
python scripts/preprocess.py \
    --traj  WT/WT-sol6.trr \
    --top   WT/WT-sol6.gro \
    --atoms 2bead \
    --out   data/wt_2bead.pt

# 4-bead (N, Cα, C, Cβ): full backbone
python scripts/preprocess.py \
    --traj  WT/WT-sol6.trr \
    --top   WT/WT-sol6.gro \
    --atoms 4bead \
    --out   data/wt_4bead.pt
```

Expected output:
```
Frames: 5001   Residues: 169   Gly (no CB): 11
CA coordinate range  min=-30.23 Å  max=40.12 Å
Saved → data/wt_2bead.pt  (18.3 MB)
```

The `.pt` file contains `t [F, P, n_beads, 3]`, `res_type`, `chain_id`, `res_index`, `gly_mask`.

---

## Step 2 — Train

```bash
# 2-bead model
python scripts/train.py \
    --frames   data/wt_2bead.pt \
    --taus     1 2 5 \
    --epochs   200 \
    --out      checkpoints/wt_2bead_200ep.pt

# 4-bead model
python scripts/train.py \
    --frames   data/wt_4bead.pt \
    --taus     1 2 5 \
    --epochs   200 \
    --out      checkpoints/wt_4bead_200ep.pt
```

`--taus 1 2 5` trains simultaneously on 200 ps, 400 ps, and 1 ns lag times.

| Flag | Default | Description |
|---|---|---|
| `--taus` | `1 2 5` | Lag schedule (frames) |
| `--epochs` | `200` | Training epochs |
| `--hidden` | `64` | GNN hidden dimension |
| `--layers` | `3` | Message-passing layers |
| `--lr` | `1e-3` | Learning rate |
| `--T_diff` | `200` | DDPM noise levels |

---

## Step 3a — Quick validation

```bash
python scripts/infer.py \
    --checkpoint  checkpoints/wt_2bead_200ep.pt \
    --frames      data/wt_2bead.pt \
    --tau         2 \
    --K           8 \
    --out         infer_out
```

Target values in `infer_out/metrics.json`:

| Metric | Target | Meaning |
|---|---|---|
| `rmsf_corr` | > 0.90 | Per-residue flexibility matches MD |
| `distance_matrix_js` | < 0.001 | Cα–Cα distances match MD |
| `ensemble_recall` | > 0.95 | Model covers MD conformational space |
| `ca_bond_mean` | 3.8–4.0 Å | Correct backbone geometry |

---

## Step 3b — Long trajectory generation

### Demo 1 — 2-bead mimic, 20 ns

```bash
python scripts/generate_md.py \
    --checkpoint  checkpoints/wt_2bead_200ep.pt \
    --frames      data/wt_2bead.pt \
    --tau         2 \
    --steps       50 \
    --min_energy \
    --k_clash     5.0 \
    --sample_mode mimic \
    --anchor_every 50 \
    --out         demo_2bead
```

Results: RMSD 7.6 Å, RMSF 2.4 Å, 88% valid steps, **1439× speedup**.

### Demo 2 — 4-bead mimic, 20 ns

```bash
python scripts/generate_md.py \
    --checkpoint  checkpoints/wt_4bead_200ep.pt \
    --frames      data/wt_4bead.pt \
    --tau         2 \
    --steps       50 \
    --min_energy \
    --k_clash     5.0 \
    --sample_mode mimic \
    --anchor_every 50 \
    --out         demo_4bead
```

Results: RMSD 8.5 Å, RMSF 2.5 Å, **1116× speedup**. Bond/clash checks pass cleanly; only the Ramachandran check fails (~70% φ/ψ outliers — an inherent limitation of the Cartesian 4-bead model, not a structural failure).

### Production runs

| Run | Command additions | Expected outcome |
|---|---|---|
| 1 μs mimic | `--steps 2500 --sample_mode mimic` | ~8 Å RMSD, ~87% valid, ~580× speedup |
| 1 μs explore ×4 chains | `--steps 1000 --n_chains 4 --tau 5 --sample_mode explore` | ~23 Å RMSD, ~86% valid, ~1640× speedup |

**All `generate_md.py` flags:**

| Flag | Default | Description |
|---|---|---|
| `--tau` | `5` | Lag per step (frames) |
| `--steps` | `50` | Number of generative steps |
| `--n_chains` | `1` | Independent trajectory chains |
| `--sample_mode` | `explore` | `explore` or `mimic` |
| `--anchor_every` | `50` | Re-anchor interval for mimic mode |
| `--min_energy` | off | L-BFGS energy minimization after each step |
| `--k_bond` | `10.0` | Bond spring constant |
| `--k_clash` | `1.0` | Clash penalty (use 5.0 for 2-bead) |
| `--diff_steps` | `50` | Reverse diffusion steps per sample |
| `--eta` | `1.0` | Stochasticity: 1.0 = DDPM, 0.0 = deterministic DDIM |

---

## Step 4 — All-atom reconstruction

```bash
# 2-bead → high-quality backbone dihedrals via rigid Cα shift
python scripts/reconstruct.py \
    --beads      demo_2bead/trajectory.pdb \
    --traj       WT/WT-sol6.trr \
    --top        WT/WT-sol6.gro \
    --checkpoint checkpoints/wt_2bead_200ep.pt \
    --out        demo_2bead/allatom.pdb

# 4-bead → per-residue Kabsch sidechain grafting
python scripts/reconstruct.py \
    --beads      demo_4bead/trajectory.pdb \
    --traj       WT/WT-sol6.trr \
    --top        WT/WT-sol6.gro \
    --checkpoint checkpoints/wt_4bead_200ep.pt \
    --out        demo_4bead/allatom.pdb
```

---

---

# Part 2 — Transferable Cross-Protein Propagator

The transferable model learns SE(3)-equivariant backbone dynamics from thousands of proteins at once. At inference, only the protein's sequence and a starting structure are needed — no per-protein MD trajectory.

## Architecture overview

Instead of Cartesian bead displacements, the transferable model works in **SE(3) local residue frames**:

1. Each residue has a local frame (R_i ∈ SO(3), t_i ∈ ℝ³) built from its backbone N/Cα/C atoms.
2. The model predicts the **relative SE(3) update** u = (ω, Δt) that maps frame_i to frame_{i+τ} in the local coordinate system of residue i.
3. Updates in local frames are invariant to global rigid-body motion, making them portable across all proteins.

The **PropagatorNet** is a union-graph GNN that processes multiple proteins simultaneously:
- Nodes carry residue type, chain membership, sequential index, current noisy update, diffusion timestep, lag τ, and simulation temperature T.
- Edges carry 13-dimensional SE(3)-relative geometric features (inter-frame distances, relative rotations).
- DDPM ε-prediction with T=200 noise levels; DDIM sampling with η=0 for fast inference.

## Checkpoint hierarchy

The recommended workflow starts from a pre-trained checkpoint and proceeds through two fine-tuning stages:

```
v2_256h_90k.pt          ← pre-trained on large protein library (hidden=256, 6 layers, 90k steps)
    └── v4_longlags.pt  ← Phase 1: universal ATLAS fine-tune, wide lag range (20k steps)
            └── v4_{protein}.pt  ← Phase 2: per-protein fine-tune (5k steps each)
```

This hierarchy is managed by `scripts/run_v4_pipeline.sh` (see [V4 pipeline](#v4-per-protein-fine-tuning-pipeline) below).

## Training datasets

| Dataset | Proteins | Frames | Temperature | Lag |
|---|---|---|---|---|
| ATLAS | 1938 | 1.9 M | 300 K (physiological) | 100 ps/frame |
| mdCATH | 1000 | 11 M | 320, 348, 379, 413, 450 K | 1–10 ns |

---

## Step 1 — Download and preprocess data

ATLAS and mdCATH shards are pre-built `.pt` files (one per protein). Download them with:

```bash
python scripts/download_atlas_full.py --out data/atlas
python scripts/download_mdcath.py     --out data/mdcath
```

Each shard contains:
- `R_aa` `[F, N, 3]` — SO(3) log-map of per-residue rotation (float16)
- `t` `[F, N, 3]` — Cα positions in Å (float16)
- `res_type`, `chain_id`, `res_index` — residue metadata
- `dt` — picoseconds per frame
- `traj_breaks` — frame indices where sub-trajectories begin (mdCATH only)
- `traj_temps` — temperature (K) per sub-trajectory (mdCATH only)

ATLAS shards are assumed to be at 300 K. mdCATH shards carry 5 temperatures (320–450 K) as separate sub-trajectories within each protein shard.

---

## Step 2 — Train the transferable model

### Recommended training commands

**Phase 1 — Universal fine-tune from a pre-trained checkpoint (ATLAS only, 20k steps):**

```bash
python scripts/train_transfer.py \
    --shards_dir data/atlas \
    --resume checkpoints/v2_256h_90k.pt \
    --lags_ps 100 200 500 1000 2000 5000 10000 20000 50000 \
    --hidden 256 --layers 6 \
    --lam 0.0 \
    --steps 20000 \
    --out checkpoints/v4_longlags.pt
```

**Phase 2 — Per-protein fine-tune from the universal checkpoint (5k steps each):**

```bash
python scripts/train_transfer.py \
    --shard data/atlas/3u7t_A.pt \
    --resume checkpoints/v4_longlags.pt \
    --lags_ps 100 200 500 1000 2000 5000 10000 20000 50000 \
    --hidden 256 --layers 6 \
    --lam 0.0 \
    --steps 5000 \
    --out checkpoints/v4_3u7t_A.pt
```

Repeat Phase 2 for each protein. The `scripts/run_v4_pipeline.sh` script automates both phases across all proteins with three-temperature validation.

**Pre-training from scratch (optional, for new architectures):**

```bash
python scripts/train_transfer.py \
    --shards_dir data/atlas data/mdcath \
    --lags_ps 2000 5000 10000 \
    --hidden 256 --layers 6 \
    --steps 50000 \
    --temp_schedule 0:320 5000:348 10000:379 17000:413 25000:450 \
    --time_reversal \
    --compile \
    --out checkpoints/v2_256h_curriculum.pt
```

This takes ~4–5 hours on a single GPU at ~10 step/s. Key flags explained:

### Dataset and sampling

| Flag | Default | Description |
|---|---|---|
| `--shards_dir` | — | One or more directories of `*.pt` shards (e.g. `data/atlas data/mdcath`) |
| `--shard PATH` | — | Individual shard file(s), repeatable. Use instead of `--shards_dir` for per-protein fine-tuning |
| `--lags_ps` | `200 1000` | Physical lag times in picoseconds. Recommended: `100 200 500 1000 2000 5000 10000 20000 50000` for ATLAS fine-tuning |
| `--resume CKPT` | — | Resume from an existing checkpoint. Weights and optimizer state are loaded; `--steps` means additional steps beyond the checkpoint |
| `--norm_dir` | first dir | Directory to sample for UpdateNorm. Always re-fitted from current training data, so scale correctly reflects the active lag distribution |
| `--no_frame_weighted` | off | By default, shards are sampled proportional to frame count so every MD frame has equal probability regardless of how many shards come from that dataset |

### Model architecture

| Flag | Default | Description |
|---|---|---|
| `--hidden` | `128` | GNN hidden dimension; use 256 for production quality |
| `--layers` | `4` | Message-passing layers; use 6 for production quality |
| `--temp_emb_dim` | `8` | Temperature embedding dimension; 0 to disable. When > 0, the model is conditioned on simulation temperature so it learns that thermal fluctuation variance scales with T |

### Physics-informed training

| Flag | Default | Description |
|---|---|---|
| `--temp_schedule` | off | Temperature curriculum: `STEP:TEMP_K` pairs. Starts training on low-temperature (well-behaved) data and gradually introduces higher temperatures. Prevents NaN gradients caused by SO(3) singularities at high-T large backbone rotations |
| `--time_reversal` | off | Enable time-reversal augmentation (`reverse_prob=0.5`). Each training example is randomly flipped (x_{t+τ}→x_t instead of x_t→x_{t+τ}), doubling effective training data and enforcing microscopic reversibility |
| `--lam` | `0.0` | Physics penalty weight (C1 soft loss: bond lengths + steric clashes). Set `--lam 0.0` for ATLAS fine-tuning — the geometric penalty conflicts with SHAKE bond constraints at inference and degrades structural metrics (see validation analysis) |
| `--lam_warmup` | `500` | Steps to ramp physics penalty from 0 to `--lam` |
| `--lam_fdt` | `0.0` | FDT step-variance loss weight. Implemented but **not recommended** for ATLAS-scale datasets — insufficient trajectory density to reliably constrain kinetics via fluctuation-dissipation. Leave at 0 |
| `--phys_warmup` | `0` | Steps to ramp `--lam_fdt` from 0 |

### Training efficiency

| Flag | Default | Description |
|---|---|---|
| `--accum` | `4` | Gradient accumulation steps (effective batch = 4 × `max_union_nodes`) |
| `--max_union_nodes` | `2000` | Max nodes per union minibatch |
| `--compile` | off | `torch.compile` for ~37% GPU speedup (requires PyTorch 2.0+) |
| `--grad_clip` | `1.0` | Gradient norm clip |

### Temperature curriculum explained

mdCATH contains trajectories at 5 temperatures: 320, 348, 379, 413, and 450 K. At high temperatures (particularly 450 K), backbone rotations can be large enough that the SO(3) log-map encounters a singularity at π radians. Combined with float16 quantization, this produces NaN gradients early in training when the model has not yet learned to handle large deformations.

The temperature curriculum solves this by starting with only 320 K data and introducing hotter trajectories gradually:

```
0:320      — start: only 320 K (smallest rotations, most stable)
5000:348   — step 5000: add 348 K
10000:379  — step 10000: add 379 K
17000:413  — step 17000: add 413 K
25000:450  — step 25000: full curriculum (all temperatures)
```

ATLAS shards (no temperature metadata) are always included at all curriculum stages.

### Console output during training

```
  data/atlas: 1938 shards, 1,939,938 total frames  (2.2s)
  data/mdcath: 1000 shards, 11,041,709 total frames  (5.5s)
Total: 2938 shards from 2 dataset(s)
  UpdateNorm fitted on: data/atlas (1938 shards)
torch.compile: model compiled
  Temperature curriculum starts at 320K (schedule: [...])
step    100/50000  loss=0.2341  10.12 step/s  76150 nodes/s  elapsed=0.2m  ETA=82.4m
step    200/50000  loss=0.1987  10.34 step/s  77800 nodes/s  elapsed=0.4m  ETA=80.6m
  [step 5000] curriculum: max_temp=348K (allowed: [320, 348]K)
  [step 10000] curriculum: max_temp=379K (allowed: [320, 348, 379]K)
...
Checkpoint saved -> checkpoints/v2_256h_curriculum.pt
```

The curriculum transition lines confirm that temperature gates fire at the right steps.

---

## Step 3 — Validate physics

`validate_physics.py` is the comprehensive validation script. It rolls out the model,
then computes structural, thermodynamic, and kinetic metrics against the MD reference
trajectory. Use it after every fine-tuning run.

```bash
python scripts/validate_physics.py \
    --checkpoint checkpoints/v4_3u7t_A.pt \
    --shard      data/atlas/3u7t_A.pt \
    --steps      300 \
    --tau_ps     2000 \
    --diff_steps 20 \
    --eta        1.0 \
    --temp_K     300.0 \
    --noether \
    --out        validation_3u7t_A_T300.json
```

To sweep inference temperatures (300 / 375 / 450 K), run the command three times with `--temp_K`.
The best temperature is usually 300–375 K; 450 K tends to degrade structural metrics.

Output JSON structure:
```json
{
  "summary": {
    "mean_rmsf_corr": 0.939,
    "mean_dist_js": 0.000142,
    "mean_fes_js": 0.360,
    "mean_relax_ratio": 0.414
  },
  "proteins": {
    "3u7t_A": {
      "structural":    { "rmsf_corr": 0.939, "dist_js": 0.000142, "rg_js": 0.025,
                         "ca_bond_mean": 3.829, "clash_count": 0.0 },
      "thermodynamic": { "fes_js": 0.360, "fes_rmse_kT": 0.0, "pop_tv": 0.234 },
      "kinetic":       { "msd_rmse": 0.246, "acf_rmse": 0.194,
                         "relax_model_ps": 3125, "relax_md_ps": 7554, "relax_ratio": 0.414 },
      "n_res": 46
    }
  }
}
```

### Metric reference

| Metric | Location | Good value | Meaning |
|---|---|---|---|
| `rmsf_corr` | structural | > 0.90 | Per-residue flexibility matches MD (Pearson r) |
| `dist_js` | structural | < 0.005 | Cα pairwise-distance distributions match MD |
| `rg_js` | structural | < 0.10 | Radius-of-gyration distribution matches MD |
| `ca_bond_mean` | structural | 3.7–3.9 Å | Backbone bond geometry correct |
| `clash_count` | structural | < 0.5 | Mean clashes per frame |
| `fes_js` | thermodynamic | < 0.5 | Free-energy surface in PCA space matches MD |
| `pop_tv` | thermodynamic | < 0.3 | Metastable-state populations match MD |
| `relax_ratio` | kinetic | 0.5–2.0 | Model relaxation time vs MD (1 = ideal) |

**All `validate_physics.py` flags:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to trained `.pt` checkpoint |
| `--shard` | required | Protein shard `.pt` (repeatable for multi-protein reports) |
| `--steps` | `200` | Number of autoregressive rollout steps |
| `--tau_ps` | `2000` | Physical lag per step in picoseconds |
| `--diff_steps` | `20` | Denoising steps per rollout step |
| `--eta` | `1.0` | Stochasticity: 1.0 = DDPM, 0.0 = deterministic DDIM |
| `--temp_K` | `300.0` | Simulation temperature in Kelvin |
| `--noether` | off | Apply Noether momentum projection after each step (recommended — prevents COM drift) |
| `--wca_sigma` | `4.5` | WCA excluded-volume diameter (Å); set to 0 to disable WCA guidance |
| `--wca_lam` | `0.05` | WCA guidance step size (normalized units) |
| `--bond_constraint_iters` | `5` | SHAKE pseudo-bond iterations per step |
| `--max_update_norm` | `3.0` | Per-residue update norm clip before de-normalization |
| `--n_states` | `6` | Number of k-means states for population analysis |
| `--kT` | `1.0` | kT in kcal/mol for FES computation |
| `--out` | `validation_baseline.json` | Output path |
| `--device` | auto | `cuda` or `cpu` |

> **Note:** `eval_transfer.py` still exists and produces a simpler four-metric JSON (rmsf_corr, dist_js, ca_bond_mean, clash_count). Use it for quick checks; use `validate_physics.py` for full kinetic + thermodynamic analysis.

### DDIM vs DDPM at inference

The model is trained with T=200 DDPM noise levels but supports any number of denoising steps at inference via **DDIM** (Denoising Diffusion Implicit Models). With η=0, the reverse process is deterministic and can use far fewer steps without quality loss.

| `--diff_steps` | `--eta` | Mode | Speed relative to T=200 DDPM |
|---|---|---|---|
| 200 | 1.0 | Full DDPM (stochastic) | 1× |
| 50 | 1.0 | Subsampled DDPM | 4× |
| 20 | 0.0 | DDIM deterministic | **10×** |
| 10 | 0.0 | Aggressive DDIM | **20×** |

For production rollouts, `--diff_steps 20 --eta 0.0` provides a good quality/speed balance.

---

---

## V4 Per-Protein Fine-Tuning Pipeline

`scripts/run_v4_pipeline.sh` automates the full two-phase workflow across multiple proteins
and produces per-protein checkpoints with three-temperature validation reports.

```bash
# Full pipeline (~5–6 hours on a single GPU for 6 proteins)
bash scripts/run_v4_pipeline.sh

# Dry run — print all commands without executing
bash scripts/run_v4_pipeline.sh --dry-run
```

**What it does:**

| Phase | What | Output |
|---|---|---|
| Phase 1 train | 20k steps, all proteins, lags 100 ps–50k ps | `checkpoints/v4_longlags.pt` |
| Phase 1 validate | 300 steps, T=300 K, all proteins | `validation_v4_longlags_T300.json` |
| Phase 2 train (×N) | 5k steps per protein from longlags checkpoint | `checkpoints/v4_{protein}.pt` |
| Phase 2 validate (×N×3) | T=300, 375, 450 K per protein | `validation_v4_{protein}_T{temp}.json` |

**Prerequisites:**
- `checkpoints/v3_lam0.pt` (Phase 1 base checkpoint)
- `data/atlas/{protein}.pt` shards for each protein in `PROTEINS`

**Customising the protein list or lag set:** edit the shell variables at the top of the script:
```bash
PROTEINS="3u7t_A 4p3a_B 1b2s_F 2y4x_B 1z0b_A 6ovk_R"
BASE_TRAIN="--hidden 256 --layers 6 --lags_ps 100 200 500 1000 2000 5000 10000 20000 50000 --lam 0.0"
VFLAGS="--steps 300 --tau_ps 2000 --diff_steps 20 --eta 1.0 --noether"
```

### Why wide lags matter

The ATLAS frame interval is **100 ps**. If the minimum training lag is larger than the
inference τ, the model must extrapolate out-of-distribution — an unstable regime for DDPM
that produces NaN positions during rollout.

The recommended lag set `[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]` ps:
- Places the inference τ = 2000 ps in the **middle** of the distribution (not at an edge)
- Provides short-lag anchoring (100–500 ps) for local structural stability
- Covers slow barrier-crossing timescales up to 50% of one ATLAS trajectory

---

## Throughput and lag time selection

The simulated time generated per day depends on the lag time τ, protein size N, and number of denoising steps T_DDPM:

```
simulated_time_per_day ≈ 86400 s × (150,000 nodes/s / N) / T_DDPM × τ
```

For the transferable model (inference throughput ~150K nodes/s on a single GPU):

### With standard DDPM (T=200 steps)

| τ | N = 100 residues | N = 200 residues | N = 300 residues |
|---|---|---|---|
| 2 ns (2000 ps) | ~1300 μs/day | ~650 μs/day | ~430 μs/day |
| 5 ns (5000 ps) | ~3200 μs/day | ~1600 μs/day | ~1080 μs/day |
| 10 ns (10000 ps) | ~6500 μs/day | ~3200 μs/day | ~2160 μs/day |

### With DDIM (T=20 steps, η=0, `--diff_steps 20 --eta 0.0`)

10× speedup — same rows × 10.

**Choosing τ**: use the smallest τ the model was trained on (2000 ps) for maximum physical accuracy. Increase τ only if throughput is insufficient. The training lags `--lags_ps 2000 5000 10000` bracket common use cases.

---

## Physics-informed features

### Temperature conditioning (`--temp_emb_dim 8`, default)

The model embeds the simulation temperature T alongside the lag time τ. This is important because thermal fluctuation variance scales linearly with T (equipartition theorem): at 450 K, backbone fluctuations are ~40% larger than at 320 K. Without temperature conditioning, the model would learn an average scale and systematically under-predict fluctuations at high T.

At inference, pass `--temp_K 300.0` for physiological simulation or `--temp_K 350.0` for thermal unfolding studies. The model interpolates between the training temperatures (320–450 K) and extrapolates modestly beyond them.

Old checkpoints trained without temperature conditioning load with `temp_emb_dim=0` automatically — backward compatible.

### Time-reversal augmentation (`--time_reversal`)

With `--time_reversal`, each training example has a 50% chance of being reversed: the model sees the end frame as the source and learns to predict the transition back to the start frame (x_{t+τ} → x_t instead of x_t → x_{t+τ}). This:

1. Doubles the effective training data at zero additional cost
2. Enforces **microscopic reversibility** — the model learns dynamics that could be run forward or backward in time, consistent with equilibrium statistical mechanics
3. Naturally prevents the model from learning irreversible drift artifacts

### C1 physics loss (`--lam`)

When `--lam > 0`, an auxiliary loss penalizes the model's predicted clean-update estimate for geometric violations (Cα–Cα bond deviations from 3.8 Å; steric clashes). **Validation on ATLAS showed this hurts structural metrics**: the geometric penalty conflicts with the SHAKE bond constraint applied during inference rollout, causing bond lengths to settle at ~4.5 Å instead of 3.8 Å. Keep `--lam 0.0` for ATLAS fine-tuning. See `docs/validation_analysis.md` for the full analysis.

### C2 WCA guidance (at inference)

During the reverse diffusion process, a WCA (Weeks-Chandler-Andersen) excluded-volume
potential steers each denoising step away from steric clashes. This is distinct from the
C1 training loss — it actively nudges `u0_hat` in normalized update space at each of
the `--diff_steps` denoising steps:

```python
# Controlled via validate_physics.py / rollout() flags:
#   --wca_sigma 4.5   WCA CA–CA diameter (Å)
#   --wca_eps   0.3   well depth (kcal/mol)
#   --wca_lam   0.05  guidance step size (normalized units)
```

Set `--wca_sigma 0` to disable WCA guidance entirely. The default `--wca_lam 0.05`
produces near-zero clash counts across all proteins without measurably reducing sample diversity.

---

## Backward compatibility

Old checkpoints (before temperature conditioning and time-reversal) load correctly:

```python
from lsmd.transfer_eval import load_checkpoint
net, schedule, update_norm = load_checkpoint(
    torch.load("checkpoints/old_checkpoint.pt"), device="cpu")
# → loads with temp_emb_dim=0, PropagatorNet unchanged
```

`load_checkpoint` reads `temp_emb_dim` from the checkpoint's `hparams` and defaults to 0 when the key is absent.

---

## Using the Python API directly

```python
import torch
from lsmd import featurize as feat
from lsmd.transfer_eval import load_checkpoint, rollout, evaluate

device = "cuda"
ckpt = torch.load("checkpoints/v2_256h_curriculum.pt", map_location="cpu")
net, schedule, update_norm = load_checkpoint(ckpt, device=device)

# Load a shard
shard = torch.load("data/atlas/1abc.pt", map_location="cpu")
if "R_aa" in shard:
    from lsmd import geometry as g
    R0 = g.so3_exp(shard["R_aa"][0].float())
else:
    R0 = shard["R"][0]
t0 = shard["t"][0].float()

# Run rollout: 100 steps × 2000 ps = 200 ns of simulated time
traj = rollout(
    net, schedule, update_norm,
    R0, t0, shard["res_type"], shard["chain_id"], shard["res_index"],
    steps=100,
    tau_ps=2000.0,   # 2 ns per step
    k=12,
    diff_steps=20,   # DDIM: 10× faster than 200-step DDPM
    eta=0.0,         # deterministic
    temp_K=300.0,    # physiological temperature
    device=device,
)
# traj: [101, N, 3] Cα positions in Å

# Score against reference MD
metrics = evaluate(traj, shard["t"].float())
print(metrics)
# {'rmsf_corr': 0.87, 'dist_js': 0.003, 'ca_bond_mean': 3.81, 'clash_count': 0.0}
```

---

## Training on your own protein dataset

If you have your own MD data and want to train the transferable model:

```python
from lsmd import data, featurize as feat, geometry as g

# Build a shard from your trajectory
shard = {
    "R_aa": g.so3_log(R_all).half(),   # [F, N, 3] SO(3) log-map, float16
    "t":    t_all.half(),               # [F, N, 3] Cα positions, float16
    "res_type":  res_type,              # [N] long
    "chain_id":  chain_id,             # [N] long
    "res_index": res_index,            # [N] long
    "dt": 200.0,                       # ps per frame
    "n_res": N,
}
torch.save(shard, "data/my_protein/my_protein.pt")
```

Then train with `--shards_dir data/my_protein` alongside ATLAS/mdCATH. Set `--lags_ps` to match your trajectory's frame interval and desired lag times.

---

## Summary

| | Per-protein DDPM | Transferable propagator (v4) |
|---|---|---|
| Training data | Single MD trajectory | ATLAS fine-tune + per-protein fine-tune |
| Proteins at inference | Same protein only | Any protein (zero-shot or fine-tuned) |
| Lag range trained | 200 ps – 1 ns | 100 ps – 50 ns |
| Inference τ | Matches training | 2000 ps (recommended) |
| Throughput (N=100) | Protein-specific | ~1300 μs/day (DDPM) / ~13 ms/day (DDIM-20) |
| Temperature | Single (training T) | 300–450 K sweep; best T chosen per protein |
| Checkpoint size | ~1 MB | ~20 MB (hidden=256) |
| Training time | ~10 min (GPU) | ~3 h Phase 1 + ~48 min/protein Phase 2 |
| Validation script | `infer.py` | `validate_physics.py` (structural + kinetic + thermo) |
| Typical RMSF corr (ATLAS) | Per-protein only | 0.72–0.98 (v4 per-protein fine-tune) |
| All-atom reconstruction | Template nearest-neighbor | Not included (Cα-level output) |
