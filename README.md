# Long-Stride MD (LSMD)

A deep-learning model that learns protein conformational dynamics directly from molecular dynamics trajectories, predicting future CA-atom conformations at timescales **10⁵× longer than the MD integration step**.

---

## Scientific Purpose

Classical MD simulations integrate Newton's equations with a 2 fs timestep. For a typical 1 μs trajectory this means 5 × 10⁸ integration steps. Rare conformational transitions — the biologically important events — are under-sampled because the simulation spends most of its time in low-energy basins.

LSMD learns a **denoising diffusion model** (DDPM) that maps a source CA conformation at frame *i* to the distribution of future conformations at frame *i+τ*, where τ is measured in trajectory frames (200 ps/frame here). At τ=5 (1 ns), the model performs a single neural-network inference that spans the same time as **500,000 MD steps**.

The model explicitly separates two physical regimes via multi-lag conditioning:

| τ (frames) | Physical time | Regime |
|---|---|---|
| 1 | 200 ps | Thermal fluctuations within a basin |
| 2 | 400 ps | Slow intra-basin relaxation |
| 5 | 1 ns | Onset of conformational transitions |

At inference the model samples the **displacement distribution** Δ = X_j − X_i (Kabsch-aligned), producing an ensemble of plausible future structures rather than a single deterministic prediction.

### Key results on the WT trajectory (5001 frames, 169 CA atoms)

| Metric | Value | Meaning |
|---|---|---|
| `rmsf_corr` | 0.948 | Per-residue flexibility profile matches MD (Pearson r) |
| `distance_matrix_js` | 0.00018 | CA–CA pairwise distance distribution identical to MD |
| `ensemble_recall` | 1.000 | Model covers 100% of MD conformational states |
| `displacement_model_mean` | 1.007 Å | Average per-atom displacement at τ=5 (MD: 0.927 Å) |
| CA bond geometry | 3.93 Å mean | Physical (ideal 3.8 Å), zero clashes |

---

## Architecture

```
Trajectory frames  ──▶  CA point cloud [F, P, 3]
                                │
                    PBC fix + CA superposition
                                │
         ┌──────────────────────▼──────────────────────┐
         │   Static reference graph (frame-0, kNN)      │
         │   Kabsch-aligned displacement targets Δ       │
         │   Inverse-density frame reweighting           │
         └──────────────────────┬──────────────────────┘
                                │
                        FlowNet (GNN)
                   node features: res type / chain / index
                   edge features: rel_pos [3] + dist [1]
                   τ embedding: sinusoidal → MLP
                                │
                    DDPM ε-prediction (T=200 steps)
                                │
                    Sample Δ ∈ ℝ^{P×3}  ──▶  X_i + Δ
                                │
                      CA-trace PDB output
```

---

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv lsmd-env
source lsmd-env/bin/activate          # Linux / macOS
# lsmd-env\Scripts\activate           # Windows
```

### 2. Install PyTorch (with CUDA if available)

Check your CUDA version first:
```bash
nvidia-smi | grep "CUDA Version"
```

Install PyTorch matching your CUDA version from https://pytorch.org/get-started/locally/

Example for CUDA 13.0:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

For CPU-only:
```bash
pip install torch
```

### 3. Install this package

```bash
pip install -e ".[dev]"      # installs lsmd + pytest
```

This also installs `mdtraj` and `numpy`.

### 4. Verify

```bash
pytest tests/ -q
# 70 passed
```

---

## Quick Start — All-in-One Demo

```bash
python -m lsmd.demo \
  --traj WT/WT-sol6.trr \
  --top  WT/WT-sol6.gro \
  --taus 1 2 5 \
  --epochs 200 \
  --out   ca_run_200ep
```

This trains a CA-DDPM for 200 epochs and writes 8 sampled future conformations to `ca_run_200ep/future_{0..7}.pdb`.

All options:

| Flag | Default | Description |
|---|---|---|
| `--traj` | required | Trajectory file (GROMACS TRR, DCD, etc.) |
| `--top` | required | Topology file (GRO, PDB, etc.) |
| `--taus` | `1 2 5` | Training lag schedule (frames) |
| `--infer_tau` | `max(taus)` | Lag for inference |
| `--out` | `demo_out` | Output directory for PDB files and report |
| `--K` | `8` | Number of samples to generate |
| `--epochs` | `50` | Training epochs |
| `--k` | `8` | kNN neighbours for graph |
| `--hidden` | `64` | GNN hidden dimension |
| `--layers` | `3` | GNN message-passing layers |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `32` | Training batch size |
| `--T_diff` | `200` | DDPM noise levels |
| `--diff_steps` | `50` | Reverse diffusion steps at inference |
| `--eta` | `1.0` | DDPM stochasticity (0 = deterministic DDIM) |
| `--device` | auto | `cuda` or `cpu` |

---

## Step-by-Step Workflow

For repeated experiments, split the pipeline into three stages so preprocessing runs only once.

### Step 1: Preprocess

```bash
python scripts/preprocess.py \
  --traj WT/WT-sol6.trr \
  --top  WT/WT-sol6.gro \
  --out  data/wt_frames.pt
```

Loads the trajectory, fixes periodic boundary conditions (`make_molecules_whole`), superimposes all frames onto frame 0 (protein CA atoms only), and saves the CA point cloud and residue attributes to `data/wt_frames.pt`.

### Step 2: Train

```bash
python scripts/train.py \
  --frames  data/wt_frames.pt \
  --taus    1 2 5 \
  --epochs  200 \
  --out     checkpoints/wt_200ep.pt
```

Trains the CA-displacement DDPM and saves a checkpoint containing the model weights, noise schedule, reference graph, and hyperparameters.

### Step 3: Infer

```bash
python scripts/infer.py \
  --checkpoint  checkpoints/wt_200ep.pt \
  --frames      data/wt_frames.pt \
  --tau         5 \
  --K           8 \
  --out         ca_run_200ep
```

Loads the checkpoint, picks a source frame from the validation split, samples K future CA conformations, writes `future_{0..K-1}.pdb`, and prints a JSON metrics report.

---

## Understanding the Output

### PDB files

Each `future_k.pdb` is a **CA-trace**: one ATOM record per residue, atom name `CA`, element `C`. The coordinates are in Ångström. Visualise with PyMOL, VMD, or ChimeraX:

```bash
pymol ca_run_200ep/future_0.pdb
```

To overlay all samples:
```
pymol ca_run_200ep/future_*.pdb
```

### Metrics report (JSON)

| Key | Description |
|---|---|
| `ca_geometry.ca_bond_mean` | Mean CA–CA bond length (Å). Ideal ≈ 3.8 Å |
| `ca_geometry.clash_count` | Number of non-bonded CA pairs < 2.0 Å |
| `pca_js` | Jensen-Shannon divergence of 2D PCA density vs MD [0–1]; lower = better |
| `ensemble_recall` | Fraction of MD frames covered by ≥1 sample within 2 Å RMSD |
| `ensemble_novelty` | Fraction of samples with no MD neighbour within 2 Å RMSD |
| `distance_matrix_js` | JS divergence of CA–CA pairwise distance distributions |
| `rmsf_corr` | Pearson r of per-residue RMSF (model vs MD); higher = better |
| `displacement_js` | JS divergence of ‖Δ‖ magnitude distributions |
| `displacement_model_mean` | Mean per-atom displacement across samples (Å) |
| `displacement_md_mean` | Mean per-atom displacement in MD reference (Å) |

Good model behaviour: `rmsf_corr > 0.9`, `distance_matrix_js < 0.001`, `ensemble_recall > 0.95`, CA bonds in [3.5, 4.2] Å.

---

## Project Structure

```
lsmd/
  data.py        — trajectory loading, PBC fix, frame pairs, density weights
  geometry.py    — Kabsch alignment, SE(3) frame construction
  featurize.py   — CA graph (kNN + edge features), CA displacement target
  model.py       — FlowNet (GNN), DDPM loss, noise schedule, samplers
  decoder.py     — CA-trace PDB writer, backbone frame decoder
  validation.py  — CA geometry, pairwise JS, RMSF, displacement distribution
  demo.py        — All-in-one CLI

scripts/
  preprocess.py  — Save processed frames to disk
  train.py       — Train and save checkpoint
  infer.py       — Load checkpoint and generate conformations

tests/           — 70 unit tests (pytest)
docs/            — Design specs and implementation plans
```

---

## Citation

If you use this code, please cite the underlying trajectory and deep-learning methods:
- Ho et al. (2020) *Denoising Diffusion Probabilistic Models* (NeurIPS)
- Kabsch (1976) *A solution for the best rotation to relate two sets of vectors* (Acta Cryst.)
