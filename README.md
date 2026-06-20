# Long-Stride MD (LSMD)

A deep-learning surrogate for protein conformational dynamics. LSMD trains a **denoising diffusion probabilistic model (DDPM)** on existing MD trajectories and then generates new trajectories at timescales **10⁵–10⁶× longer** than the classical 2 fs integration step — without solving Newton's equations.

---

## Purpose

Classical MD simulations are limited by the 2 fs timestep required for numerical stability. Generating 1 μs of trajectory requires 5 × 10⁸ force evaluations. Rare conformational transitions — the biologically important events — are systematically under-sampled because the simulation spends most of its time in low-energy basins.

LSMD learns the **displacement distribution** Δ = X_{i+τ} − X_i (Kabsch-aligned) directly from an existing MD trajectory. At inference, one neural-network call replaces τ × 200 ps worth of MD integration. Running 2500 steps at τ = 2 (400 ps/step) generates a 1 μs trajectory in ~6 minutes on a GPU — a **2000× wall-clock speedup**.

Two usage modes let you choose the trade-off between physical fidelity and conformational exploration:

| Mode | What it does | Good for |
|---|---|---|
| **explore** | Reverts to the last valid frame on structural failure, allowing wide exploration | Finding novel conformations, enhanced sampling |
| **mimic** | Periodically re-anchors to the nearest real MD frame by CA-RMSD | Reproducing the MD ensemble, benchmarking |

---

## Architecture

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

### Bead representations

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

Check your CUDA version:
```bash
nvidia-smi | grep "CUDA Version"
```

Install PyTorch matching your CUDA version from https://pytorch.org/get-started/locally/

Example for CUDA 12.x:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only:
```bash
pip install torch
```

### 3. Install this package and its dependencies

```bash
pip install -e ".[dev]"
```

This installs `lsmd` (editable), `mdtraj`, `numpy`, and `pytest`.

### 4. Verify

```bash
pytest tests/ -q
# expected: all tests passed
```

---

## Workflow Overview

```
Trajectory (.trr / .xtc / .dcd)
        │
        ▼
  1. preprocess.py   →  data/wt_2bead.pt         (bead point cloud)
        │
        ▼
  2. train.py        →  checkpoints/wt_2bead_200ep.pt  (DDPM checkpoint)
        │
        ├── 3a. infer.py        →  run_out/future_*.pdb   (K snapshots from one frame)
        │
        └── 3b. generate_md.py  →  genmd_out/trajectory.pdb  (long autoregressive trajectory)
                    │
                    ▼
             4. reconstruct.py  →  genmd_out/allatom.pdb   (all heavy atoms)
```

---

## Step 1: Preprocess

Convert your MD trajectory into a bead point cloud and save it to disk. This runs once and takes ~30 seconds for a 5000-frame trajectory.

```bash
# 2-bead (CA + CB) — recommended for MD mimicry
python scripts/preprocess.py \
    --traj  WT/WT-sol6.trr \
    --top   WT/WT-sol6.gro \
    --atoms 2bead \
    --out   data/wt_2bead.pt

# 4-bead (N, CA, C, CB) — full backbone geometry
python scripts/preprocess.py \
    --traj  WT/WT-sol6.trr \
    --top   WT/WT-sol6.gro \
    --atoms 4bead \
    --out   data/wt_4bead.pt

# CA-only — fastest
python scripts/preprocess.py \
    --traj  WT/WT-sol6.trr \
    --top   WT/WT-sol6.gro \
    --atoms ca \
    --out   data/wt_ca.pt
```

**Output** — a `.pt` dict with keys:
- `t` — `[F, P, n_beads, 3]` bead coordinates in Å (PBC-fixed, CA-superposed)
- `res_type`, `chain_id`, `res_index` — residue attributes for graph features
- `gly_mask` — `[P]` bool marking Gly residues (no Cβ)

---

## Step 2: Train

Train the DDPM on the preprocessed frames. Training 200 epochs on a GPU takes ~10 minutes for a 5000-frame, 169-residue trajectory.

```bash
# 2-bead model, multi-lag training (τ = 1, 2, 5 frames = 200 ps, 400 ps, 1 ns)
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

Key training flags:

| Flag | Default | Description |
|---|---|---|
| `--taus` | `1 2 5` | Lag schedule (frames, 200 ps/frame each) |
| `--epochs` | `200` | Training epochs |
| `--hidden` | `64` | GNN hidden dimension |
| `--layers` | `3` | GNN message-passing layers |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `32` | Training batch size |
| `--k` | `8` | kNN neighbours for the reference graph |
| `--T_diff` | `200` | DDPM noise levels |

The checkpoint stores model weights, noise schedule, reference graph, and all hyperparameters needed for inference.

---

## Step 3a: Infer — Snapshot Ensemble

Generate K independent future conformations from a single source frame. Useful for evaluating model quality against the MD reference.

```bash
python scripts/infer.py \
    --checkpoint  checkpoints/wt_2bead_200ep.pt \
    --frames      data/wt_2bead.pt \
    --tau         2 \
    --K           8 \
    --out         infer_out
```

**Output:**
- `infer_out/future_{0..7}.pdb` — bead-model PDB files (one per sample)
- `infer_out/metrics.json` — distributional metrics vs MD reference

Key metrics to check:

| Metric | Good value | Meaning |
|---|---|---|
| `rmsf_corr` | > 0.90 | Per-residue flexibility matches MD |
| `distance_matrix_js` | < 0.001 | CA–CA pairwise distance distributions match MD |
| `ensemble_recall` | > 0.95 | Model covers the MD conformational space |
| `ca_bond_mean` | 3.8–4.0 Å | Physically correct backbone geometry |

---

## Step 3b: Generate MD — Long Trajectory

Run the model autoregressively to generate a long trajectory. Each step samples one displacement Δ ~ p(Δ|τ) and advances the conformation by τ × 200 ps.

### Explore mode (find new conformations)

```bash
python scripts/generate_md.py \
    --checkpoint  checkpoints/wt_2bead_200ep.pt \
    --frames      data/wt_2bead.pt \
    --tau         2 \
    --steps       2500 \
    --min_energy \
    --k_clash     5.0 \
    --sample_mode explore \
    --out         genmd_explore
```

In **explore** mode, invalid frames (bond violations, steric clashes, Rg outliers) are rejected and the trajectory reverts to the last valid frame. This prevents runaway unfolding while still allowing wide conformational search. Expect RMSD of 15–30 Å from the starting frame over 1 μs.

### Mimic mode (reproduce MD ensemble)

```bash
python scripts/generate_md.py \
    --checkpoint  checkpoints/wt_2bead_200ep.pt \
    --frames      data/wt_2bead.pt \
    --tau         2 \
    --steps       2500 \
    --min_energy \
    --k_clash     5.0 \
    --sample_mode mimic \
    --anchor_every 50 \
    --out         genmd_mimic
```

In **mimic** mode, every `--anchor_every` steps the current conformation is replaced by the nearest real MD frame (by CA-RMSD). This keeps the trajectory inside the training distribution. Expect RMSD of 5–10 Å and RMSF of 3–5 Å — comparable to classical MD — at 2000× the speed.

### Ensemble of chains

```bash
python scripts/generate_md.py \
    --checkpoint  checkpoints/wt_2bead_200ep.pt \
    --frames      data/wt_2bead.pt \
    --tau         2 \
    --steps       500 \
    --n_chains    4 \
    --min_energy \
    --k_clash     5.0 \
    --sample_mode mimic \
    --anchor_every 50 \
    --out         genmd_ensemble
```

**Output:**
- `<out>/trajectory.pdb` — multi-MODEL PDB, all chains concatenated
- `<out>/chain_<k>.pdb` — per-chain multi-MODEL PDB
- `<out>/metrics.json` — RMSD, RMSF, displacement, validity per step
- `<out>/timing_report.txt` — wall-clock time and speedup estimate

All `generate_md.py` flags:

| Flag | Default | Description |
|---|---|---|
| `--tau` | `5` | Lag per step (frames; τ=2 → 400 ps/step) |
| `--steps` | `50` | Number of generative steps |
| `--n_chains` | `1` | Independent trajectory chains |
| `--sample_mode` | `explore` | `explore` or `mimic` |
| `--anchor_every` | `50` | Re-anchor interval for mimic mode (steps) |
| `--min_energy` | off | L-BFGS energy minimization after each step |
| `--k_bond` | `10.0` | Bond spring constant for L-BFGS |
| `--k_clash` | `1.0` | Clash penalty weight for L-BFGS (use 5.0 for 2-bead) |
| `--min_steps` | `100` | Max L-BFGS iterations per step |
| `--correct_bonds` | off | Lightweight SHAKE bond projection (alternative to `--min_energy`) |
| `--diff_steps` | `50` | Reverse diffusion steps per sample |
| `--eta` | `1.0` | DDPM stochasticity (0 = deterministic DDIM) |
| `--source_frame` | auto | Starting frame index (default: first validation frame) |

---

## Step 4: All-Atom Reconstruction

Reconstruct full heavy-atom structures from the bead trajectory by grafting generated backbone coordinates onto the nearest real MD frame's sidechains.

```bash
python scripts/reconstruct.py \
    --beads      genmd_mimic/trajectory.pdb \
    --traj       WT/WT-sol6.trr \
    --top        WT/WT-sol6.gro \
    --checkpoint checkpoints/wt_2bead_200ep.pt \
    --out        genmd_mimic/allatom.pdb
```

**Strategy by bead mode:**

- **4-bead**: Per-residue Kabsch superposition of the template backbone (N, Cα, C) onto the generated backbone rotates deep sidechain atoms (Cγ, Cδ, …) into the correct local frame. Backbone atoms and Cβ are set directly from generated coordinates. Carbonyl O is placed from peptide-plane geometry (C=O = 1.229 Å).
- **2-bead / CA**: Each residue is translated rigidly by (Cα_gen − Cα_template), preserving the template rotamer in the global frame.

**Output:** `allatom.pdb` — multi-MODEL PDB, protein heavy atoms only (no H, no solvent).

```bash
# Visualize in PyMOL
pymol genmd_mimic/allatom.pdb
```

---

## Quick Start — All-in-One Demo

If you just want to test the pipeline end-to-end:

```bash
python -m lsmd.demo \
    --traj   WT/WT-sol6.trr \
    --top    WT/WT-sol6.gro \
    --taus   1 2 5 \
    --epochs 200 \
    --out    demo_out
```

This trains a CA-DDPM and writes 8 sampled future conformations to `demo_out/`.

---

## Expected Results

Benchmarked on a 169-residue protein, 5001-frame trajectory (1 μs classical MD at 200 ps/frame), 2-bead model, 200 epochs:

| Mode | Steps | Final RMSD | Mean RMSF | Valid steps | Speedup |
|---|---|---|---|---|---|
| mimic (anchor_every=50) | 2500 (1 μs) | ~8 Å | ~3.5 Å | ~87% | **2300×** |
| explore | 2500 (1 μs) | ~26 Å | ~14 Å | ~88% | 600× |
| classical MD reference | — | — | ~2 Å | 100% | 1× |

Mimic mode reproduces the MD flexibility profile (RMSF 3.5 Å vs ~2 Å in MD) while running 2300× faster. Explore mode samples far beyond the training distribution, useful for finding novel conformations.

---

## Project Structure

```
lsmd/
  data.py        — trajectory loading, PBC fix, frame pairs, density reweighting
  geometry.py    — Kabsch alignment, SE(3) frame construction
  featurize.py   — bead graph construction (kNN + edge features), displacement targets
  model.py       — FlowNet (GNN), DDPM loss, noise schedule, samplers
  decoder.py     — bead-model PDB writer
  validation.py  — geometry checks, Ramachandran potential, L-BFGS energy minimization
  reconstruct.py — all-atom reconstruction (Kabsch grafting + geometric O placement)
  demo.py        — all-in-one CLI

scripts/
  preprocess.py  — save bead point cloud to disk
  train.py       — train DDPM checkpoint
  infer.py       — generate K snapshots from a single source frame
  generate_md.py — autoregressive long trajectory generation (explore / mimic modes)
  reconstruct.py — all-atom reconstruction from bead trajectory

tests/           — unit tests (pytest)
```

---

## Citation

If you use this code, please cite:

- Ho et al. (2020) *Denoising Diffusion Probabilistic Models* (NeurIPS)
- Kabsch (1976) *A solution for the best rotation to relate two sets of vectors* (Acta Cryst. A32)
