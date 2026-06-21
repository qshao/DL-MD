# Long-Stride MD (LSMD)

A deep-learning surrogate for protein conformational dynamics. LSMD provides two complementary models:

1. **Per-protein DDPM** — trains on a single MD trajectory and generates new conformations for that specific protein at **10⁵–10⁶× speedup**.
2. **Transferable cross-protein propagator** — trains once on thousands of proteins (ATLAS + mdCATH) and generalizes **zero-shot** to any new protein from its sequence alone.

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
| UpdateNorm (ATLAS-only) | Normalizes updates via 99th-percentile abs; fitted on room-temperature data to avoid high-T mdCATH scale inflation |
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

```
ATLAS + mdCATH shards  (pre-built .pt files, one per protein)
        │
        ▼
  train_transfer.py  →  checkpoints/v2_256h_curriculum.pt
                              (hidden=256, layers=6, temp-conditioned)
        │
        ▼
  eval_transfer.py   →  eval.json
                         (RMSF correlation, distance JS, geometry validity)
```

Quick start (see [tutorial Part 2](docs/tutorial.md#part-2--transferable-cross-protein-propagator)):

```bash
# Download pre-built shards (1938 ATLAS + 1000 mdCATH proteins)
python scripts/download_atlas_full.py   --out data/atlas
python scripts/download_mdcath.py       --out data/mdcath

# Train (50 K steps, ~4–5 h on a single GPU)
python scripts/train_transfer.py \
    --shards_dir data/atlas data/mdcath \
    --lags_ps 2000 5000 10000 \
    --hidden 256 --layers 6 --steps 50000 \
    --temp_schedule 0:320 5000:348 10000:379 17000:413 25000:450 \
    --time_reversal --compile \
    --out checkpoints/v2_256h_curriculum.pt

# Evaluate zero-shot on a held-out protein
python scripts/eval_transfer.py \
    --checkpoint checkpoints/v2_256h_curriculum.pt \
    --shard data/atlas/1abc.pt \
    --tau_ps 2000 --diff_steps 20 --eta 0.0 \
    --out eval_1abc.json
```

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

## Expected Results (per-protein model)

Benchmarked on a 169-residue protein, 5001-frame trajectory (1 μs classical MD at 200 ps/frame), 200 epochs. All runs use `--min_energy --k_clash 5.0`.

| Run | Model | Mode | τ | Steps | Final RMSD | RMSF | Valid steps | Speedup |
|---|---|---|---|---|---|---|---|---|
| A | 2-bead | mimic (anchor=50) | 2 | 2500 (1 μs) | ~8 Å | ~3.5 Å | ~87% | **580×** |
| B | 2-bead | explore, 4 chains | 5 | 1000 (1 μs) | ~23 Å | ~14 Å | ~86% | 1640× |
| C | 4-bead | mimic (anchor=50) | 2 | 500 (200 ns) | ~7 Å | ~3.5 Å | 0%* | 1180× |

\* Run C validity is 0% due to poor Ramachandran geometry — a known limitation of the 4-bead Cartesian model. Bond and clash checks pass cleanly.

---

## Project Structure

```
lsmd/
  data.py            — trajectory loading, PBC fix, frame pairs, density reweighting;
                       physical_lag_pairs with trajectory-boundary and temperature filtering
  geometry.py        — Kabsch alignment, SO(3)/SE(3) operations
  featurize.py       — bead graph construction; SE(3) frame graph + relative update targets
  model.py           — FlowNet (GNN), DDPM/DDIM loss, noise schedule, unified samplers
  transfer_model.py  — PropagatorNet (union-graph DDPM with temperature conditioning);
                       StructuralEncoder + Denoiser (CachedPropagator for fast rollout);
                       sample_ddpm_union / sample_ddpm_union_cached (DDPM + DDIM via η)
  transfer_train.py  — cross-protein trainer: frame-proportional sampling, UpdateNorm,
                       temperature curriculum, time-reversal augmentation, NaN guard
  transfer_eval.py   — zero-shot rollout (DDPM/DDIM) and evaluation metrics
  normalize.py       — UpdateNorm: 99th-percentile per-component normalization
  physics_loss.py    — C1 soft loss: geometric penalty (bond/clash/Ramachandran) on x0_hat
  guidance.py        — C2 reconstruction guidance: gradient nudge toward valid geometry
  batching.py        — union_collate: disjoint-union batching for variable-size proteins
  atlas.py           — ATLAS shard builder: SE(3) frames + degenerate-frame filtering
  mdcath.py          — mdCATH shard builder: multi-temperature trajectories + traj_breaks
  splits.py          — protein-level train/val/test splitting by sequence cluster
  decoder.py         — bead-model PDB writer
  validation.py      — geometry checks, Ramachandran potential, L-BFGS energy minimization
  reconstruct.py     — all-atom reconstruction
  demo.py            — all-in-one CLI

scripts/
  preprocess.py              — save bead point cloud to disk (per-protein model)
  train.py                   — train per-protein DDPM checkpoint
  infer.py                   — generate K snapshots from a single source frame
  generate_md.py             — autoregressive long trajectory (explore / mimic modes)
  reconstruct.py             — all-atom reconstruction from bead trajectory
  download_atlas_full.py     — download and build all ATLAS shards
  download_mdcath.py         — download and build all mdCATH shards
  train_transfer.py          — train transferable cross-protein propagator
  eval_transfer.py           — zero-shot evaluation on held-out proteins
  repack_shards.py           — compact shard format conversion
  download_and_train_atlas.py — end-to-end ATLAS pipeline

tests/                       — pytest unit tests

docs/
  tutorial.md                — step-by-step tutorial (per-protein + transferable)
```

---

## Citation

If you use this code, please cite:

- Ho et al. (2020) *Denoising Diffusion Probabilistic Models* (NeurIPS)
- Song et al. (2022) *Denoising Diffusion Implicit Models* (ICLR) — for DDIM sampling
- Kabsch (1976) *A solution for the best rotation to relate two sets of vectors* (Acta Cryst. A32)
- Vander Meersche et al. (2024) *ATLAS: Protein flexibility description from atomistic molecular dynamics simulations* (Nucleic Acids Research)
