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

## Training datasets

| Dataset | Proteins | Frames | Temperature | Lag |
|---|---|---|---|---|
| ATLAS | 1938 | 1.9 M | 300 K (physiological) | 200 ps |
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

### Recommended training command

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
| `--shards_dir` | required | One or more shard directories (e.g. `data/atlas data/mdcath`) |
| `--lags_ps` | `200 1000` | Physical lag times in picoseconds (can list multiple) |
| `--norm_dir` | first dir | Directory to fit UpdateNorm on; defaults to ATLAS to avoid high-T mdCATH scale inflation |
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
| `--lam` | `0.0` | Physics penalty weight (C1 soft loss: bond lengths + steric clashes). Enabled by setting > 0 |
| `--lam_warmup` | `500` | Steps to ramp physics penalty from 0 to `--lam` |

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

## Step 3 — Evaluate zero-shot

Evaluate on a protein that was never seen during training:

```bash
python scripts/eval_transfer.py \
    --checkpoint checkpoints/v2_256h_curriculum.pt \
    --shard      data/atlas/1abc.pt \
    --steps      200 \
    --tau_ps     2000 \
    --diff_steps 20 \
    --eta        0.0 \
    --temp_K     300.0 \
    --out        eval_1abc.json
```

Output `eval_1abc.json`:
```json
{
  "model": {
    "rmsf_corr": 0.87,
    "dist_js": 0.003,
    "ca_bond_mean": 3.81,
    "clash_count": 0.0
  }
}
```

| Metric | Meaning | Good value |
|---|---|---|
| `rmsf_corr` | Pearson correlation of per-residue RMSF vs MD reference | > 0.80 for zero-shot |
| `dist_js` | Jensen-Shannon divergence of Cα–Cα pairwise distance distributions | < 0.01 |
| `ca_bond_mean` | Mean Cα–Cα bond length in generated frames | 3.7–3.9 Å |
| `clash_count` | Mean steric clashes per frame (non-bonded Cα pairs < 3.0 Å) | < 1.0 |

**All `eval_transfer.py` flags:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to trained `.pt` checkpoint |
| `--shard` | required | Held-out protein shard `.pt` |
| `--steps` | `200` | Number of autoregressive rollout steps |
| `--tau_ps` | `1000` | Lag time in picoseconds |
| `--diff_steps` | `50` | Denoising steps per rollout step (50 = DDPM quality; 10–20 = DDIM) |
| `--eta` | `1.0` | Reverse-process stochasticity: 1.0 = DDPM, 0.0 = deterministic DDIM |
| `--temp_K` | `300.0` | Simulation temperature in Kelvin passed to the model |
| `--oracle` | — | Per-protein checkpoint (upper-bound bracket) |
| `--lower` | — | Marginal-prior checkpoint (lower-bound bracket) |
| `--out` | `eval.json` | Output path |
| `--device` | auto | `cuda` or `cpu` |

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

### C1 physics loss (`--lam 0.01`)

When `--lam > 0`, an auxiliary loss penalizes the model's predicted clean-update estimate x0_hat for geometric violations:
- Cα–Cα consecutive bond deviations from 3.8 Å
- Steric clashes (non-bonded Cα pairs < 3.0 Å)

This does not change the DDPM noise schedule or sampling procedure; it provides a supervision signal that the predicted endpoint should be geometrically reasonable.

### C2 guidance (at inference, via `lsmd.guidance`)

At inference, `sample_ddpm_union_guided` applies a per-step gradient nudge toward valid geometry during the reverse diffusion process. This is distinct from the C1 training loss — it actively steers each denoising step:

```python
from lsmd.guidance import sample_ddpm_union_guided
u = sample_ddpm_union_guided(net, ..., gamma=0.1)  # gamma=0 → plain DDPM
```

`gamma > 0` reduces geometric violations at the cost of some sample diversity. Start with `gamma=0.05–0.1`.

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

| | Per-protein DDPM | Transferable propagator |
|---|---|---|
| Training data | Single MD trajectory | ATLAS (1938) + mdCATH (1000) proteins |
| Proteins at inference | Same protein only | Any new protein (zero-shot) |
| Step interval | 200 ps – 1 ns | 2 – 10 ns |
| Throughput (N=100) | Protein-specific | ~1300 μs/day (DDPM) / ~13 ms/day (DDIM-20) |
| Temperature | Single (training T) | 300–450 K (conditioning) |
| Checkpoint size | ~1 MB | ~20 MB (hidden=256) |
| Training time | ~10 min (GPU) | ~4–5 h (GPU, 50 K steps) |
| All-atom reconstruction | Template nearest-neighbor | Not included (Cα-level output) |
