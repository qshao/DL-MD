# LSMD Tutorial

This tutorial walks through the complete LSMD pipeline on a real protein system, from preprocessing an MD trajectory to generating and analysing long generative MD runs.

**Pre-generated demo output is included in the repository** under `demo_2bead/` and `demo_4bead/`. You can inspect or visualize those files immediately after cloning, without running any of the generation steps yourself.

---

## Prerequisites

### 1. Create a virtual environment

```bash
python -m venv lsmd-env
source lsmd-env/bin/activate        # Linux / macOS
# lsmd-env\Scripts\activate         # Windows
```

### 2. Install PyTorch with CUDA

Check your CUDA version:
```bash
nvidia-smi | grep "CUDA Version"
```

Install PyTorch matching your CUDA version (https://pytorch.org/get-started/locally/):
```bash
# Example for CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU-only fallback
pip install torch
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

## Demo data

After cloning, two ready-to-visualize demo runs are available immediately:

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
# Visualize immediately (no training or generation needed)
pymol demo_2bead/allatom.pdb
pymol demo_4bead/allatom.pdb
```

Both demos use mimic mode (τ=2, anchor_every=50) on a 169-residue protein. See [below](#demo-1--2-bead-mimic-20-ns) for the exact commands used to generate them.

---

## Input data

LSMD reads any trajectory format supported by MDtraj (GROMACS TRR/XTC, CHARMM DCD, AMBER NetCDF, …) plus a matching topology file. For this tutorial we use:

```
WT/WT-sol6.trr   — 5001-frame GROMACS trajectory (1 μs, 200 ps/frame)
WT/WT-sol6.gro   — GROMACS topology (169 protein residues + solvent)
```

> The `WT/` directory is not distributed with the repository. To reproduce the demo runs, you need the original trajectory files.

---

## Step 1 — Preprocess

Convert the MD trajectory to bead point clouds and save to disk. This runs once and takes ~30 seconds.

```bash
# 2-bead (Cα + Cβ): recommended for most use cases
python scripts/preprocess.py \
    --traj  WT/WT-sol6.trr \
    --top   WT/WT-sol6.gro \
    --atoms 2bead \
    --out   data/wt_2bead.pt

# 4-bead (N, Cα, C, Cβ): full backbone, slower generation
python scripts/preprocess.py \
    --traj  WT/WT-sol6.trr \
    --top   WT/WT-sol6.gro \
    --atoms 4bead \
    --out   data/wt_4bead.pt
```

Expected output:
```
Loading  WT/WT-sol6.trr
Topology WT/WT-sol6.gro
Mode     2bead
Frames: 5001   Residues: 169   Gly (no CB): 11   Residue types: 19
CA coordinate range  min=-30.23 Å  max=40.12 Å
Saved → data/wt_2bead.pt  (18.3 MB)
```

The `.pt` file contains:
- `t` — `[5001, 169, n_beads, 3]` bead coordinates in Å (PBC-fixed, Cα-superposed)
- `res_type`, `chain_id`, `res_index` — residue attributes
- `gly_mask` — `[169]` bool marking the 11 Gly residues (no Cβ)

---

## Step 2 — Train

Train the DDPM on preprocessed frames. 200 epochs on a GPU takes ~10 minutes for this system.

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

`--taus 1 2 5` trains on three lag times simultaneously (200 ps, 400 ps, 1 ns), letting the model capture dynamics from fast thermal fluctuations to slow conformational changes.

The checkpoint stores model weights, noise schedule, reference graph, and all hyperparameters. It is self-contained — no other files are needed at inference time.

---

## Step 3a — Quick validation: snapshot ensemble

Before running long trajectories, check that the model reproduces known MD statistics.

```bash
python scripts/infer.py \
    --checkpoint  checkpoints/wt_2bead_200ep.pt \
    --frames      data/wt_2bead.pt \
    --tau         2 \
    --K           8 \
    --out         infer_out
```

Check `infer_out/metrics.json` for:

| Metric | Target | Meaning |
|---|---|---|
| `rmsf_corr` | > 0.90 | Per-residue flexibility profile matches MD |
| `distance_matrix_js` | < 0.001 | Cα–Cα pairwise distance distributions match MD |
| `ensemble_recall` | > 0.95 | Model covers the MD conformational space |
| `ca_bond_mean` | 3.8–4.0 Å | Correct backbone geometry |

---

## Step 3b — Long trajectory generation

LSMD supports two sampling modes that trade conformational coverage against physical fidelity.

### Choosing a mode

| Mode | Mechanism | Typical RMSD | Best for |
|---|---|---|---|
| **mimic** | Re-anchor to nearest real MD frame every N steps | 5–10 Å | MD reproduction, benchmarking |
| **explore** | Revert to last valid frame on structural failure | 15–30 Å | Novel conformations, enhanced sampling |

---

### Demo 1 — 2-bead mimic, 20 ns

This is the command that generated `demo_2bead/`:

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

Terminal output:
```
Generative MD
  Checkpoint : checkpoints/wt_2bead_200ep.pt
  Starting frame : 3999  (5001 total, 169 residues)
  Tau per step   : 2 frames = 400 ps
  Steps          : 50
  Total time     : 20.0 ns  (10000000× faster than 2 fs MD)
  Energy min     : ON  (L-BFGS, 100 steps, k_bond=10.0, k_clash=5.0)
  Sample mode    : mimic  (re-anchor to nearest MD frame every 50 steps)
  Mode           : 2-bead (CA, CB)  point_dim=6

Chain 1/1 ...
  Done. Mean disp/step: 0.996 Å  Final RMSD: 7.618 Å  Valid steps: 44/50  re-anchors: 1

Generated 20.0 ns of CA dynamics
  Final RMSD      : 7.62 Å from start
  RMSF (mean/max) : 2.43 / 9.10 Å
  Valid steps     : 44/50 (88%)  [bond viol=0, clashes=6, Rg viol=0]
  Time/step       : 0.240 s  → 10.0 min for 1000 ns (1439× vs classical MD)
Output → demo_2bead/
```

**Results:** RMSD 7.6 Å, RMSF 2.4 Å, 88% valid steps, **1439× speedup**. The trajectory stays near the MD ensemble; 2-bead reconstruction (rigid Cα translation) preserves the template's backbone φ/ψ angles.

To scale up to a full 1 μs production run, change `--steps 50` to `--steps 2500`.

---

### Demo 2 — 4-bead mimic, 20 ns

This is the command that generated `demo_4bead/`:

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

Terminal output:
```
  Mode           : 4-bead (N, CA, C, CB)  point_dim=12

Chain 1/1 ...
  Done. Mean disp/step: 0.892 Å  Final RMSD: 8.452 Å  Valid steps: 0/50  re-anchors: 1

Generated 20.0 ns of CA dynamics
  Final RMSD      : 8.45 Å from start
  RMSF (mean/max) : 2.51 / 9.43 Å
  Valid steps     : 0/50 (0%)  [bond viol=0, clashes=0, Rg viol=0,
                                 rama viol=50 (mean 69.6% outliers/step)]
  Time/step       : 0.310 s  → 12.9 min for 1000 ns (1116× vs classical MD)
Output → demo_4bead/
```

**Results:** RMSD 8.5 Å, RMSF 2.5 Å, **1116× speedup**. Cα-level geometry is excellent (0 bonds, 0 clashes). The 0% valid steps is entirely due to Ramachandran: the 4-bead Cartesian model generates N/Cα/C displacements independently, producing ~70% φ/ψ outliers versus ~2% in real MD. This is a known limitation — not a structural failure. For dihedral analysis use the 2-bead demo instead.

**Validity check breakdown for 4-bead:**

| Check | demo_4bead result | Real MD baseline |
|---|---|---|
| Bond violations | 0 | 0 |
| Steric clashes | 0 | 0 |
| Rg in range | 50/50 | ~100% |
| Ramachandran < 5% | 0/50 | 97% |

### Explore mode (conformational sampling)

To sample beyond the training distribution, switch to explore mode and increase the lag:

```bash
python scripts/generate_md.py \
    --checkpoint  checkpoints/wt_2bead_200ep.pt \
    --frames      data/wt_2bead.pt \
    --tau         5 \
    --steps       1000 \
    --n_chains    4 \
    --min_energy \
    --k_clash     5.0 \
    --sample_mode explore \
    --out         my_explore_run
```

Reverts to the last valid frame on structural failure. Expected RMSD 20–25 Å, RMSF ~14 Å over 1 μs.

---

**All `generate_md.py` flags:**

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
| `--diff_steps` | `50` | Reverse diffusion steps per sample |
| `--eta` | `1.0` | DDPM stochasticity (0 = deterministic DDIM) |
| `--source_frame` | auto | Starting frame index (default: first validation frame) |

**Output files:**
- `trajectory.pdb` — multi-MODEL PDB, one MODEL per step
- `chain_<k>.pdb` — per-chain PDB (when `--n_chains > 1`)
- `metrics.json` — RMSD, RMSF, per-step RMSD array, validity breakdown
- `timing_report.txt` — wall-clock time per step and speedup vs classical MD

**Validity reporting** — each step is checked for:

| Check | Criterion | Notes |
|---|---|---|
| Bond geometry | All bead-bead bonds within ±20% of ideal length | CA–CA 3.8 Å; N–CA 1.46 Å; CA–C 1.52 Å etc. |
| Steric clashes | No non-bonded heavy-atom pair < 2.0 Å | Gly Cβ = Cα position is correctly excluded |
| Radius of gyration | Rg within 0.5×–2.0× of expected (2.2 × P^0.38 Å) | Detects complete unfolding |
| Ramachandran (4-bead only) | < 5% residues outside allowed φ/ψ regions | ~97% of real MD frames pass at this threshold |

---

## Step 4 — All-atom reconstruction

Recover full heavy-atom coordinates from the bead trajectory using the nearest real MD frame as a sidechain template.

### 2-bead reconstruction (demo_2bead → high-quality backbone dihedrals)

```bash
python scripts/reconstruct.py \
    --beads      demo_2bead/trajectory.pdb \
    --traj       WT/WT-sol6.trr \
    --top        WT/WT-sol6.gro \
    --checkpoint checkpoints/wt_2bead_200ep.pt \
    --out        demo_2bead/allatom.pdb
```

Output:
```
Loaded 51 bead frames  (169 residues, mode=2bead)
Loading template trajectory WT/WT-sol6.trr … 5001 frames, 35875 atoms
  169 reconstructable residues
Reconstructing 51 frames …
Saved 51 frames → demo_2bead/allatom.pdb  (5.6 MB)
  Heavy atoms per frame: 1351
```

Each frame finds the nearest real MD frame by Cα-RMSD, then shifts every residue rigidly by (Cα_gen − Cα_template). This preserves the template's backbone φ/ψ angles — Ramachandran plots will look like real MD.

### 4-bead reconstruction (demo_4bead → Kabsch-grafted sidechains)

```bash
python scripts/reconstruct.py \
    --beads      demo_4bead/trajectory.pdb \
    --traj       WT/WT-sol6.trr \
    --top        WT/WT-sol6.gro \
    --checkpoint checkpoints/wt_4bead_200ep.pt \
    --out        demo_4bead/allatom.pdb
```

Output:
```
Loaded 51 bead frames  (169 residues, mode=4bead)
Reconstructing 51 frames …
Saved 51 frames → demo_4bead/allatom.pdb  (5.6 MB)
  Heavy atoms per frame: 1351
```

For each residue, the template backbone (N, Cα, C) is Kabsch-superposed onto the generated backbone to rotate deep sidechain atoms (Cγ, Cδ, …) into the correct local frame. N, Cα, C, Cβ come directly from the generated coordinates. Carbonyl O is placed geometrically in the peptide plane (C=O = 1.229 Å). Backbone dihedrals reflect the model's poor φ/ψ geometry — use the 2-bead reconstruction for dihedral analysis.

---

## Visualisation

The demo all-atom PDBs are ready to open immediately after cloning:

```bash
# 2-bead demo — backbone dihedrals match real MD
pymol demo_2bead/allatom.pdb

# 4-bead demo — per-residue Kabsch sidechain placement
pymol demo_4bead/allatom.pdb

# Compare both at once
pymol demo_2bead/allatom.pdb demo_4bead/allatom.pdb

# Bead-model trajectories (before reconstruction)
pymol demo_2bead/trajectory.pdb   # shows Cα and Cβ
pymol demo_4bead/trajectory.pdb   # shows N, Cα, C, Cβ

# VMD
vmd demo_2bead/allatom.pdb
```

---

## Demo run summary

Both demos use mimic mode, τ=2 (400 ps/step), 50 steps (20 ns), same starting frame.

| | demo_2bead | demo_4bead |
|---|---|---|
| Atoms per residue | Cα, Cβ | N, Cα, C, Cβ |
| Final RMSD | 7.6 Å | 8.5 Å |
| RMSF (mean / max) | 2.4 / 9.1 Å | 2.5 / 9.4 Å |
| Valid steps | 88% | 0% (Ramachandran†) |
| Speedup vs MD | 1439× | 1116× |
| Reconstruction | rigid Cα shift | per-residue Kabsch |
| Backbone φ/ψ quality | ✓ template quality | ✗ model generates poor dihedrals |
| Sidechain placement | template rotamers | Kabsch-rotated into generated frame |

†Bond violations and steric clashes are both 0; only the Ramachandran check fails due to the Cartesian model's lack of dihedral constraints.

To scale up, increase `--steps` (e.g. `--steps 2500` for 1 μs) or switch to `--sample_mode explore` for wider conformational sampling.
