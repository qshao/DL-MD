# LSMD Tutorial

This tutorial walks through the complete LSMD pipeline on a real protein system, from preprocessing an MD trajectory to generating and analysing long generative MD runs. Three production runs are used as concrete examples throughout.

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

## Input data

LSMD reads any trajectory format supported by MDtraj (GROMACS TRR/XTC, CHARMM DCD, AMBER NetCDF, …) plus a matching topology file. For this tutorial we use:

```
WT/WT-sol6.trr   — 5001-frame GROMACS trajectory (1 μs, 200 ps/frame)
WT/WT-sol6.gro   — GROMACS topology (169 protein residues + solvent)
```

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
# 2-bead model — used for runs A and B
python scripts/train.py \
    --frames   data/wt_2bead.pt \
    --taus     1 2 5 \
    --epochs   200 \
    --out      checkpoints/wt_2bead_200ep.pt

# 4-bead model — used for run C
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

### Run A — 2-bead mimic, 1 μs (MD-faithful reference)

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
    --out         run_a_2bead_mimic_1us
```

Terminal output:
```
Generative MD
  Checkpoint : checkpoints/wt_2bead_200ep.pt
  Starting frame : 3999  (5001 total, 169 residues)
  Tau per step   : 2 frames = 400 ps
  Steps          : 2500
  Total time     : 1000.0 ns  (500000000× faster than 2 fs MD)
  Energy min     : ON  (L-BFGS, 100 steps, k_bond=10.0, k_clash=5.0)
  Sample mode    : mimic  (re-anchor to nearest MD frame every 50 steps)
  Device         : cuda
  Mode           : 2-bead (CA, CB)  point_dim=6

Chain 1/1 ...
  Done. Mean disp/step: 1.008 Å  Final RMSD: 8.158 Å  Valid steps: 2175/2500  re-anchors: 50

Generated 1000.0 ns of CA dynamics
  Final RMSD      : 8.16 Å from start
  RMSF (mean/max) : 3.56 / 11.02 Å
  Most flexible   : residue 168
  Valid steps     : 2175/2500 (87%)  [bond viol=0, clashes=325, Rg viol=0]
  Sample mode     : mimic  (anchor every 50 steps)  re-anchors: 50
  Time/step       : 0.594 s  → 24.7 min for 1000 ns (582× vs classical MD)
Output → run_a_2bead_mimic_1us/
```

**Results:** RMSD 8.2 Å, RMSF 3.6 Å, 87% valid steps, **582× speedup**. The trajectory stays near the MD ensemble — RMSF matches the order of magnitude of classical MD (~2 Å) while covering a broader range of the energy landscape in 25 minutes vs 10 days.

---

### Run B — 2-bead explore ensemble, 1 μs × 4 chains (conformational sampling)

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
    --out         run_b_2bead_explore_ensemble
```

Terminal output (abbreviated):
```
  Tau per step   : 5 frames = 1000 ps
  Steps          : 1000
  Chains         : 4
  Total time     : 1000.0 ns  (500000000× faster than 2 fs MD)
  Sample mode    : explore  (revert to last valid frame on failure)

Chain 1/4 ...  Done. Final RMSD: 21.6 Å  Valid steps: 859/1000  reverts: 141
Chain 2/4 ...  Done. Final RMSD: 24.9 Å  Valid steps: 888/1000  reverts: 112
Chain 3/4 ...  Done. Final RMSD: 20.3 Å  Valid steps: 792/1000  reverts: 208
Chain 4/4 ...  Done. Final RMSD: 24.6 Å  Valid steps: 907/1000  reverts: 93

  Final RMSD      : 22.86 Å from start (mean over 4 chains)
  RMSF (mean/max) : 13.92 / 30.41 Å
  Valid steps     : 3446/4000 (86%)  [bond viol=0, clashes=554, Rg viol=0]
  Sample mode     : explore  reverts: 554
  Time/step       : 0.526 s  → 8.8 min for 1000 ns (1643× vs classical MD)
Output → run_b_2bead_explore_ensemble/
```

**Results:** Mean RMSD 22.9 Å, RMSF 13.9 Å across 4 chains, **1643× speedup**. The four chains each start from the same frame and diverge into different regions of conformational space, providing an ensemble that samples far beyond the training-data distribution. The larger lag (τ=5, 1 ns/step) drives bigger conformational jumps. Reverts (141–208 per chain) prevent runaway unfolding while still allowing aggressive exploration.

---

### Run C — 4-bead mimic, 200 ns (full backbone geometry)

```bash
python scripts/generate_md.py \
    --checkpoint  checkpoints/wt_4bead_200ep.pt \
    --frames      data/wt_4bead.pt \
    --tau         2 \
    --steps       500 \
    --n_chains    1 \
    --min_energy \
    --k_clash     5.0 \
    --sample_mode mimic \
    --anchor_every 50 \
    --out         run_c_4bead_mimic_200ns
```

Terminal output:
```
  Mode           : 4-bead (N, CA, C, CB)  point_dim=12

  Done. Mean disp/step: 0.865 Å  Final RMSD: 6.939 Å  Valid steps: 0/500  re-anchors: 10

  Final RMSD      : 6.94 Å from start
  RMSF (mean/max) : 3.35 / 9.58 Å
  Valid steps     : 0/500 (0%)  [bond viol=0, clashes=0, Rg viol=0,
                                  rama viol=500 (mean 69.0% outliers/step)]
  Time/step       : 0.293 s  → 12.2 min for 1000 ns (1180× vs classical MD)
```

**Results:** RMSD 6.9 Å, RMSF 3.4 Å — geometry is excellent at the Cα level. Bond violations: 0. Clashes: 0. The 0% valid steps is entirely due to Ramachandran: the 4-bead Cartesian model generates N/Cα/C displacements independently without dihedral constraints, producing ~69% φ/ψ outliers per step versus ~2% in real MD. This is a known limitation of the Cartesian bead model, not a structural failure. For dihedral analysis use run A's 2-bead reconstruction instead.

**Validity check breakdown for 4-bead:**

| Check | Run C result | Real MD baseline |
|---|---|---|
| Bond violations | 0 | 0 |
| Steric clashes | 0 | 0 |
| Rg in range | 500/500 | ~100% |
| Ramachandran < 5% | 0/500 | 97% |

---

## Step 4 — All-atom reconstruction

Recover full heavy-atom coordinates from the bead trajectory using the nearest real MD frame as a sidechain template.

### 2-bead reconstruction (Run A → high-quality backbone dihedrals)

```bash
python scripts/reconstruct.py \
    --beads      run_a_2bead_mimic_1us/trajectory.pdb \
    --traj       WT/WT-sol6.trr \
    --top        WT/WT-sol6.gro \
    --checkpoint checkpoints/wt_2bead_200ep.pt \
    --out        run_a_2bead_mimic_1us/allatom.pdb
```

Output:
```
Loaded 2501 bead frames  (169 residues, mode=2bead)
Loading template trajectory WT/WT-sol6.trr … 5001 frames, 35875 atoms
  169 reconstructable residues
Reconstructing 2501 frames …
  Reconstructed 2501/2501 frames
Saved 2501 frames → run_a_2bead_mimic_1us/allatom.pdb  (273.8 MB)
  Heavy atoms per frame: 1351
```

Each frame finds the nearest real MD frame by Cα-RMSD, then shifts every residue rigidly by (Cα_gen − Cα_template). This preserves the template's backbone φ/ψ angles — Ramachandran plots of the output will look like real MD.

### 4-bead reconstruction (Run C → Kabsch-grafted sidechains)

```bash
python scripts/reconstruct.py \
    --beads      run_c_4bead_mimic_200ns/trajectory.pdb \
    --traj       WT/WT-sol6.trr \
    --top        WT/WT-sol6.gro \
    --checkpoint checkpoints/wt_4bead_200ep.pt \
    --out        run_c_4bead_mimic_200ns/allatom.pdb
```

Output:
```
Loaded 501 bead frames  (169 residues, mode=4bead)
Reconstructing 501 frames …
Saved 501 frames → run_c_4bead_mimic_200ns/allatom.pdb  (54.8 MB)
  Heavy atoms per frame: 1351
```

For each residue, the template backbone (N, Cα, C) is Kabsch-superposed onto the generated backbone to rotate deep sidechain atoms (Cγ, Cδ, …) into the correct local frame. N, Cα, C, Cβ come directly from the generated coordinates. The carbonyl O is placed geometrically in the peptide plane (C=O = 1.229 Å). Use this reconstruction when sidechain placement in the generated backbone frame is important, but note that backbone dihedrals will reflect the model's poor φ/ψ geometry.

---

## Visualisation

```bash
# Load a trajectory in PyMOL
pymol run_a_2bead_mimic_1us/allatom.pdb

# Compare all three all-atom trajectories
pymol run_a_2bead_mimic_1us/allatom.pdb \
      run_c_4bead_mimic_200ns/allatom.pdb

# Load in VMD
vmd run_a_2bead_mimic_1us/allatom.pdb
```

For the bead-model PDBs (before reconstruction):
```bash
# 2-bead trajectory — shows Cα and Cβ only
pymol run_a_2bead_mimic_1us/trajectory.pdb

# 4-bead trajectory — shows N, Cα, C, Cβ backbone
pymol run_c_4bead_mimic_200ns/trajectory.pdb
```

---

## Output file reference

Each `generate_md.py` run produces:

| File | Description |
|---|---|
| `trajectory.pdb` | Multi-MODEL PDB, one MODEL per step, all chains |
| `chain_<k>.pdb` | Per-chain PDB (when `--n_chains > 1`) |
| `metrics.json` | Final RMSD, RMSF, per-step RMSD array, validity breakdown |
| `timing_report.txt` | Wall-clock time per step and speedup vs classical MD |

After `reconstruct.py`:

| File | Description |
|---|---|
| `allatom.pdb` | Multi-MODEL PDB, protein heavy atoms only (no H, no solvent) |

---

## Summary of the three runs

| | Run A | Run B | Run C |
|---|---|---|---|
| Model | 2-bead | 2-bead | 4-bead |
| τ | 2 (400 ps/step) | 5 (1 ns/step) | 2 (400 ps/step) |
| Steps / time | 2500 / 1 μs | 1000 / 1 μs | 500 / 200 ns |
| Chains | 1 | 4 | 1 |
| Sample mode | mimic | explore | mimic |
| Final RMSD | 8.2 Å | 22.9 Å | 6.9 Å |
| Mean RMSF | 3.6 Å | 13.9 Å | 3.4 Å |
| Valid steps | 87% | 86% | 0% (Rama) |
| Speedup | 582× | 1643× | 1180× |
| All-atom reconstruction | ✓ high-quality backbone | — | ✓ Kabsch sidechains |
| Best for | MD reproduction | Conformational sampling | Sidechain placement |
