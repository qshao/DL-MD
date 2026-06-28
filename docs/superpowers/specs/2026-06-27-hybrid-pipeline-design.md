# Hybrid ML-MD Pipeline Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A four-stage pipeline that uses the SE(3) PropagatorNet to generate diverse protein conformations and OpenMM to validate and extract kinetic/thermodynamic properties, all controlled by a single `--objective` flag.

**Architecture:** Sequential: model proposals → Cα→all-atom reconstruction → OpenMM MD validation → objective-specific analysis. Each stage writes files to disk; the next stage reads them. Stages are independently re-runnable via checkpoint detection.

**Tech Stack:** PyTorch (existing model), OpenMM (MD), mdtraj (trajectory I/O, already a dependency), PyEMMA (MSM construction, kinetics objective only), numpy/scipy/matplotlib.

---

## Global Constraints

- Python ≥ 3.9
- All file paths relative to the repo root
- Output directory layout: `<out>/proposals/`, `<out>/allatom/`, `<out>/md_runs/<id>/`, `<out>/results/`
- `--objective` accepts exactly: `explore`, `kinetics`, `fes`
- Default parallelism: `--n_parallel 4` (ProcessPoolExecutor)
- MD force field: AMBER14 protein (`amber14-all.xml`) + implicit solvent GBSAOBCForce (`implicit/gbn2.xml`)
- MD temperature: 310 K, Langevin thermostat, friction 1 ps⁻¹, timestep 2 fs
- MD trajectory saved every 10 ps (every 5000 steps)
- Energy minimization: 1000 steps before every MD run
- Reconstruction: `AllAtomReconstructor.reconstruct_frame_ca()` from existing `lsmd/reconstruct.py`
- CV space for FES: loaded from `<proposals_dir>/cv_basis.pt` (written by `explore_conformations.py`)
- Failed stages (reconstruction error, MD crash) are logged and skipped; pipeline continues

---

## Stage 1 — Model Proposals

**Entry point:** existing `scripts/explore_conformations.py` called as a subprocess (or imported as a module).

**Per-objective settings:**

| Objective | Mode | n_proposals (default) | Notes |
|-----------|------|-----------------------|-------|
| `explore` | `explore` (CV-guided) | 200 | diverse structures via CV repulsion |
| `kinetics` | `sample` (unguided) | 500 | broad coverage, no diversity bias |
| `fes` | `sample` (unguided) | 300 | dense coverage for histogram |

**Outputs written to `<out>/proposals/`:**
- `candidates/*.pdb` — Cα-only PDB per accepted structure
- `summary.json` — per-structure metadata (id, rmsd_native, cv, geometry)
- `cv_basis.pt` — CVSpace PCA basis (written by explore/sample mode; required for FES stage)
- `structures.pt` — stacked Cα tensors [N_accepted, N_res, 3]

**Checkpoint detection:** if `<out>/proposals/summary.json` exists and contains ≥ `n_proposals` entries, skip Stage 1.

---

## Stage 2 — Cα → All-Atom Reconstruction

**Module:** `lsmd/reconstruct.py` → `AllAtomReconstructor.reconstruct_frame_ca(ca_gen)`

**Inputs:**
- Cα PDB files from `<out>/proposals/candidates/`
- `--ref_traj`: all-atom MD trajectory (TRR, DCD, XTC) for template sidechain borrowing
- `--ref_top`: topology file (GRO, PDB) matching `ref_traj`

**Algorithm (already implemented):**
1. Load template trajectory once, cache CA positions and all heavy-atom xyz
2. For each Cα proposal: find nearest template frame by CA-RMSD
3. Rigidly shift each residue's all heavy atoms by `(CA_gen − CA_template)`, preserving rotamer

**Outputs written to `<out>/allatom/`:**
- `<id>.pdb` — all heavy atoms, no hydrogens, no solvent (OpenMM adds H at runtime)

**Error handling:** if `AllAtomReconstructor` raises for a structure, log to `<out>/allatom/failed.txt` and skip.

**Checkpoint detection:** if `<out>/allatom/<id>.pdb` already exists, skip that structure.

---

## Stage 3 — OpenMM MD Validation

**New module:** `lsmd/md_validation.py`

**Per-objective MD length:**

| Objective | MD length | Rationale |
|-----------|-----------|-----------|
| `explore` | 10 ns | stability check only — does the structure hold? |
| `kinetics` | 50 ns | capture local transitions around the proposed state |
| `fes` | 25 ns | thermodynamic relaxation to sample local basin |

**Setup per run (function `run_md`):**
```python
def run_md(pdb_path, out_dir, md_ns, temp_K=310.0, n_steps_min=1000) -> dict:
```
1. Load PDB with `PDBFile`; add missing H with `Modeller.addHydrogens(forcefield)`
2. Create `System` with `forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, implicitSolvent=GBn2, soluteDielectric=1.0, solventDielectric=78.5, hydrogenMass=1.5*amu)`
3. `LangevinMiddleIntegrator(temp_K*kelvin, 1/picosecond, 0.002*picoseconds)`
4. `LocalEnergyMinimizer.minimize(simulation, maxIterations=n_steps_min)`
5. Add `DCDReporter(<out_dir>/trajectory.dcd, reportInterval=5000)` and `StateDataReporter`
6. `simulation.step(md_ns * 1e6 / 2)`  (steps = ns × 500,000)
7. Compute and return metrics dict

**Metrics collected:**
```json
{
  "id": "00042",
  "md_ns": 10,
  "final_pe_kJ": -4521.3,
  "rmsd_initial_A": 1.23,
  "rmsd_final_A": 2.14,
  "rmsd_mean_A": 1.67,
  "rmsd_std_A": 0.31,
  "stable": true,
  "error": null
}
```

**Stability criterion:** `rmsd_std_A < 3.0` AND `rmsd_final_A < 8.0` Å. Structures failing either threshold are marked `"stable": false` and excluded from Stage 4.

**Parallelism:** `concurrent.futures.ProcessPoolExecutor(max_workers=n_parallel)`. Each worker calls `run_md` in an isolated process (OpenMM is not fork-safe).

**Outputs written to `<out>/md_runs/<id>/`:**
- `trajectory.dcd` — full MD trajectory
- `topology.pdb` — first frame (for MDtraj loading)
- `metrics.json` — stability metrics

**Checkpoint detection:** if `<out>/md_runs/<id>/metrics.json` exists, skip that structure.

---

## Stage 4 — Objective-Specific Analysis

### 4A — `explore`: Diverse Stable Library

**New function:** `lsmd/pipeline_analysis.py::analyze_explore(md_runs_dir, proposals_summary, out_dir)`

1. Load `metrics.json` for all completed runs; keep only `stable=true` entries
2. Load final frame from each stable `trajectory.dcd` using mdtraj
3. Extract Cα coordinates; compute pairwise RMSD matrix [N_stable, N_stable]
4. Hierarchical clustering (Ward linkage, `scipy.cluster.hierarchy`) with cutoff 2.0 Å
5. Select cluster medoid (structure closest to cluster centroid) as representative
6. Write representative all-atom PDBs to `<out>/results/explore/library/`
7. Write `<out>/results/explore/cluster_summary.json`:
```json
{
  "n_proposals": 200,
  "n_stable": 183,
  "n_clusters": 47,
  "rmsd_cutoff_A": 2.0,
  "representatives": [
    {"cluster_id": 0, "size": 12, "medoid_id": "00031", "rmsd_to_native_A": 3.21},
    ...
  ]
}
```

### 4B — `kinetics`: Markov State Model

**New function:** `lsmd/pipeline_analysis.py::analyze_kinetics(md_runs_dir, ref_top, out_dir)`

**Dependencies:** PyEMMA (`conda install -c conda-forge pyemma`)

1. Load all stable MD trajectories with mdtraj; align to first frame by Cα
2. Featurize: Cα pairwise distances for all residue pairs > 3 apart → [N_frames, N_features]
3. TICA: `pyemma.coordinates.tica(data, lag=50, dim=5)` → 5 slow ICs
4. k-means clustering: `pyemma.cluster.kmeans(tica_output, k=100, max_iter=100)`
5. MSM: `pyemma.msm.estimate_markov_model(cluster.dtrajs, lag=5)` (lag in frames = 50 ps)
6. Compute implied timescales: `msm.timescales(k=5)` → top 5 processes
7. Chapman-Kolmogorov test: `msm.cktest(5)`
8. Write outputs to `<out>/results/kinetics/`:
   - `transition_matrix.npy` — [k, k] row-stochastic MSM transition matrix
   - `timescales.json` — implied timescales in ns
   - `timescales.png` — ITS plot vs lag time
   - `tica_projection.npy` — [N_frames, 5] TICA coordinates
   - `state_assignments.npy` — [N_frames] discrete state index per frame
   - `msm_summary.json`:
```json
{
  "n_trajectories": 487,
  "total_frames": 121750,
  "n_states": 100,
  "tica_lag_ps": 50,
  "msm_lag_ps": 250,
  "implied_timescales_ns": [142.3, 67.1, 31.4, 18.2, 9.7],
  "n_metastable_pcca": 4
}
```

### 4C — `fes`: Free Energy Surface

**New function:** `lsmd/pipeline_analysis.py::analyze_fes(md_runs_dir, cv_basis_path, ref_coords, out_dir, temp_K=310.0)`

1. Load `cv_basis.pt` CVSpace (fitted during Stage 1)
2. Load all stable MD trajectories (Cα only extracted from all-atom traj via mdtraj)
3. Project all frames onto CV space: `cv_space.project_batch(ca_frames)` → [N_frames, n_cv+2]
4. 2D histogram over PC1 × PC2 with 50×50 bins; normalize to probability density P(x)
5. FES: `F(x) = -kT ln P(x)` in kcal/mol; set minimum to 0
6. Overlay training data distribution (from proposal stage `cv_basis.pt` ref_cv)
7. Write to `<out>/results/fes/`:
   - `fes.npy` — [50, 50] FES array in kcal/mol
   - `cv_edges.npy` — bin edges for PC1 and PC2
   - `fes.png` — 2D heatmap with training data contour overlay
   - `fes_summary.json`:
```json
{
  "n_frames_total": 45600,
  "n_frames_stable": 41230,
  "temp_K": 310.0,
  "cv_dims": ["PC1", "PC2", "PC3", "PC4", "PC5", "Rg", "RMSD"],
  "fes_min_kcal": 0.0,
  "fes_max_kcal": 8.34,
  "n_bins": 50
}
```

---

## Entry Point: `scripts/hybrid_pipeline.py`

```
usage: hybrid_pipeline.py [-h]
  --checkpoint CHECKPOINT
  --shard SHARD
  --ref_traj REF_TRAJ
  --ref_top REF_TOP
  --objective {explore,kinetics,fes}
  [--n_proposals N_PROPOSALS]
  [--n_parallel N_PARALLEL]
  [--md_ns MD_NS]
  [--temp_K TEMP_K]
  [--device DEVICE]
  [--out OUT]
  # Proposal stage pass-throughs (explore objective):
  [--k_guide K_GUIDE]
  [--sigma_cv SIGMA_CV]
  [--guide_warmup GUIDE_WARMUP]
  [--n_steps N_STEPS]
  [--tau_ps TAU_PS]
```

**Execution flow:**
```python
stage1_done = run_proposals(args)      # calls explore_conformations.py
stage2_done = run_reconstruction(args) # calls AllAtomReconstructor
stage3_done = run_md_validation(args)  # calls md_validation.run_md in parallel
run_analysis(args)                     # calls pipeline_analysis.analyze_*
```

Each `run_*` function checks checkpoints before doing work, logs per-structure status, and writes a stage completion marker (`<out>/.stage{N}_done`) on success.

---

## New Files

| File | Purpose |
|------|---------|
| `scripts/hybrid_pipeline.py` | Main orchestrator (~200 lines) |
| `lsmd/md_validation.py` | OpenMM MD runner, `run_md()` function (~120 lines) |
| `lsmd/pipeline_analysis.py` | Three analysis functions for explore/kinetics/fes (~250 lines) |

**Modified files:** none — all existing modules used as-is.

---

## Dependencies

```bash
# Required for all objectives
conda install -c conda-forge openmm mdtraj

# Required for kinetics objective only
conda install -c conda-forge pyemma
```

---

## Example Usage

```bash
# Diverse conformation library (10 ns MD per structure)
python scripts/hybrid_pipeline.py \
    --checkpoint checkpoints/kras_ft.pt \
    --shard data/kras_wt_shard.pt \
    --ref_traj WT/WT-sol6.trr --ref_top WT/WT-sol6.gro \
    --objective explore \
    --n_proposals 200 --n_parallel 4 \
    --device cuda --out kras_hybrid_explore

# Kinetic model (50 ns MD per structure, MSM output)
python scripts/hybrid_pipeline.py \
    --checkpoint checkpoints/kras_ft.pt \
    --shard data/kras_wt_shard.pt \
    --ref_traj WT/WT-sol6.trr --ref_top WT/WT-sol6.gro \
    --objective kinetics \
    --n_proposals 500 --n_parallel 4 \
    --device cuda --out kras_hybrid_kinetics

# Free energy surface (25 ns MD per structure)
python scripts/hybrid_pipeline.py \
    --checkpoint checkpoints/kras_ft.pt \
    --shard data/kras_wt_shard.pt \
    --ref_traj WT/WT-sol6.trr --ref_top WT/WT-sol6.gro \
    --objective fes \
    --n_proposals 300 --n_parallel 4 \
    --device cuda --out kras_hybrid_fes
```
