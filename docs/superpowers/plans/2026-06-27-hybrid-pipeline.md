# Hybrid ML-MD Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/hybrid_pipeline.py` — a four-stage CLI that combines the SE(3) PropagatorNet with OpenMM implicit-solvent MD to produce a diverse conformation library, a Markov State Model, or a free energy surface, selected by `--objective {explore,kinetics,fes}`.

**Architecture:** Sequential pipeline: (1) model proposals via existing `explore_conformations.py`, (2) Cα→all-atom reconstruction via existing `AllAtomReconstructor`, (3) parallel OpenMM MD runs via `ProcessPoolExecutor`, (4) objective-specific analysis. Each stage writes a `.stageN_done` marker and is skipped on re-run.

**Tech Stack:** PyTorch (existing), OpenMM ≥ 8.0 (new, implicit solvent MD), mdtraj ≥ 1.9 (already installed, trajectory I/O), PyEMMA (new, kinetics objective only), numpy, scipy.

## Global Constraints

- Python ≥ 3.9
- All output paths are relative to `--out` directory
- Output layout: `<out>/proposals/`, `<out>/allatom/`, `<out>/md_runs/<id>/`, `<out>/results/`
- `--objective` accepts exactly: `explore`, `kinetics`, `fes`
- Default `--n_parallel 4` (ProcessPoolExecutor workers)
- MD force field: `amber14-all.xml` + implicit solvent `implicit/gbn2.xml` (GBn2)
- MD temperature: 310 K, LangevinMiddleIntegrator, friction 1 ps⁻¹, timestep 2 fs
- MD trajectory saved every 5000 steps (= every 10 ps)
- Energy minimization: 1000 steps before every MD run
- Stability criterion: `rmsd_std_A < 3.0` AND `rmsd_final_A < 8.0`
- Reconstruction: `AllAtomReconstructor.reconstruct_frame_ca()` from `lsmd/reconstruct.py`
- Cα PDB coordinates are in Å (written by `lsmd/decoder.write_ca_pdb`)
- mdtraj loads coordinates in nm; multiply by 10 to get Å
- Failed structures logged to `<out>/allatom/failed.txt`; pipeline continues
- CV basis loaded from `<out>/proposals/cv_basis.pt` (written by `explore_conformations.py`)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `lsmd/md_validation.py` | Create | OpenMM MD runner; `run_md()` |
| `lsmd/pipeline_analysis.py` | Create | Three analysis functions; helpers |
| `scripts/hybrid_pipeline.py` | Create | Orchestrator; argument parsing; stage management |
| `tests/test_md_validation.py` | Create | Tests for `run_md` |
| `tests/test_pipeline_analysis.py` | Create | Tests for analysis functions |
| `tests/test_hybrid_pipeline.py` | Create | Tests for orchestrator |

---

## Task 1: `lsmd/md_validation.py` — OpenMM MD Runner

**Files:**
- Create: `lsmd/md_validation.py`
- Test: `tests/test_md_validation.py`

**Interfaces:**
- Produces: `run_md(pdb_path, out_dir, md_ns, temp_K=310.0, n_steps_min=1000) -> dict`
  - Returns dict with keys: `id`, `md_ns`, `final_pe_kJ`, `rmsd_initial_A`, `rmsd_final_A`, `rmsd_mean_A`, `rmsd_std_A`, `stable`, `error`
  - Writes: `<out_dir>/trajectory.dcd`, `<out_dir>/topology.pdb`, `<out_dir>/metrics.json`
  - If `<out_dir>/metrics.json` already exists, reads and returns it (checkpoint)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_md_validation.py
import json
import os
import pytest
from pathlib import Path
from lsmd.md_validation import run_md


def test_checkpoint_returns_cached(tmp_path):
    """run_md returns cached metrics.json without running MD."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    cached = {
        "id": "test", "md_ns": 10, "final_pe_kJ": -1234.5,
        "rmsd_initial_A": 0.0, "rmsd_final_A": 1.5,
        "rmsd_mean_A": 1.2, "rmsd_std_A": 0.3,
        "stable": True, "error": None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(cached))
    result = run_md("nonexistent.pdb", str(out_dir), md_ns=10)
    assert result == cached


def test_run_md_missing_openmm_raises(tmp_path, monkeypatch):
    """run_md raises ImportError when openmm is unavailable."""
    import lsmd.md_validation as mdv
    monkeypatch.setattr(mdv, "HAS_OPENMM", False)
    with pytest.raises(ImportError, match="openmm is required"):
        run_md("dummy.pdb", str(tmp_path / "out"), md_ns=1)


def test_run_md_bad_pdb_writes_error(tmp_path):
    """run_md on a nonexistent PDB writes error to metrics.json, returns dict."""
    openmm = pytest.importorskip("openmm")
    out_dir = tmp_path / "run"
    result = run_md(str(tmp_path / "ghost.pdb"), str(out_dir), md_ns=0.001)
    assert result["error"] is not None
    assert result["stable"] is False
    assert (out_dir / "metrics.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_md_validation.py -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'run_md' from 'lsmd.md_validation'`

- [ ] **Step 3: Write `lsmd/md_validation.py`**

```python
"""OpenMM implicit-solvent MD validation for all-atom protein structures."""
import json
import os

import numpy as np

try:
    import openmm as omm
    import openmm.app as app
    import openmm.unit as unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


def run_md(pdb_path, out_dir, md_ns, temp_K=310.0, n_steps_min=1000):
    """Run AMBER14/GBn2 implicit-solvent MD on a heavy-atom PDB structure.

    Args:
        pdb_path (str):   Path to all-atom heavy-atom PDB (no H, no solvent).
        out_dir (str):    Directory for trajectory.dcd, topology.pdb, metrics.json.
        md_ns (float):    Simulation length in nanoseconds.
        temp_K (float):   Temperature in Kelvin (default 310.0).
        n_steps_min (int):Energy minimisation steps (default 1000).

    Returns:
        dict: id, md_ns, final_pe_kJ, rmsd_initial_A, rmsd_final_A,
              rmsd_mean_A, rmsd_std_A, stable, error
    """
    if not HAS_OPENMM:
        raise ImportError(
            "openmm is required: conda install -c conda-forge openmm"
        )

    struct_id = os.path.splitext(os.path.basename(pdb_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")

    # Checkpoint: return cached result if already computed
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            return json.load(fh)

    traj_path = os.path.join(out_dir, "trajectory.dcd")
    top_path  = os.path.join(out_dir, "topology.pdb")

    result = {
        "id": struct_id, "md_ns": md_ns,
        "final_pe_kJ": None,
        "rmsd_initial_A": None, "rmsd_final_A": None,
        "rmsd_mean_A": None, "rmsd_std_A": None,
        "stable": False, "error": None,
    }

    try:
        pdb = app.PDBFile(pdb_path)
        forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            implicitSolvent=app.GBn2,
            soluteDielectric=1.0,
            solventDielectric=78.5,
            hydrogenMass=1.5 * unit.amu,
        )
        integrator = omm.LangevinMiddleIntegrator(
            temp_K * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        omm.LocalEnergyMinimizer.minimize(
            simulation.context, maxIterations=n_steps_min
        )

        # Save minimised structure as topology reference for later mdtraj loading
        with open(top_path, "w") as fh:
            app.PDBFile.writeFile(
                simulation.topology,
                simulation.context.getState(getPositions=True).getPositions(),
                fh,
            )

        # Reporters
        simulation.reporters.append(app.DCDReporter(traj_path, 5000))

        # Run MD: ns → steps at 2 fs/step
        simulation.step(int(md_ns * 1e6 / 2))

        # Final potential energy
        state_f = simulation.context.getState(getPositions=True, getEnergy=True)
        pe = state_f.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # Per-frame RMSD via mdtraj (already installed)
        import mdtraj as md
        traj = md.load(traj_path, top=top_path)
        ca_idx = traj.topology.select("name CA")
        ca_traj = traj.atom_slice(ca_idx)
        rmsd_nm = md.rmsd(ca_traj, ca_traj, frame=0)   # nm, relative to frame 0
        rmsd_A  = rmsd_nm * 10.0                        # → Å

        result.update({
            "final_pe_kJ":   round(float(pe), 2),
            "rmsd_initial_A": 0.0,
            "rmsd_final_A":  round(float(rmsd_A[-1]), 4),
            "rmsd_mean_A":   round(float(rmsd_A.mean()), 4),
            "rmsd_std_A":    round(float(rmsd_A.std()), 4),
            "stable": float(rmsd_A.std()) < 3.0 and float(rmsd_A[-1]) < 8.0,
            "error": None,
        })

    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    with open(metrics_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_md_validation.py -v
```
Expected: all 3 tests PASS (OpenMM smoke test skipped if not installed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/md_validation.py tests/test_md_validation.py
git commit -m "feat: add md_validation.run_md() — OpenMM implicit-solvent MD runner"
```

---

## Task 2: `lsmd/pipeline_analysis.py` — `analyze_explore`

**Files:**
- Create: `lsmd/pipeline_analysis.py`
- Test: `tests/test_pipeline_analysis.py`

**Interfaces:**
- Consumes: `md_runs_dir` with `<id>/metrics.json` and `<id>/trajectory.dcd` + `<id>/topology.pdb`
- Produces: `analyze_explore(md_runs_dir, out_dir, rmsd_cutoff_A=2.0) -> dict`
  - Writes: `<out_dir>/library/<id>.pdb` (one per cluster representative)
  - Writes: `<out_dir>/cluster_summary.json`
  - Returns: cluster summary dict
- Internal helpers (also tested): `_pairwise_rmsd(ca_list)`, `_cluster_structures(ca_list, cutoff)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_analysis.py
import json
import os
import numpy as np
import pytest
from pathlib import Path


def _write_fake_md_run(md_runs_dir, run_id, ca_coords_A, stable=True):
    """Write fake metrics.json for a stable or unstable run (no DCD needed for unit tests)."""
    run_dir = Path(md_runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "id": run_id, "md_ns": 10,
        "final_pe_kJ": -1000.0,
        "rmsd_initial_A": 0.0, "rmsd_final_A": 1.0,
        "rmsd_mean_A": 0.8, "rmsd_std_A": 0.2,
        "stable": stable, "error": None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    # Store coords for frame loading mock
    np.save(str(run_dir / "_ca_coords.npy"), ca_coords_A)


def test_pairwise_rmsd_zero_diagonal():
    from lsmd.pipeline_analysis import _pairwise_rmsd
    ca = [np.zeros((5, 3)), np.ones((5, 3))]
    mat = _pairwise_rmsd(ca)
    assert mat.shape == (2, 2)
    assert mat[0, 0] == pytest.approx(0.0)
    assert mat[1, 1] == pytest.approx(0.0)
    assert mat[0, 1] == pytest.approx(mat[1, 0])
    assert mat[0, 1] > 0


def test_pairwise_rmsd_known_value():
    from lsmd.pipeline_analysis import _pairwise_rmsd
    # Two structures: one shifted by 1 Å along x for all 4 residues
    ca_a = np.zeros((4, 3))
    ca_b = np.zeros((4, 3)); ca_b[:, 0] = 1.0
    mat = _pairwise_rmsd([ca_a, ca_b])
    assert mat[0, 1] == pytest.approx(1.0, abs=1e-5)


def test_cluster_structures_two_groups():
    from lsmd.pipeline_analysis import _cluster_structures
    # 4 structures: 2 near (0,0,0), 2 near (20,0,0) — should cluster at 2 Å
    rng = np.random.default_rng(42)
    group_a = [rng.normal(0,  0.1, (10, 3)) for _ in range(2)]
    group_b = [rng.normal(20, 0.1, (10, 3)) for _ in range(2)]
    ca_list = group_a + group_b
    labels, mat = _cluster_structures(ca_list, rmsd_cutoff_A=2.0)
    assert labels.shape == (4,)
    # The two groups should be in different clusters
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_structures_single_structure():
    from lsmd.pipeline_analysis import _cluster_structures
    labels, mat = _cluster_structures([np.zeros((5, 3))], rmsd_cutoff_A=2.0)
    assert len(labels) == 1
    assert labels[0] == 1


def test_analyze_explore_filters_unstable(tmp_path, monkeypatch):
    from lsmd.pipeline_analysis import analyze_explore
    md_runs = tmp_path / "md_runs"
    # 2 stable + 1 unstable
    _write_fake_md_run(md_runs, "00001", np.zeros((10, 3)), stable=True)
    _write_fake_md_run(md_runs, "00002", np.ones((10, 3)) * 0.5, stable=True)
    _write_fake_md_run(md_runs, "00003", np.ones((10, 3)) * 100, stable=False)

    # Monkeypatch the DCD loader to return the saved npy coords
    def _mock_load_frames(md_runs_dir):
        frames, ids = [], []
        for d in sorted(os.listdir(md_runs_dir)):
            m = json.loads((Path(md_runs_dir) / d / "metrics.json").read_text())
            if not m["stable"]:
                continue
            coords = np.load(str(Path(md_runs_dir) / d / "_ca_coords.npy"))
            frames.append(coords)
            ids.append(d)
        return frames, ids

    import lsmd.pipeline_analysis as pa
    monkeypatch.setattr(pa, "_load_stable_ca_frames", _mock_load_frames)

    out_dir = tmp_path / "results" / "explore"
    result = analyze_explore(str(md_runs), str(out_dir))
    assert result["n_proposals_attempted"] == 3
    assert result["n_stable"] == 2
    summary_path = out_dir / "cluster_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["n_stable"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_analysis.py -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name '_pairwise_rmsd' from 'lsmd.pipeline_analysis'`

- [ ] **Step 3: Write `lsmd/pipeline_analysis.py` with `analyze_explore` and helpers**

```python
"""Objective-specific analysis for the hybrid ML-MD pipeline."""
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pairwise_rmsd(ca_list):
    """Compute [M, M] pairwise RMSD matrix from list of [N, 3] Å arrays."""
    M = len(ca_list)
    mat = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(i + 1, M):
            diff = ca_list[i] - ca_list[j]
            rmsd = float(np.sqrt((diff ** 2).sum(-1).mean()))
            mat[i, j] = mat[j, i] = rmsd
    return mat


def _medoid(indices, rmsd_matrix):
    """Return the medoid index (min sum of distances within cluster)."""
    sub = rmsd_matrix[np.ix_(indices, indices)]
    return indices[int(sub.sum(axis=1).argmin())]


def _cluster_structures(ca_list, rmsd_cutoff_A):
    """Ward hierarchical clustering on Cα RMSD.

    Returns:
        labels: [M] int array, cluster index per structure (1-based)
        rmsd_matrix: [M, M] float32 pairwise RMSD matrix in Å
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    M = len(ca_list)
    if M == 0:
        return np.array([], dtype=int), np.zeros((0, 0), dtype=np.float32)
    if M == 1:
        return np.array([1], dtype=int), np.zeros((1, 1), dtype=np.float32)
    mat = _pairwise_rmsd(ca_list)
    condensed = squareform(mat, checks=False)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=rmsd_cutoff_A, criterion="distance")
    return labels, mat


def _load_stable_ca_frames(md_runs_dir):
    """Load the final Cα frame from each stable MD run.

    Returns:
        frames: list of [N, 3] Å arrays
        ids:    list of run_id strings in the same order
    """
    import mdtraj as md

    frames, ids = [], []
    for run_id in sorted(os.listdir(md_runs_dir)):
        metrics_path = os.path.join(md_runs_dir, run_id, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as fh:
            m = json.load(fh)
        if not m.get("stable", False):
            continue
        traj_path = os.path.join(md_runs_dir, run_id, "trajectory.dcd")
        top_path  = os.path.join(md_runs_dir, run_id, "topology.pdb")
        if not os.path.exists(traj_path):
            continue
        traj   = md.load(traj_path, top=top_path)
        ca_idx = traj.topology.select("name CA")
        ca_nm  = traj.atom_slice(ca_idx)[-1].xyz[0]   # [N, 3] nm
        frames.append(ca_nm * 10.0)                    # → Å
        ids.append(run_id)
    return frames, ids


# ---------------------------------------------------------------------------
# Explore: diverse stable library
# ---------------------------------------------------------------------------

def analyze_explore(md_runs_dir, out_dir, rmsd_cutoff_A=2.0):
    """Cluster stable MD structures into a diverse library.

    Args:
        md_runs_dir (str): Directory containing per-structure MD run subdirs.
        out_dir (str):     Output directory for library/ and cluster_summary.json.
        rmsd_cutoff_A (float): Ward clustering distance cutoff in Å (default 2.0).

    Returns:
        dict: n_proposals_attempted, n_stable, n_clusters, representatives list.
    """
    import mdtraj as md

    os.makedirs(out_dir, exist_ok=True)
    lib_dir = os.path.join(out_dir, "library")
    os.makedirs(lib_dir, exist_ok=True)

    # Count total runs attempted
    all_run_ids = [d for d in os.listdir(md_runs_dir)
                   if os.path.exists(os.path.join(md_runs_dir, d, "metrics.json"))]
    n_attempted = len(all_run_ids)

    frames, stable_ids = _load_stable_ca_frames(md_runs_dir)
    n_stable = len(frames)

    representatives = []
    if n_stable == 0:
        summary = {
            "n_proposals_attempted": n_attempted,
            "n_stable": 0,
            "n_clusters": 0,
            "rmsd_cutoff_A": rmsd_cutoff_A,
            "representatives": [],
        }
        with open(os.path.join(out_dir, "cluster_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    labels, rmsd_matrix = _cluster_structures(frames, rmsd_cutoff_A)
    unique_labels = sorted(set(labels))

    for cl in unique_labels:
        members = np.where(labels == cl)[0]
        med_idx = _medoid(members, rmsd_matrix)
        run_id  = stable_ids[med_idx]

        # Load all-atom PDB of medoid and copy to library
        traj_path = os.path.join(md_runs_dir, run_id, "trajectory.dcd")
        top_path  = os.path.join(md_runs_dir, run_id, "topology.pdb")
        traj = md.load(traj_path, top=top_path)
        traj[-1].save_pdb(os.path.join(lib_dir, f"cluster{cl:04d}_{run_id}.pdb"))

        # RMSD-to-native placeholder (populated if shard available)
        representatives.append({
            "cluster_id": int(cl),
            "size": int(len(members)),
            "medoid_id": run_id,
        })

    summary = {
        "n_proposals_attempted": n_attempted,
        "n_stable": n_stable,
        "n_clusters": len(unique_labels),
        "rmsd_cutoff_A": rmsd_cutoff_A,
        "representatives": representatives,
    }
    with open(os.path.join(out_dir, "cluster_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"explore: {n_stable}/{n_attempted} stable → {len(unique_labels)} clusters")
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_analysis.py -v -k "explore or pairwise or cluster"
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add lsmd/pipeline_analysis.py tests/test_pipeline_analysis.py
git commit -m "feat: add pipeline_analysis.analyze_explore — Ward clustering of stable MD structures"
```

---

## Task 3: `lsmd/pipeline_analysis.py` — `analyze_fes`

**Files:**
- Modify: `lsmd/pipeline_analysis.py` (append `analyze_fes`)
- Modify: `tests/test_pipeline_analysis.py` (append FES tests)

**Interfaces:**
- Consumes: `md_runs_dir` with stable run trajectories; `cv_basis_path` pointing to a `cv_basis.pt` saved by `explore_conformations.py`
- Produces: `analyze_fes(md_runs_dir, cv_basis_path, out_dir, temp_K=310.0, n_bins=50) -> dict`
  - Writes: `<out_dir>/fes.npy`, `<out_dir>/cv_edges.npy`, `<out_dir>/fes.png`, `<out_dir>/fes_summary.json`

- [ ] **Step 1: Append failing FES tests to `tests/test_pipeline_analysis.py`**

```python
def test_analyze_fes_boltzmann_inversion(tmp_path, monkeypatch):
    """FES minimum is 0.0 and values are non-negative."""
    from lsmd.pipeline_analysis import analyze_fes
    from lsmd.cv_guidance import CVSpace
    import torch

    # Build a tiny CVSpace (5 residues, n_pc=2) and save it
    rng = torch.Generator(); rng.manual_seed(0)
    ca_ref = torch.randn(20, 5, 3, generator=rng)
    cv_space = CVSpace(n_pc=2)
    cv_space.fit(ca_ref)
    cv_basis_path = str(tmp_path / "cv_basis.pt")
    cv_space.save(cv_basis_path)

    # Fake MD runs: 3 stable, each with known CA coords
    md_runs = tmp_path / "md_runs"
    n_res = 5

    def _mock_load_all_ca(md_runs_dir, n_frames_per_run=10):
        # Return synthetic frames without loading real DCD files
        all_frames = torch.randn(30, n_res, 3, generator=rng)
        return all_frames, 30

    import lsmd.pipeline_analysis as pa
    monkeypatch.setattr(pa, "_load_all_ca_frames", _mock_load_all_ca)

    out_dir = tmp_path / "fes"
    result = analyze_fes(str(md_runs), cv_basis_path, str(out_dir), temp_K=310.0, n_bins=10)

    assert result["n_frames_stable"] == 30
    assert result["fes_min_kcal"] == pytest.approx(0.0)
    assert result["fes_max_kcal"] >= 0.0
    assert (out_dir / "fes.npy").exists()
    fes = np.load(str(out_dir / "fes.npy"))
    assert fes.shape == (10, 10)
    assert fes.min() == pytest.approx(0.0, abs=1e-6)
    assert (fes >= 0).all()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_pipeline_analysis.py::test_analyze_fes_boltzmann_inversion -v
```
Expected: `AttributeError: module 'lsmd.pipeline_analysis' has no attribute 'analyze_fes'`

- [ ] **Step 3: Append `_load_all_ca_frames` and `analyze_fes` to `lsmd/pipeline_analysis.py`**

```python
# ---------------------------------------------------------------------------
# Helpers for FES and kinetics: load all frames from stable runs
# ---------------------------------------------------------------------------

def _load_all_ca_frames(md_runs_dir, n_frames_per_run=None):
    """Load all Cα frames from every stable MD run.

    Returns:
        all_frames: torch.Tensor [T, N, 3] in Å
        n_frames:   int — total frames loaded
    """
    import torch
    import mdtraj as md

    frame_list = []
    for run_id in sorted(os.listdir(md_runs_dir)):
        metrics_path = os.path.join(md_runs_dir, run_id, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as fh:
            m = json.load(fh)
        if not m.get("stable", False):
            continue
        traj_path = os.path.join(md_runs_dir, run_id, "trajectory.dcd")
        top_path  = os.path.join(md_runs_dir, run_id, "topology.pdb")
        if not os.path.exists(traj_path):
            continue
        traj   = md.load(traj_path, top=top_path)
        ca_idx = traj.topology.select("name CA")
        ca_nm  = traj.atom_slice(ca_idx).xyz          # [F, N, 3] nm
        ca_A   = torch.tensor(ca_nm * 10.0)           # → Å
        if n_frames_per_run is not None:
            ca_A = ca_A[:n_frames_per_run]
        frame_list.append(ca_A)

    if not frame_list:
        return torch.zeros(0), 0
    all_frames = torch.cat(frame_list, dim=0)   # [T, N, 3]
    return all_frames, all_frames.shape[0]


# ---------------------------------------------------------------------------
# FES: free energy surface over CV space
# ---------------------------------------------------------------------------

_kB_KCAL = 0.001987   # kcal / (mol · K)


def analyze_fes(md_runs_dir, cv_basis_path, out_dir,
                temp_K=310.0, n_bins=50):
    """Estimate free energy surface by projecting MD frames onto CV space.

    Args:
        md_runs_dir (str):   Directory with per-run MD subdirs.
        cv_basis_path (str): Path to cv_basis.pt written by explore_conformations.py.
        out_dir (str):       Output directory.
        temp_K (float):      Temperature for Boltzmann inversion (default 310.0).
        n_bins (int):        Histogram bins per CV axis (default 50).

    Returns:
        dict: n_frames_total, n_frames_stable, temp_K, fes_min_kcal, fes_max_kcal, n_bins.
    """
    import torch
    from lsmd.cv_guidance import CVSpace

    os.makedirs(out_dir, exist_ok=True)

    cv_space = CVSpace.load(cv_basis_path)
    cv_space.to("cpu")

    all_frames, n_frames = _load_all_ca_frames(md_runs_dir)

    if n_frames == 0:
        print("fes: no stable frames found")
        summary = {"n_frames_stable": 0, "fes_min_kcal": None, "fes_max_kcal": None,
                   "temp_K": temp_K, "n_bins": n_bins}
        with open(os.path.join(out_dir, "fes_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    # Project all frames onto first two CVs (PC1, PC2)
    with torch.no_grad():
        cv_all = cv_space.project_batch(all_frames)   # [T, n_cv+2]
    pc1 = cv_all[:, 0].numpy()
    pc2 = cv_all[:, 1].numpy()

    # 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        pc1, pc2, bins=n_bins, density=True
    )

    # Boltzmann inversion: F = -kT ln P, shift minimum to 0
    kT = _kB_KCAL * temp_K
    with np.errstate(divide="ignore"):
        fes = -kT * np.log(hist + 1e-12)
    fes -= fes.min()

    np.save(os.path.join(out_dir, "fes.npy"), fes.astype(np.float32))
    np.save(os.path.join(out_dir, "cv_edges.npy"),
            np.array([x_edges, y_edges], dtype=object))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        pcm = ax.pcolormesh(x_edges, y_edges, fes.T, cmap="viridis_r", vmin=0)
        plt.colorbar(pcm, ax=ax, label="FES (kcal/mol)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"FES — {n_frames} frames, T={temp_K} K")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fes.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    summary = {
        "n_frames_total": n_frames,
        "n_frames_stable": n_frames,
        "temp_K": temp_K,
        "n_bins": n_bins,
        "fes_min_kcal": round(float(fes.min()), 4),
        "fes_max_kcal": round(float(fes.max()), 4),
    }
    with open(os.path.join(out_dir, "fes_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"fes: {n_frames} frames → FES max {fes.max():.2f} kcal/mol")
    return summary
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pipeline_analysis.py::test_analyze_fes_boltzmann_inversion -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lsmd/pipeline_analysis.py tests/test_pipeline_analysis.py
git commit -m "feat: add pipeline_analysis.analyze_fes — FES via CV-space Boltzmann inversion"
```

---

## Task 4: `lsmd/pipeline_analysis.py` — `analyze_kinetics`

**Files:**
- Modify: `lsmd/pipeline_analysis.py` (append `analyze_kinetics`)
- Modify: `tests/test_pipeline_analysis.py` (append kinetics test)

**Interfaces:**
- Consumes: stable MD trajectories in `md_runs_dir`
- Produces: `analyze_kinetics(md_runs_dir, out_dir, tica_lag=50, n_clusters=100, msm_lag=5) -> dict`
  - Writes: `transition_matrix.npy`, `timescales.json`, `timescales.png`, `tica_projection.npy`, `state_assignments.npy`, `msm_summary.json`
  - Raises `ImportError` with install instructions if PyEMMA not installed

- [ ] **Step 1: Append failing kinetics test to `tests/test_pipeline_analysis.py`**

```python
def test_analyze_kinetics_no_pyemma_raises(tmp_path, monkeypatch):
    """analyze_kinetics raises ImportError when PyEMMA is unavailable."""
    import lsmd.pipeline_analysis as pa
    monkeypatch.setattr(pa, "_HAS_PYEMMA", False)
    with pytest.raises(ImportError, match="pyemma is required"):
        pa.analyze_kinetics(str(tmp_path / "md"), str(tmp_path / "out"))


def test_analyze_kinetics_smoke(tmp_path, monkeypatch):
    """analyze_kinetics runs on synthetic featurised data when PyEMMA available."""
    pytest.importorskip("pyemma")
    from lsmd.pipeline_analysis import analyze_kinetics

    # Provide 5 fake trajectories via monkeypatched featuriser
    n_res = 10
    rng = np.random.default_rng(0)

    def _mock_load_featurised(md_runs_dir):
        # Return list of 5 synthetic [100, n_features] arrays
        n_pairs = n_res * (n_res - 1) // 2
        return [rng.standard_normal((100, n_pairs)).astype(np.float32)
                for _ in range(5)]

    import lsmd.pipeline_analysis as pa
    monkeypatch.setattr(pa, "_load_featurised_trajs", _mock_load_featurised)

    out_dir = tmp_path / "kinetics"
    result = analyze_kinetics(
        str(tmp_path / "md"), str(out_dir),
        tica_lag=5, n_clusters=10, msm_lag=2,
    )
    assert result["n_trajectories"] == 5
    assert (out_dir / "transition_matrix.npy").exists()
    assert (out_dir / "msm_summary.json").exists()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_pipeline_analysis.py -k "kinetics" -v
```
Expected: `AttributeError: module 'lsmd.pipeline_analysis' has no attribute 'analyze_kinetics'`

- [ ] **Step 3: Append `_load_featurised_trajs` and `analyze_kinetics` to `lsmd/pipeline_analysis.py`**

Add this near the top of the file after existing imports:
```python
try:
    import pyemma  # noqa: F401
    _HAS_PYEMMA = True
except ImportError:
    _HAS_PYEMMA = False
```

Then append:

```python
# ---------------------------------------------------------------------------
# Kinetics: MSM construction via PyEMMA
# ---------------------------------------------------------------------------

def _load_featurised_trajs(md_runs_dir):
    """Load and featurise stable MD trajectories as Cα pairwise distances.

    Returns:
        list of [T_i, n_features] float32 arrays — one per stable run
    """
    import mdtraj as md

    trajs = []
    for run_id in sorted(os.listdir(md_runs_dir)):
        metrics_path = os.path.join(md_runs_dir, run_id, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as fh:
            m = json.load(fh)
        if not m.get("stable", False):
            continue
        traj_path = os.path.join(md_runs_dir, run_id, "trajectory.dcd")
        top_path  = os.path.join(md_runs_dir, run_id, "topology.pdb")
        if not os.path.exists(traj_path):
            continue
        traj   = md.load(traj_path, top=top_path)
        ca_idx = traj.topology.select("name CA")
        ca_traj = traj.atom_slice(ca_idx)
        n = ca_traj.n_atoms
        # All Cα pairs more than 3 residues apart
        pairs = [(i, j) for i in range(n) for j in range(i + 3, n)]
        if not pairs:
            continue
        dists = md.compute_distances(ca_traj, pairs)  # [T, n_pairs] in nm
        trajs.append(dists.astype(np.float32))
    return trajs


def analyze_kinetics(md_runs_dir, out_dir,
                     tica_lag=50, n_clusters=100, msm_lag=5):
    """Build a Markov State Model from stable MD trajectories.

    Args:
        md_runs_dir (str): Directory with per-run MD subdirs.
        out_dir (str):     Output directory.
        tica_lag (int):    TICA lag time in frames (1 frame = 10 ps; default 50 = 500 ps).
        n_clusters (int):  k-means cluster count (default 100).
        msm_lag (int):     MSM lag time in frames (default 5 = 50 ps).

    Returns:
        dict: n_trajectories, total_frames, n_states, implied_timescales_ns, etc.
    """
    if not _HAS_PYEMMA:
        raise ImportError(
            "pyemma is required: conda install -c conda-forge pyemma"
        )
    import pyemma

    os.makedirs(out_dir, exist_ok=True)

    trajs = _load_featurised_trajs(md_runs_dir)
    n_traj = len(trajs)
    if n_traj == 0:
        print("kinetics: no stable trajectories found")
        return {"n_trajectories": 0}

    total_frames = sum(t.shape[0] for t in trajs)

    # TICA
    tica = pyemma.coordinates.tica(trajs, lag=tica_lag, dim=5, kinetic_map=True)
    tica_output = tica.get_output()
    tica_coords = np.concatenate(tica_output, axis=0)  # [T_total, 5]
    np.save(os.path.join(out_dir, "tica_projection.npy"), tica_coords)

    # k-means clustering
    k = min(n_clusters, total_frames // 2)
    cluster = pyemma.cluster.kmeans(tica_output, k=k, max_iter=100, stride=1)
    np.save(os.path.join(out_dir, "state_assignments.npy"),
            np.concatenate(cluster.dtrajs))

    # MSM
    msm = pyemma.msm.estimate_markov_model(cluster.dtrajs, lag=msm_lag)
    np.save(os.path.join(out_dir, "transition_matrix.npy"),
            msm.transition_matrix.astype(np.float32))

    # Implied timescales (top 5 processes)
    its_frames = msm.timescales(k=min(5, k - 1))
    dt_ps = 10.0  # 1 frame = 10 ps (saved every 5000 steps at 2 fs)
    its_ns = (its_frames * msm_lag * dt_ps / 1000).tolist()

    timescales_path = os.path.join(out_dir, "timescales.json")
    with open(timescales_path, "w") as fh:
        json.dump({"implied_timescales_ns": its_ns, "msm_lag_frames": msm_lag}, fh, indent=2)

    # ITS plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(1, len(its_ns) + 1), its_ns)
        ax.set_xlabel("Process"); ax.set_ylabel("Implied timescale (ns)")
        ax.set_title(f"MSM implied timescales ({k} states, lag={msm_lag} frames)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "timescales.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    summary = {
        "n_trajectories": n_traj,
        "total_frames": total_frames,
        "n_states": k,
        "tica_lag_frames": tica_lag,
        "msm_lag_frames": msm_lag,
        "implied_timescales_ns": its_ns,
    }
    with open(os.path.join(out_dir, "msm_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"kinetics: {n_traj} trajs → {k} states, top ITS={its_ns[0]:.1f} ns")
    return summary
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pipeline_analysis.py -k "kinetics" -v
```
Expected: `test_analyze_kinetics_no_pyemma_raises` PASS; `test_analyze_kinetics_smoke` SKIP or PASS depending on PyEMMA installation

- [ ] **Step 5: Commit**

```bash
git add lsmd/pipeline_analysis.py tests/test_pipeline_analysis.py
git commit -m "feat: add pipeline_analysis.analyze_kinetics — PyEMMA TICA+k-means+MSM"
```

---

## Task 5: `scripts/hybrid_pipeline.py` — Orchestrator

**Files:**
- Create: `scripts/hybrid_pipeline.py`
- Test: `tests/test_hybrid_pipeline.py`

**Interfaces:**
- Consumes: all three `lsmd/` modules above + existing `explore_conformations.py` + existing `lsmd/reconstruct.AllAtomReconstructor`
- Produces: fully wired four-stage CLI with checkpoint detection and parallel MD

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hybrid_pipeline.py
import json
import subprocess
import sys
from pathlib import Path
import pytest


def test_help_shows_objective(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/hybrid_pipeline.py", "--help"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0
    assert "--objective" in result.stdout
    assert "explore" in result.stdout
    assert "kinetics" in result.stdout
    assert "fes" in result.stdout


def test_stage1_skipped_when_done_marker_exists(tmp_path, monkeypatch):
    """Stage 1 is skipped entirely when .stage1_done exists."""
    import scripts.hybrid_pipeline as hp
    (tmp_path / ".stage1_done").touch()
    called = []
    monkeypatch.setattr(hp, "_run_proposals_subprocess", lambda args: called.append(1))
    # Build minimal args namespace
    import argparse
    args = argparse.Namespace(out=str(tmp_path))
    hp.run_proposals(args)
    assert called == [], "Stage 1 should have been skipped"


def test_md_ns_defaults_by_objective():
    """Verify the per-objective MD length defaults."""
    import scripts.hybrid_pipeline as hp
    assert hp._MD_NS_DEFAULT["explore"]   == 10
    assert hp._MD_NS_DEFAULT["kinetics"]  == 50
    assert hp._MD_NS_DEFAULT["fes"]       == 25


def test_missing_required_args(tmp_path):
    """Pipeline exits with error when required args are missing."""
    result = subprocess.run(
        [sys.executable, "scripts/hybrid_pipeline.py",
         "--objective", "explore",
         "--out", str(tmp_path)],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode != 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_hybrid_pipeline.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'scripts.hybrid_pipeline'`

- [ ] **Step 3: Write `scripts/hybrid_pipeline.py`**

```python
"""Hybrid ML-MD pipeline: model proposals → reconstruction → OpenMM MD → analysis.

Usage
-----
python scripts/hybrid_pipeline.py \
    --checkpoint checkpoints/kras_ft.pt \
    --shard      data/kras_wt_shard.pt \
    --ref_traj   WT/WT-sol6.trr --ref_top WT/WT-sol6.gro \
    --objective  explore \
    --n_proposals 200 --n_parallel 4 \
    --device cuda --out kras_hybrid_explore
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

_MD_NS_DEFAULT = {"explore": 10, "kinetics": 50, "fes": 25}
_N_PROPOSALS_DEFAULT = {"explore": 200, "kinetics": 500, "fes": 300}


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _run_proposals_subprocess(args):
    """Call explore_conformations.py as a subprocess for Stage 1."""
    mode = "explore" if args.objective == "explore" else "sample"
    out = os.path.join(args.out, "proposals")
    cmd = [
        sys.executable, "scripts/explore_conformations.py",
        "--checkpoint", args.checkpoint,
        "--shard",      args.shard,
        "--mode",       mode,
        "--n_explore",  str(args.n_proposals),
        "--n_steps",    str(args.n_steps),
        "--tau_ps",     str(args.tau_ps),
        "--temp_K",     str(args.temp_K),
        "--device",     args.device,
        "--out",        out,
        "--seed",       str(args.seed),
    ]
    if args.objective == "explore":
        cmd += [
            "--k_guide",      str(args.k_guide),
            "--sigma_cv",     str(args.sigma_cv),
            "--guide_warmup", str(args.guide_warmup),
        ]
    print(f"[Stage 1] Running model proposals ({args.n_proposals} attempts)…")
    subprocess.run(cmd, check=True)


def run_proposals(args):
    """Stage 1: generate Cα proposals with the ML model."""
    done = os.path.join(args.out, ".stage1_done")
    if os.path.exists(done):
        print("[Stage 1] Already done — skipping.")
        return
    summary_path = os.path.join(args.out, "proposals", "summary.json")
    if os.path.exists(summary_path):
        n = len(json.load(open(summary_path)))
        if n >= args.n_proposals:
            Path(done).touch()
            print(f"[Stage 1] Found {n} existing proposals — skipping.")
            return
    _run_proposals_subprocess(args)
    Path(done).touch()


def run_reconstruction(args):
    """Stage 2: Cα → all-atom heavy-atom PDB via AllAtomReconstructor."""
    done = os.path.join(args.out, ".stage2_done")
    if os.path.exists(done):
        print("[Stage 2] Already done — skipping.")
        return

    from lsmd.reconstruct import AllAtomReconstructor
    import mdtraj as md

    print(f"[Stage 2] Loading template trajectory {args.ref_traj} …")
    rec = AllAtomReconstructor(args.ref_traj, args.ref_top)

    proposals_dir = os.path.join(args.out, "proposals", "candidates")
    allatom_dir   = os.path.join(args.out, "allatom")
    os.makedirs(allatom_dir, exist_ok=True)

    failed = []
    pdb_files = sorted(f for f in os.listdir(proposals_dir) if f.endswith(".pdb"))
    print(f"[Stage 2] Reconstructing {len(pdb_files)} structures…")

    for pdb_file in pdb_files:
        out_pdb = os.path.join(allatom_dir, pdb_file)
        if os.path.exists(out_pdb):
            continue
        ca_pdb = os.path.join(proposals_dir, pdb_file)
        try:
            ca_traj   = md.load(ca_pdb)
            ca_gen_A  = torch.tensor(ca_traj.xyz[0] * 10.0)  # nm → Å
            xyz_A     = rec.reconstruct_frame_ca(ca_gen_A)    # [N_heavy, 3] Å
            out_traj  = md.Trajectory(
                xyz_A[np.newaxis] / 10.0, rec._out_top      # Å → nm
            )
            out_traj.save_pdb(out_pdb)
        except Exception as exc:
            failed.append(f"{pdb_file}: {exc}")

    if failed:
        fail_log = os.path.join(allatom_dir, "failed.txt")
        with open(fail_log, "w") as fh:
            fh.write("\n".join(failed))
        print(f"[Stage 2] {len(failed)} reconstruction failures — see {fail_log}")

    print(f"[Stage 2] Done. {len(pdb_files) - len(failed)} all-atom PDBs written.")
    Path(done).touch()


def _md_worker(pdb_path, out_dir, md_ns, temp_K):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    from lsmd.md_validation import run_md
    return run_md(pdb_path, out_dir, md_ns, temp_K=temp_K)


def run_md_validation(args):
    """Stage 3: parallel OpenMM MD runs."""
    done = os.path.join(args.out, ".stage3_done")
    if os.path.exists(done):
        print("[Stage 3] Already done — skipping.")
        return

    allatom_dir  = os.path.join(args.out, "allatom")
    md_runs_dir  = os.path.join(args.out, "md_runs")
    md_ns = args.md_ns if args.md_ns is not None else _MD_NS_DEFAULT[args.objective]

    pdb_files = sorted(f for f in os.listdir(allatom_dir) if f.endswith(".pdb"))
    tasks = []
    for pdb_file in pdb_files:
        struct_id = pdb_file[:-4]
        run_dir   = os.path.join(md_runs_dir, struct_id)
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            continue
        tasks.append((os.path.join(allatom_dir, pdb_file), run_dir, md_ns))

    print(f"[Stage 3] Running {len(tasks)} MD jobs "
          f"({md_ns} ns each, {args.n_parallel} workers)…")

    n_stable = 0
    with ProcessPoolExecutor(max_workers=args.n_parallel) as ex:
        futs = {ex.submit(_md_worker, pdb, out_d, ns, args.temp_K): pdb
                for pdb, out_d, ns in tasks}
        for fut in as_completed(futs):
            pdb = futs[fut]
            try:
                m = fut.result()
                status = "stable" if m["stable"] else "unstable"
                print(f"  {os.path.basename(pdb)}: {status}  "
                      f"rmsd_final={m['rmsd_final_A']} Å", flush=True)
                if m["stable"]:
                    n_stable += 1
            except Exception as exc:
                print(f"  {os.path.basename(pdb)}: ERROR {exc}", flush=True)

    print(f"[Stage 3] Done. {n_stable}/{len(pdb_files)} structures stable.")
    Path(done).touch()


def run_analysis(args):
    """Stage 4: objective-specific analysis."""
    from lsmd import pipeline_analysis as pa

    md_runs_dir = os.path.join(args.out, "md_runs")
    results_dir = os.path.join(args.out, "results", args.objective)

    print(f"[Stage 4] Running analysis: {args.objective}")

    if args.objective == "explore":
        pa.analyze_explore(md_runs_dir, results_dir)

    elif args.objective == "kinetics":
        pa.analyze_kinetics(md_runs_dir, results_dir)

    elif args.objective == "fes":
        cv_basis = os.path.join(args.out, "proposals", "cv_basis.pt")
        if not os.path.exists(cv_basis):
            raise FileNotFoundError(
                f"cv_basis.pt not found at {cv_basis}. "
                "Run proposals stage first (it is written by explore_conformations.py)."
            )
        pa.analyze_fes(md_runs_dir, cv_basis, results_dir, temp_K=args.temp_K)


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Hybrid ML-MD pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard",      required=True)
    ap.add_argument("--ref_traj",   required=True,
                    help="All-atom MD trajectory for reconstruction template (TRR/DCD/XTC).")
    ap.add_argument("--ref_top",    required=True,
                    help="Topology matching ref_traj (GRO/PDB).")
    ap.add_argument("--objective",  required=True,
                    choices=["explore", "kinetics", "fes"])
    ap.add_argument("--n_proposals", type=int, default=None,
                    help="Model proposals to generate (default: 200/500/300 by objective).")
    ap.add_argument("--n_parallel",  type=int, default=4)
    ap.add_argument("--md_ns",       type=float, default=None,
                    help="MD length per structure in ns (default: 10/50/25 by objective).")
    ap.add_argument("--temp_K",      type=float, default=310.0)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",         default="hybrid_out")
    ap.add_argument("--seed",        type=int, default=42)
    # Proposal stage pass-throughs
    ap.add_argument("--n_steps",     type=int,   default=50)
    ap.add_argument("--tau_ps",      type=float, default=2000.0)
    ap.add_argument("--k_guide",     type=float, default=0.15)
    ap.add_argument("--sigma_cv",    type=float, default=0.8)
    ap.add_argument("--guide_warmup",type=int,   default=20)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.n_proposals is None:
        args.n_proposals = _N_PROPOSALS_DEFAULT[args.objective]
    os.makedirs(args.out, exist_ok=True)

    print(f"Hybrid pipeline — objective={args.objective}  out={args.out}")
    run_proposals(args)
    run_reconstruction(args)
    run_md_validation(args)
    run_analysis(args)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_hybrid_pipeline.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -x -q --ignore=tests/test_atlas.py --ignore=tests/test_mdcath.py 2>&1 | tail -10
```
Expected: all existing tests PASS; new tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/hybrid_pipeline.py tests/test_hybrid_pipeline.py
git commit -m "feat: add hybrid_pipeline.py — four-stage ML+MD pipeline with --objective flag"
```

---

## Self-Review

**Spec coverage:**
- Stage 1 (model proposals) → `run_proposals` + `_run_proposals_subprocess` ✓
- Stage 2 (reconstruction) → `run_reconstruction` using existing `AllAtomReconstructor` ✓
- Stage 3 (OpenMM MD) → `lsmd/md_validation.run_md` + `run_md_validation` ✓
- Stage 4 `explore` → `analyze_explore` with Ward clustering ✓
- Stage 4 `kinetics` → `analyze_kinetics` with TICA+k-means+MSM ✓
- Stage 4 `fes` → `analyze_fes` with CV projection + Boltzmann inversion ✓
- Checkpoint detection (`.stageN_done`) → `run_proposals`, `run_reconstruction`, `run_md_validation` ✓
- Per-structure checkpoint (`metrics.json` exists) → `run_md` ✓
- Failed structures logged to `failed.txt` → `run_reconstruction` ✓
- Stability criterion: `rmsd_std_A < 3.0 AND rmsd_final_A < 8.0` → `run_md` ✓
- `_MD_NS_DEFAULT` and `_N_PROPOSALS_DEFAULT` constants → ✓
- `--n_parallel` via `ProcessPoolExecutor` → `run_md_validation` ✓

**Type consistency:**
- `_load_stable_ca_frames` returns `(list[ndarray], list[str])` — used consistently in `analyze_explore` ✓
- `_load_all_ca_frames` returns `(torch.Tensor, int)` — used consistently in `analyze_fes` ✓
- `_load_featurised_trajs` returns `list[ndarray]` — used consistently in `analyze_kinetics` ✓
- `run_md` returns dict with exactly the keys defined in Task 1 — used in `run_md_validation` ✓
