# Active Learning Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sequential active learning loop that explores the conformational ensemble of any protein from a single static PDB, iteratively fine-tuning the PropagatorNet on validated MD data.

**Architecture:** Two new files (`lsmd/active_loop.py`, `scripts/active_learning.py`) plus a one-line guard in `lsmd/cv_guidance.py`. The orchestrator calls existing `rollout()`, `run_md()`, `AllAtomReconstructor`, and `train_transfer.py` without modifying them. Each round writes to `{out}/round_{N}/` and stamps `.done` for crash-safe resume.

**Tech Stack:** Python 3.10+, PyTorch, mdtraj, OpenMM (via existing `run_md()`), numpy, scipy, concurrent.futures.

## Global Constraints

- No new pip/conda dependencies — torch, mdtraj, openmm, numpy, scipy already installed.
- All new logic is in `lsmd/active_loop.py` and `scripts/active_learning.py`.
- Only existing-module change: a single guard in `CVSpace.fit()` (`lsmd/cv_guidance.py:27–50`).
- Fine-tuning always calls `train_transfer.py` as a subprocess with `--resume checkpoints/v2_256h_90k.pt` (universal base), never from the previous round's checkpoint.
- `accumulated_frames.pt` stores `{"R": [F,N,3,3], "t": [F,N,3]}` and is append-only.
- Resume is round-level: a `.done`-stamped round is never re-run.
- All Cα distances in Å throughout (× 10 when converting from mdtraj nm).
- `shard["dt"]` is a Python `float` (ps/frame), not a tensor — `float(shard["dt"])` is valid.
- Device flag `--device cuda` must be passed explicitly (never default to CPU).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `lsmd/cv_guidance.py` | Modify `CVSpace.fit()` | Guard against F < 2 so single-frame shard doesn't crash SVD |
| `lsmd/active_loop.py` | Create | All pure-logic functions: pdb loading, geometry checks, shard building, convergence |
| `tests/test_active_loop.py` | Create | 12 unit tests for `lsmd/active_loop.py` |
| `scripts/active_learning.py` | Create | CLI orchestrator: round loop, subprocess fine-tuning, round checkpointing |

---

## Task 1: CVSpace.fit() Cold-Start Guard

**Files:**
- Modify: `lsmd/cv_guidance.py:27–50`
- Test: `tests/test_active_loop.py`

**Interfaces:**
- Produces: `CVSpace.fit(coords)` now accepts `coords` with F=1 without crashing; when F < 2, `self.components` is all-zeros and PC scores from `project_single()` are 0.

- [ ] **Step 1: Write the failing test**

Create `tests/test_active_loop.py`:

```python
"""Tests for lsmd/active_loop.py and the CVSpace cold-start guard."""
import pytest
import torch
from lsmd.cv_guidance import CVSpace


def test_cvspace_single_frame():
    """CVSpace.fit() must not crash on F=1 and must return 2D CV (Rg+RMSD only)."""
    N = 30
    coords = torch.randn(1, N, 3) * 10.0
    cv = CVSpace(n_pc=5)
    cv.fit(coords)  # must not raise
    assert cv.mean is not None
    assert cv.components is not None
    assert cv.components.shape == (5, N * 3)
    # All PC scores must be 0 (zero components → zero projection)
    proj = cv.project_single(coords[0])
    assert proj.shape == (7,)              # 5 PC + Rg + RMSD
    assert proj[:5].abs().max() < 1e-6    # PC scores ≈ 0


def test_cvspace_multi_frame():
    """CVSpace.fit() still works normally for F >= 2."""
    N = 20
    coords = torch.randn(10, N, 3) * 10.0
    cv = CVSpace(n_pc=3)
    cv.fit(coords)
    proj = cv.project_single(coords[0])
    assert proj.shape == (5,)   # 3 PC + Rg + RMSD
    assert proj.isfinite().all()
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/qshao/DL-MD
pytest tests/test_active_loop.py::test_cvspace_single_frame -xvs 2>&1 | head -30
```

Expected: FAIL — `torch.linalg.svd` of zero matrix or `nan` in std.

- [ ] **Step 3: Apply the guard to CVSpace.fit()**

Open `lsmd/cv_guidance.py`. Replace the entire `fit()` method body (lines 27–50):

```python
    def fit(self, coords: torch.Tensor) -> None:
        """Fit PCA basis from training shard Cα frames.

        Args:
            coords: [F, N, 3] Cα positions from the training shard.
        """
        F, N, _ = coords.shape
        X = coords.reshape(F, N * 3).float()
        mean = X.mean(dim=0)                          # [3N]
        self.mean = mean.cpu()

        if F >= 2:
            _, _, Vh = torch.linalg.svd(X - mean, full_matrices=False)
            self.components = Vh[:self.n_pc].cpu()    # [n_pc, 3N]
        else:
            # F=1: SVD of zero matrix is numerically unstable. Zero components
            # → PC scores stay 0; guidance acts on Rg+RMSD dims only.
            self.components = torch.zeros(self.n_pc, N * 3)

        centroid = coords.mean(dim=1, keepdim=True)   # [F, 1, 3]
        rg = ((coords - centroid) ** 2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()  # [F]
        self.rg_mean = rg.mean().float().cpu()
        self.rg_std  = rg.std().float().nan_to_num(0.0).clamp_min(1e-8).cpu()

        mean_ca = mean.reshape(N, 3)
        rmsd = ((coords.float() - mean_ca.unsqueeze(0)) ** 2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()
        self.rmsd_std = rmsd.std().float().nan_to_num(0.0).clamp_min(1e-8).cpu()
```

The two changes from the original are:
1. `self.mean = mean.cpu()` moved before the SVD block.
2. `if F >= 2` guard around SVD; `else` sets zero components.
3. `.nan_to_num(0.0).clamp_min(1e-8)` on `rg_std` and `rmsd_std` (F=1 → `std()` is `nan`).

- [ ] **Step 4: Run both tests**

```bash
pytest tests/test_active_loop.py::test_cvspace_single_frame tests/test_active_loop.py::test_cvspace_multi_frame -xvs
```

Expected: both PASS.

- [ ] **Step 5: Run full existing CV test suite to check for regressions**

```bash
pytest tests/ -x -q 2>&1 | tail -10
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add lsmd/cv_guidance.py tests/test_active_loop.py
git commit -m "feat: CVSpace.fit() cold-start guard for F<2; add test skeleton"
```

---

## Task 2: lsmd/active_loop.py — Bootstrap Helpers

**Files:**
- Create: `lsmd/active_loop.py`
- Modify: `tests/test_active_loop.py`

**Interfaces:**
- Produces:
  - `_pdb_to_shard(pdb_path, dt_ps=200.0) -> dict` — keys: `{R, t, res_type, chain_id, res_index, n_res, dt, seq}`
  - `_geometry_pass_rate(proposals, ref_bond_A) -> float` — fraction passing bond + clash check
  - `_min_rmsd_kabsch(query, refs) -> float` — batch Kabsch min-RMSD (Å)
  - `bootstrap_check(pdb_path, checkpoint, device, bootstrap_ns, out_dir) -> dict` — returns shard

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_active_loop.py`:

```python
import os, json, tempfile
import numpy as np
from lsmd.active_loop import (
    _pdb_to_shard, _geometry_pass_rate, _min_rmsd_kabsch, bootstrap_check,
)


def _write_tiny_pdb(path, n_res=5):
    """Write a minimal CA-only PDB with ideal 3.8 Å bonds."""
    lines = ["REMARK tiny test PDB"]
    res_names = ["ALA", "GLY", "VAL", "LEU", "ILE"][:n_res]
    atom_names_full = [" N  ", " CA ", " C  ", " O  "]
    coords_per_res = [
        [0.0, 0.0, 0.0],   # N
        [1.458, 0.0, 0.0], # CA
        [2.009, 1.420, 0.0], # C
        [1.251, 2.390, 0.0], # O (approximate)
    ]
    serial = 1
    for ri, rn in enumerate(res_names):
        z_offset = ri * 3.8
        for aname, xyz in zip(atom_names_full, coords_per_res):
            x, y, z = xyz[0], xyz[1], xyz[2] + z_offset
            lines.append(
                f"ATOM  {serial:5d} {aname}{rn:>3s} A{ri+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def test_pdb_to_shard():
    with tempfile.TemporaryDirectory() as tmp:
        pdb = os.path.join(tmp, "test.pdb")
        _write_tiny_pdb(pdb, n_res=5)
        shard = _pdb_to_shard(pdb)
    assert shard["t"].shape == (1, 5, 3)
    assert shard["R"].shape == (1, 5, 3, 3)
    assert shard["res_type"].shape == (5,)
    assert shard["n_res"] == 5
    assert isinstance(shard["dt"], float)


def test_geometry_pass_rate_good():
    """Ideal 3.8 Å bonds → high pass rate."""
    N = 10
    # Build ideal Cα chain: each residue 3.8 Å apart
    ca = torch.zeros(N, 3)
    for i in range(1, N):
        ca[i, 2] = i * 3.8
    proposals = [ca for _ in range(5)]
    rate = _geometry_pass_rate(proposals, ref_bond_A=3.8)
    assert rate >= 0.8


def test_geometry_pass_rate_bad():
    """Bonds at 4.6 Å (outside threshold) → low pass rate."""
    N = 10
    ca = torch.zeros(N, 3)
    for i in range(1, N):
        ca[i, 2] = i * 4.6   # stretched bonds
    proposals = [ca for _ in range(5)]
    rate = _geometry_pass_rate(proposals, ref_bond_A=3.8)
    assert rate == 0.0


def test_min_rmsd_kabsch_identical():
    """Min RMSD of structure to itself must be 0."""
    coords = torch.randn(20, 3)
    refs   = coords.unsqueeze(0)
    assert _min_rmsd_kabsch(coords, refs) < 1e-4


def test_min_rmsd_kabsch_shifted():
    """Min RMSD after translation must still be near 0 (Kabsch is translation-invariant)."""
    coords = torch.randn(20, 3)
    shifted = coords + torch.tensor([5.0, 3.0, -2.0])
    refs = shifted.unsqueeze(0)
    assert _min_rmsd_kabsch(coords, refs) < 1e-4
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_active_loop.py::test_pdb_to_shard tests/test_active_loop.py::test_geometry_pass_rate_good -x 2>&1 | tail -5
```

Expected: ImportError — `lsmd.active_loop` does not exist.

- [ ] **Step 3: Create lsmd/active_loop.py with bootstrap helpers**

```python
"""Active learning loop utilities for single-protein conformational exploration.

Provides:
  _pdb_to_shard      — load a static PDB into a 1-frame shard dict
  _geometry_pass_rate — fraction of proposals with good bonds and no clashes
  _min_rmsd_kabsch   — minimum Kabsch-aligned RMSD from one structure to many
  bootstrap_check    — decide zero-MD or short-MD starting shard
  shard_from_md_runs — extract (R, t) Cα frames from completed OpenMM MD runs
  build_replay_shard — combine new frames with replay buffer for fine-tuning
  check_convergence  — budget / coverage / fes stopping criterion
"""
import json
import os
import shutil

import mdtraj as md
import numpy as np
import torch

from lsmd import data, geometry as g
from lsmd.cv_guidance import CVSpace
from lsmd.transfer_eval import load_checkpoint, rollout


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pdb_to_shard(pdb_path: str, dt_ps: float = 200.0) -> dict:
    """Load a static all-atom PDB and return a 1-frame shard dict.

    Uses mdtraj to extract protein backbone (N, CA, C) atoms, computes SE(3)
    rotation matrices via geometry.build_frames(), and assembles the full shard
    dict consumed by train_transfer.py and explore_conformations.py.

    Args:
        pdb_path: path to heavy-atom PDB (crystal structure or AlphaFold).
        dt_ps:    nominal frame spacing in ps (200 ps default).

    Returns:
        dict with keys:
            R         [1, N, 3, 3] float32 — per-residue rotation matrices
            t         [1, N, 3]    float32 — Cα positions in Å
            res_type  [N] long     — residue type indices (0-based)
            chain_id  [N] long     — chain indices (0-based)
            res_index [N] long     — sequential residue index
            n_res     int
            dt        float        — ps per frame
            seq       list[str]    — residue 3-letter names
    """
    traj = md.load(pdb_path)
    top  = traj.topology

    # Collect backbone atoms in residue order; skip non-protein residues
    residues = [r for r in top.residues if r.is_protein]
    n_idx, ca_idx, c_idx = [], [], []
    res_names, chain_ids = [], []
    for r in residues:
        atoms = {a.name: a.index for a in r.atoms}
        if not all(k in atoms for k in ("N", "CA", "C")):
            continue
        n_idx.append(atoms["N"])
        ca_idx.append(atoms["CA"])
        c_idx.append(atoms["C"])
        res_names.append(r.name)
        chain_ids.append(r.chain.index)

    xyz = torch.tensor(traj.xyz, dtype=torch.float32) * 10.0  # nm → Å  [1, n_atoms, 3]
    N_pos  = xyz[:, n_idx,  :]   # [1, N, 3]
    CA_pos = xyz[:, ca_idx, :]
    C_pos  = xyz[:, c_idx,  :]

    R, t = g.build_frames(N_pos, CA_pos, C_pos)  # [1, N, 3, 3], [1, N, 3]

    uniq     = sorted(set(res_names))
    type_map = {nm: i for i, nm in enumerate(uniq)}
    res_type  = torch.tensor([type_map[nm] for nm in res_names], dtype=torch.long)
    chain_id  = torch.tensor(chain_ids, dtype=torch.long)
    res_index = torch.arange(len(res_names), dtype=torch.long)

    return {
        "R": R, "t": t,
        "res_type": res_type,
        "chain_id": chain_id,
        "res_index": res_index,
        "n_res": len(res_names),
        "dt": float(dt_ps),
        "seq": res_names,
    }


def _geometry_pass_rate(proposals: list, ref_bond_A: float,
                        bond_tol: float = 0.15, clash_dist: float = 3.5) -> float:
    """Fraction of [N,3] Cα proposals passing bond-length and clash checks.

    Bond check: every adjacent Cα–Cα bond within ±bond_tol Å of ref_bond_A.
    Clash check: all non-adjacent (|i-j|>1) Cα–Cα distances > clash_dist Å.

    Args:
        proposals:    list of [N, 3] float tensors in Å.
        ref_bond_A:   reference Cα–Cα bond length in Å.
        bond_tol:     maximum allowed deviation from ref_bond_A (default 0.15 Å).
        clash_dist:   minimum allowed non-adjacent Cα distance (default 3.5 Å).

    Returns:
        float in [0.0, 1.0] — fraction that pass both checks.
    """
    if not proposals:
        return 0.0
    n_pass = 0
    for ca in proposals:
        ca = ca.float()
        # Bond check
        bonds = (ca[1:] - ca[:-1]).norm(dim=-1)  # [N-1]
        if (bonds - ref_bond_A).abs().max().item() > bond_tol:
            continue
        # Clash check (pairwise, non-adjacent)
        N = ca.shape[0]
        if N > 1:
            diff = ca.unsqueeze(0) - ca.unsqueeze(1)  # [N, N, 3]
            dists = diff.norm(dim=-1)                   # [N, N]
            mask = torch.ones(N, N, dtype=torch.bool)
            mask.fill_diagonal_(False)
            for k in range(-1, 2):
                if k != 0:
                    idx = torch.arange(max(0, -k), min(N, N - k))
                    mask[idx, idx + k] = False
            min_noadj = dists[mask].min().item() if mask.any() else float("inf")
            if min_noadj < clash_dist:
                continue
        n_pass += 1
    return n_pass / len(proposals)


def _min_rmsd_kabsch(query: torch.Tensor, refs: torch.Tensor) -> float:
    """Minimum Cα RMSD from query [N,3] to any frame in refs [F,N,3] via Kabsch.

    Vectorised over F — fast for up to tens of thousands of reference frames.

    Returns:
        Minimum RMSD in Å (0 if refs is empty).
    """
    if refs.shape[0] == 0:
        return 0.0
    q = query.float()                                     # [N, 3]
    r = refs.float()                                      # [F, N, 3]
    q_c = q - q.mean(0, keepdim=True)                    # center query
    r_c = r - r.mean(1, keepdim=True)                    # center each ref [F, N, 3]

    H   = torch.einsum("ni,fnj->fij", q_c, r_c)          # [F, 3, 3]
    U, _, Vt = torch.linalg.svd(H)
    d   = torch.linalg.det(Vt.mT @ U.mT)                 # [F] — reflection sign
    D   = torch.eye(3, dtype=q.dtype).unsqueeze(0).expand(H.shape[0], -1, -1).clone()
    D[:, 2, 2] = d
    R_opt  = Vt.mT @ D @ U.mT                            # [F, 3, 3]
    q_rot  = torch.einsum("fij,nj->fni", R_opt, q_c)     # [F, N, 3]
    rmsds  = (q_rot - r_c).pow(2).sum(-1).mean(-1).sqrt() # [F]
    return float(rmsds.min().item())


# ---------------------------------------------------------------------------
# bootstrap_check
# ---------------------------------------------------------------------------

def bootstrap_check(pdb_path: str, checkpoint: str, device: str,
                    bootstrap_ns: float, out_dir: str) -> dict:
    """Decide whether to start from the static PDB or run short bootstrap MD.

    Runs 20 DDIM proposals from the universal model. If geometry pass rate
    ≥ 80 %, returns a 1-frame shard from the PDB. Otherwise runs
    `bootstrap_ns` ns of OpenMM MD and returns a multi-frame shard.

    Args:
        pdb_path:     path to input heavy-atom PDB.
        checkpoint:   path to universal pretrained checkpoint (.pt).
        device:       "cuda" or "cpu".
        bootstrap_ns: MD length if bootstrap is needed (nanoseconds).
        out_dir:      directory for bootstrap MD outputs (created if needed).

    Returns:
        shard dict with keys {R, t, res_type, chain_id, res_index, n_res, dt, seq}.
    """
    from lsmd.md_validation import run_md

    shard_1f = _pdb_to_shard(pdb_path)
    N = shard_1f["n_res"]

    # Load model
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net, schedule, update_norm = load_checkpoint(ckpt, device)

    # Reference bond length from the single input frame
    ca = shard_1f["t"][0]  # [N, 3]
    ref_bond = (ca[1:] - ca[:-1]).norm(dim=-1).mean().item()

    # Generate 20 proposals (no CV guidance)
    R0 = shard_1f["R"][0].to(device)
    t0 = shard_1f["t"][0].to(device)
    proposals = []
    for _ in range(20):
        traj = rollout(
            net, schedule, update_norm, R0, t0,
            shard_1f["res_type"].to(device),
            shard_1f["chain_id"].to(device),
            shard_1f["res_index"].to(device),
            steps=50, tau_ps=2000, k=12,
            diff_steps=20, eta=1.0, temp_K=375.0,
            device=device,
        )
        proposals.append(traj[-1].cpu())

    pass_rate = _geometry_pass_rate(proposals, ref_bond_A=ref_bond)
    print(f"[bootstrap_check] geometry pass rate: {pass_rate:.1%}", flush=True)

    if pass_rate >= 0.80:
        print("[bootstrap_check] zero-MD path: universal model sufficient", flush=True)
        return shard_1f

    # Run bootstrap MD
    print(f"[bootstrap_check] pass rate < 80%; running {bootstrap_ns} ns bootstrap MD",
          flush=True)
    os.makedirs(out_dir, exist_ok=True)
    result = run_md(pdb_path, out_dir, md_ns=bootstrap_ns, temp_K=310.0)
    if result.get("error"):
        print(f"[bootstrap_check] bootstrap MD failed: {result['error']}; "
              "falling back to 1-frame shard", flush=True)
        return shard_1f

    # Load bootstrap trajectory → multi-frame shard
    traj_path = os.path.join(out_dir, "trajectory.dcd")
    top_path  = os.path.join(out_dir, "topology.pdb")
    frames    = data.load_frames(traj_path, top_path)
    # data.load_frames() returns res_names but not seq/n_res; add from PDB shard
    return {
        **frames,
        "dt":    200.0,
        "seq":   shard_1f["seq"],
        "n_res": len(frames["res_type"]),
    }
```

- [ ] **Step 4: Run the bootstrap helper tests**

```bash
pytest tests/test_active_loop.py::test_pdb_to_shard \
       tests/test_active_loop.py::test_geometry_pass_rate_good \
       tests/test_active_loop.py::test_geometry_pass_rate_bad \
       tests/test_active_loop.py::test_min_rmsd_kabsch_identical \
       tests/test_active_loop.py::test_min_rmsd_kabsch_shifted \
       -xvs
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add lsmd/active_loop.py tests/test_active_loop.py
git commit -m "feat: active_loop.py — _pdb_to_shard, _geometry_pass_rate, _min_rmsd_kabsch, bootstrap_check"
```

---

## Task 3: lsmd/active_loop.py — Shard Builders

**Files:**
- Modify: `lsmd/active_loop.py` (append two functions)
- Modify: `tests/test_active_loop.py` (append four tests)

**Interfaces:**
- Consumes: `data.load_frames(traj_path, top_path)` → `{R [F,N,3,3], t [F,N,3], ...}`
- Produces:
  - `shard_from_md_runs(md_run_dirs, dt_ps=200) -> tuple[Tensor, Tensor]` — `(R [F,N,3,3], t [F,N,3])`
  - `build_replay_shard(new_R, new_t, accumulated_pt, protein_meta, replay_cap=5000, dt_ps=200) -> dict` — complete shard for `train_transfer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_active_loop.py`:

```python
from lsmd.active_loop import shard_from_md_runs, build_replay_shard


def _make_fake_md_run(tmp_dir, run_id, n_frames, n_res, error=None):
    """Create a fake md_run directory with metrics.json and stub trajectory."""
    run_dir = os.path.join(tmp_dir, f"run_{run_id:04d}")
    os.makedirs(run_dir, exist_ok=True)
    metrics = {"id": f"run_{run_id}", "md_ns": 10.0, "error": error}
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)
    # Write stub DCD via mdtraj (requires a topology)
    # Use a pre-built PDB for topology; store CA coords as proxy
    top_path = os.path.join(run_dir, "topology.pdb")
    _write_tiny_pdb(top_path, n_res=n_res)
    # Create fake trajectory: reuse topology PDB itself (1 frame)
    # For tests, we duplicate the frame n_frames times using mdtraj
    import mdtraj as md
    traj = md.load(top_path)
    coords = np.tile(traj.xyz, (n_frames, 1, 1))
    # Add small random displacements
    coords += np.random.randn(*coords.shape) * 0.01
    traj_out = md.Trajectory(coords, traj.topology)
    traj_out.save_dcd(os.path.join(run_dir, "trajectory.dcd"))
    return run_dir


def test_shard_from_md_runs_skips_failed(tmp_path):
    """shard_from_md_runs skips runs where metrics.json has error != null."""
    n_res = 5
    good_dir = _make_fake_md_run(str(tmp_path), 0, n_frames=10, n_res=n_res)
    bad_dir  = _make_fake_md_run(str(tmp_path), 1, n_frames=10, n_res=n_res, error="OOM")
    R, t = shard_from_md_runs([good_dir, bad_dir], dt_ps=1)
    assert t.shape[1] == n_res
    assert t.shape[0] > 0
    # bad run excluded: total frames come only from good run
    assert t.shape[0] <= 10 + 1  # allow some rounding in stride


def test_shard_from_md_runs_empty(tmp_path):
    """shard_from_md_runs returns empty tensors when all runs failed."""
    bad_dir = _make_fake_md_run(str(tmp_path), 0, n_frames=5, n_res=5, error="crash")
    R, t = shard_from_md_runs([bad_dir])
    assert t.shape[0] == 0


def test_build_replay_shard_capped(tmp_path):
    """build_replay_shard never returns more than replay_cap frames."""
    N = 5
    accumulated_pt = str(tmp_path / "acc.pt")
    protein_meta = {
        "res_type": torch.zeros(N, dtype=torch.long),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "seq": ["ALA"] * N,
        "n_res": N,
    }
    # Pre-fill history with 200 frames
    big_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(200, N, -1, -1).clone()
    big_t = torch.randn(200, N, 3)
    torch.save({"R": big_R, "t": big_t}, accumulated_pt)

    new_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(30, N, -1, -1).clone()
    new_t = torch.randn(30, N, 3)
    shard = build_replay_shard(new_R, new_t, accumulated_pt, protein_meta, replay_cap=50)
    assert len(shard["t"]) == 50


def test_build_replay_shard_small_history(tmp_path):
    """build_replay_shard uses all history when history < replay_cap - new."""
    N = 5
    accumulated_pt = str(tmp_path / "acc.pt")
    protein_meta = {
        "res_type": torch.zeros(N, dtype=torch.long),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "seq": ["ALA"] * N,
        "n_res": N,
    }
    # Pre-fill history with 10 frames
    hist_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(10, N, -1, -1).clone()
    hist_t = torch.randn(10, N, 3)
    torch.save({"R": hist_R, "t": hist_t}, accumulated_pt)

    new_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(5, N, -1, -1).clone()
    new_t = torch.randn(5, N, 3)
    shard = build_replay_shard(new_R, new_t, accumulated_pt, protein_meta, replay_cap=5000)
    assert len(shard["t"]) == 15  # 5 new + 10 all history

    # accumulated_pt must now have 10 + 5 = 15 frames
    acc = torch.load(accumulated_pt, weights_only=False)
    assert acc["t"].shape[0] == 15
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_active_loop.py::test_shard_from_md_runs_skips_failed \
       tests/test_active_loop.py::test_build_replay_shard_capped -x 2>&1 | tail -5
```

Expected: ImportError — functions not yet defined.

- [ ] **Step 3: Append shard_from_md_runs and build_replay_shard to lsmd/active_loop.py**

```python
# ---------------------------------------------------------------------------
# shard_from_md_runs
# ---------------------------------------------------------------------------

def shard_from_md_runs(md_run_dirs: list, dt_ps: float = 200.0):
    """Extract Cα backbone frames from completed OpenMM MD run directories.

    For each run directory that has a successful `metrics.json` (error == null)
    and valid `trajectory.dcd` + `topology.pdb`, loads the full-atom trajectory,
    extracts backbone SE(3) frames with `data.load_frames()`, and strides to
    approximately dt_ps ps between frames.

    Args:
        md_run_dirs: list of run directory paths (order does not matter).
        dt_ps:       desired frame spacing in ps (default 200 ps).

    Returns:
        (R, t) where R is [F_total, N, 3, 3] and t is [F_total, N, 3] float32.
        Returns (empty, empty) tensors if all runs failed or no directories given.
    """
    all_R, all_t = [], []

    for run_dir in sorted(md_run_dirs):
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as fh:
            m = json.load(fh)
        if m.get("error") is not None:
            continue

        traj_path = os.path.join(run_dir, "trajectory.dcd")
        top_path  = os.path.join(run_dir, "topology.pdb")
        if not (os.path.exists(traj_path) and os.path.exists(top_path)):
            continue

        try:
            # Get frame spacing from the DCD header via mdtraj
            traj_info = md.load(traj_path, top=top_path)
            dt_traj_ps = float(traj_info.timestep)   # ps per frame
            stride = max(1, round(dt_ps / dt_traj_ps))

            frames = data.load_frames(traj_path, top_path)
            R_run = frames["R"][::stride]   # [F_run, N, 3, 3]
            t_run = frames["t"][::stride]   # [F_run, N, 3]
            all_R.append(R_run)
            all_t.append(t_run)
        except Exception as exc:
            print(f"[shard_from_md_runs] skipping {run_dir}: {exc}", flush=True)
            continue

    if not all_R:
        return torch.empty(0), torch.empty(0)

    return torch.cat(all_R, dim=0), torch.cat(all_t, dim=0)


# ---------------------------------------------------------------------------
# build_replay_shard
# ---------------------------------------------------------------------------

def build_replay_shard(new_R: torch.Tensor, new_t: torch.Tensor,
                       accumulated_pt: str, protein_meta: dict,
                       replay_cap: int = 5000, dt_ps: float = 200.0) -> dict:
    """Build a fine-tuning shard from new frames + replay of historical frames.

    Appends new_R / new_t to accumulated_pt (the growing history store), then
    returns a shard dict whose `t` and `R` are:
        all new frames  +  random_sample(history_before_this_round, n_old)
    where n_old = min(replay_cap − len(new_frames), len(history)).

    Args:
        new_R:          [F_new, N, 3, 3] rotation matrices from this round's MD.
        new_t:          [F_new, N, 3] Cα positions from this round's MD.
        accumulated_pt: path to accumulated_frames.pt (appended in-place).
        protein_meta:   dict with {res_type, chain_id, res_index, seq, n_res}.
        replay_cap:     maximum total frames in returned shard (default 5000).
        dt_ps:          frame spacing label for the shard (default 200 ps).

    Returns:
        shard dict with {res_type, chain_id, res_index, seq, n_res, R, t, dt}.
    """
    # Load existing history (frames accumulated before this round)
    if os.path.exists(accumulated_pt):
        acc = torch.load(accumulated_pt, map_location="cpu", weights_only=False)
        hist_R = acc["R"]   # [F_hist, N, 3, 3]
        hist_t = acc["t"]   # [F_hist, N, 3]
    else:
        N = new_t.shape[1]
        hist_R = torch.empty(0, N, 3, 3)
        hist_t = torch.empty(0, N, 3)

    # Append new frames to accumulated store
    updated_R = torch.cat([hist_R, new_R], dim=0)
    updated_t = torch.cat([hist_t, new_t], dim=0)
    torch.save({"R": updated_R, "t": updated_t}, accumulated_pt)

    # Build replay buffer: all new + sample of old history
    n_old = min(max(0, replay_cap - len(new_t)), len(hist_t))
    if n_old > 0 and len(hist_t) > 0:
        idx = torch.randperm(len(hist_t))[:n_old]
        combined_R = torch.cat([new_R, hist_R[idx]], dim=0)
        combined_t = torch.cat([new_t, hist_t[idx]], dim=0)
    else:
        combined_R = new_R
        combined_t = new_t

    return {
        **protein_meta,
        "R": combined_R,
        "t": combined_t,
        "dt": float(dt_ps),
    }
```

- [ ] **Step 4: Run shard builder tests**

```bash
pytest tests/test_active_loop.py::test_shard_from_md_runs_skips_failed \
       tests/test_active_loop.py::test_shard_from_md_runs_empty \
       tests/test_active_loop.py::test_build_replay_shard_capped \
       tests/test_active_loop.py::test_build_replay_shard_small_history \
       -xvs
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add lsmd/active_loop.py tests/test_active_loop.py
git commit -m "feat: active_loop.py — shard_from_md_runs, build_replay_shard"
```

---

## Task 4: lsmd/active_loop.py — Convergence Checkers

**Files:**
- Modify: `lsmd/active_loop.py` (append `check_convergence` and `_check_fes`)
- Modify: `tests/test_active_loop.py` (append four tests)

**Interfaces:**
- Produces: `check_convergence(criterion, threshold, state) -> tuple[bool, float]`
  - `state` keys: `total_md_ns` (float), `last_novel_fraction` (float), `accumulated_t` (Tensor [F,N,3] | None), `prev_accumulated_t` (Tensor | None), `round` (int)
  - Returns `(converged: bool, metric: float)` — metric is logged every round.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_active_loop.py`:

```python
from lsmd.active_loop import check_convergence


def test_convergence_budget_not_reached():
    state = {"total_md_ns": 45.0, "last_novel_fraction": 0.5, "round": 1}
    converged, metric = check_convergence("budget", 100.0, state)
    assert not converged
    assert abs(metric - 45.0) < 1e-6


def test_convergence_budget_reached():
    state = {"total_md_ns": 100.0, "last_novel_fraction": 0.5, "round": 1}
    converged, metric = check_convergence("budget", 100.0, state)
    assert converged


def test_convergence_coverage_not_converged():
    state = {"total_md_ns": 10.0, "last_novel_fraction": 0.30, "round": 2}
    converged, metric = check_convergence("coverage", 0.10, state)
    assert not converged
    assert abs(metric - 0.30) < 1e-6


def test_convergence_coverage_converged():
    state = {"total_md_ns": 10.0, "last_novel_fraction": 0.05, "round": 3}
    converged, metric = check_convergence("coverage", 0.10, state)
    assert converged


def test_convergence_fes_insufficient_data():
    """FES criterion returns (False, nan) when < 50 frames or round < 2."""
    N = 10
    state = {
        "total_md_ns": 10.0, "last_novel_fraction": 0.5,
        "round": 0,
        "accumulated_t":      torch.randn(20, N, 3),
        "prev_accumulated_t": torch.randn(10, N, 3),
    }
    converged, metric = check_convergence("fes", 0.05, state)
    assert not converged
    assert metric != metric  # nan


def test_convergence_fes_converged():
    """FES converges when JS divergence < threshold."""
    N = 10
    # Same frames for current and prev → JS ≈ 0
    frames = torch.randn(80, N, 3)
    state = {
        "total_md_ns": 10.0, "last_novel_fraction": 0.2,
        "round": 3,
        "accumulated_t":      frames,
        "prev_accumulated_t": frames + torch.randn_like(frames) * 0.001,
    }
    converged, metric = check_convergence("fes", 0.05, state)
    assert converged
    assert metric < 0.05
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_active_loop.py::test_convergence_budget_not_reached -x 2>&1 | tail -5
```

Expected: ImportError — `check_convergence` not defined.

- [ ] **Step 3: Append check_convergence to lsmd/active_loop.py**

```python
# ---------------------------------------------------------------------------
# Convergence checkers
# ---------------------------------------------------------------------------

def check_convergence(criterion: str, threshold: float, state: dict):
    """Check whether the active learning loop has converged.

    Args:
        criterion: "budget" | "coverage" | "fes"
        threshold: criterion-specific stopping value.
        state: dict with keys:
            total_md_ns        (float)   — cumulative MD nanoseconds run so far
            last_novel_fraction (float)  — fraction of proposals that were novel
                                           in the most recent round
            round              (int)     — current round number (0-indexed)
            accumulated_t      (Tensor|None) — [F_curr, N, 3] current accumulated Cα
            prev_accumulated_t (Tensor|None) — [F_prev, N, 3] previous round's Cα

    Returns:
        (converged: bool, metric: float)
        metric is always returned (nan when not yet computable) for logging.
    """
    if criterion == "budget":
        val = float(state["total_md_ns"])
        return val >= threshold, val

    elif criterion == "coverage":
        val = float(state.get("last_novel_fraction", 1.0))
        return val < threshold, val

    elif criterion == "fes":
        return _check_fes(state, threshold)

    else:
        raise ValueError(f"Unknown convergence criterion: {criterion!r}. "
                         "Choose 'budget', 'coverage', or 'fes'.")


def _check_fes(state: dict, threshold: float):
    """JS divergence between current and previous FES histograms.

    Requires: round >= 2 AND accumulated_t has >= 50 frames.
    Otherwise returns (False, nan).
    """
    from scipy.spatial.distance import jensenshannon

    curr = state.get("accumulated_t")
    prev = state.get("prev_accumulated_t")

    if (curr is None or prev is None
            or len(curr) < 50 or state.get("round", 0) < 2):
        return False, float("nan")

    # Fit PCA on current accumulated frames
    F, N, _ = curr.shape
    X = curr.reshape(F, N * 3).float()
    mean = X.mean(0)
    _, _, Vh = torch.linalg.svd(X - mean, full_matrices=False)
    comps = Vh[:2].numpy()            # [2, 3N]
    mean_np = mean.numpy()

    def _project(frames):
        Fp = frames.shape[0]
        Xp = frames.reshape(Fp, N * 3).numpy().astype(np.float32)
        return (Xp - mean_np) @ comps.T  # [Fp, 2]

    proj_curr = _project(curr)   # [F_curr, 2]
    proj_prev = _project(prev)   # [F_prev, 2]

    # Common histogram range
    all_pts = np.vstack([proj_curr, proj_prev])
    r = [[all_pts[:, 0].min(), all_pts[:, 0].max()],
         [all_pts[:, 1].min(), all_pts[:, 1].max()]]
    bins = 50
    h_c, _, _ = np.histogram2d(proj_curr[:, 0], proj_curr[:, 1],
                                bins=bins, range=r, density=True)
    h_p, _, _ = np.histogram2d(proj_prev[:, 0], proj_prev[:, 1],
                                bins=bins, range=r, density=True)

    # Normalise to probability vectors, add epsilon to avoid log(0)
    eps = 1e-10
    p = (h_c.flatten() + eps); p /= p.sum()
    q = (h_p.flatten() + eps); q /= q.sum()

    js = float(jensenshannon(p, q))
    return js < threshold, js
```

- [ ] **Step 4: Run convergence tests**

```bash
pytest tests/test_active_loop.py::test_convergence_budget_not_reached \
       tests/test_active_loop.py::test_convergence_budget_reached \
       tests/test_active_loop.py::test_convergence_coverage_not_converged \
       tests/test_active_loop.py::test_convergence_coverage_converged \
       tests/test_active_loop.py::test_convergence_fes_insufficient_data \
       tests/test_active_loop.py::test_convergence_fes_converged \
       -xvs
```

Expected: all PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -x -q 2>&1 | tail -10
```

Expected: all pass (no regressions in existing tests).

- [ ] **Step 6: Commit**

```bash
git add lsmd/active_loop.py tests/test_active_loop.py
git commit -m "feat: active_loop.py — check_convergence (budget / coverage / fes)"
```

---

## Task 5: scripts/active_learning.py — Orchestrator

**Files:**
- Create: `scripts/active_learning.py`
- Modify: `tests/test_active_loop.py` (append two integration tests)

**Interfaces:**
- Consumes: all functions from `lsmd/active_loop.py`; `run_md()` from `lsmd/md_validation`; `load_checkpoint`, `rollout` from `lsmd/transfer_eval`; `AllAtomReconstructor` from `lsmd/reconstruct`; `write_ca_pdb` from `lsmd/decoder`; `CVSpace`, `build_cv_guidance` from `lsmd/cv_guidance`; `train_transfer.py` via subprocess.

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_active_loop.py`:

```python
import subprocess, sys
from pathlib import Path


def test_active_learning_help():
    """scripts/active_learning.py --help must exit 0."""
    result = subprocess.run(
        [sys.executable, "scripts/active_learning.py", "--help"],
        capture_output=True, cwd="/home/qshao/DL-MD"
    )
    assert result.returncode == 0
    assert b"--pdb" in result.stdout


def test_active_learning_resume_skips_done(tmp_path, monkeypatch):
    """Orchestrator skips rounds with .done stamps without re-running them."""
    # Create a round_0 with .done stamp and pre-populated summary
    round0 = tmp_path / "round_0"
    round0.mkdir()
    summary = {
        "round": 0, "n_proposals_generated": 10, "n_novel_filtered": 5,
        "n_md_attempted": 5, "n_md_success": 4,
        "new_frames_this_round": 40, "total_frames_accumulated": 40,
        "total_md_ns": 40.0, "last_novel_fraction": 0.5,
        "fes_js": None, "converged": False,
        "stop_criterion": "budget", "stop_threshold": 1000.0,
    }
    with open(round0 / "round_summary.json", "w") as fh:
        json.dump(summary, fh)
    (round0 / ".done").touch()

    # Import and call _load_completed_rounds to verify resume logic
    from scripts.active_learning import _load_completed_rounds
    completed = _load_completed_rounds(str(tmp_path))
    assert 0 in completed
    assert completed[0]["total_md_ns"] == 40.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_active_loop.py::test_active_learning_help -x 2>&1 | tail -5
```

Expected: FAIL — `scripts/active_learning.py` does not exist.

- [ ] **Step 3: Create scripts/active_learning.py**

```python
"""Active learning loop for single-protein conformational exploration.

Usage
-----
python scripts/active_learning.py \\
    --pdb             input.pdb                    \\
    --checkpoint      checkpoints/v2_256h_90k.pt   \\
    --out             my_protein_loop              \\
    --rounds          10                           \\
    --proposals       100                          \\
    --batch-size      20                           \\
    --md-ns           10                           \\
    --replay-cap      5000                         \\
    --novel-threshold 1.5                          \\
    --stop            coverage                     \\
    --stop-threshold  0.10                         \\
    --bootstrap-ns    10                           \\
    --fine-tune-steps 2000                         \\
    --n-parallel      4                            \\
    --device          cuda
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from lsmd.active_loop import (
    _pdb_to_shard, _min_rmsd_kabsch, bootstrap_check,
    shard_from_md_runs, build_replay_shard, check_convergence,
)
from lsmd.cv_guidance import CVSpace
from lsmd.decoder import write_ca_pdb
from lsmd.md_validation import run_md
from lsmd.reconstruct import AllAtomReconstructor
from lsmd.transfer_eval import load_checkpoint, rollout


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_completed_rounds(out_dir: str) -> dict:
    """Return {round_num: summary_dict} for all .done-stamped rounds."""
    completed = {}
    for entry in sorted(Path(out_dir).iterdir()):
        if not entry.name.startswith("round_"):
            continue
        done = entry / ".done"
        summary_path = entry / "round_summary.json"
        if done.exists() and summary_path.exists():
            with open(summary_path) as fh:
                completed[int(entry.name.split("_")[1])] = json.load(fh)
    return completed


# ---------------------------------------------------------------------------
# Novel filtering
# ---------------------------------------------------------------------------

def _filter_novel(proposals: list, accumulated_t: torch.Tensor,
                  novel_threshold: float):
    """Return (novel_list, min_rmsds_all) where novel proposals have min-RMSD > threshold.

    Args:
        proposals:        list of [N, 3] Cα tensors (from rollout).
        accumulated_t:    [F, N, 3] all accumulated Cα frames.
        novel_threshold:  Å — proposals with min-RMSD > this are novel.

    Returns:
        (novel: list[Tensor], min_rmsds: list[float])
    """
    novel, min_rmsds = [], []
    for prop in proposals:
        mr = _min_rmsd_kabsch(prop, accumulated_t)
        min_rmsds.append(mr)
        if mr > novel_threshold:
            novel.append(prop)
    return novel, min_rmsds


# ---------------------------------------------------------------------------
# Main round loop
# ---------------------------------------------------------------------------

def run_round(round_num: int, args, current_ckpt: str, protein_meta: dict,
              shard_1f: dict, accumulated_pt: str, prev_total_md_ns: float,
              prev_novel_fraction: float, prev_accumulated_t,
              loop_summary: list):
    """Execute one active learning round; return updated state or None if converged."""
    round_dir  = os.path.join(args.out, f"round_{round_num}")
    done_stamp = os.path.join(round_dir, ".done")
    os.makedirs(round_dir, exist_ok=True)

    # ── 1. Load current accumulated Cα frames ────────────────────────────────
    if os.path.exists(accumulated_pt):
        acc = torch.load(accumulated_pt, map_location="cpu", weights_only=False)
        accumulated_t = acc["t"]   # [F_acc, N, 3]
    else:
        accumulated_t = shard_1f["t"]  # F=1 from input PDB

    # ── 2. Build / update CV space ──────────────────────────────────────────
    cv_space = CVSpace(n_pc=5)
    cv_space.fit(accumulated_t)
    cv_basis_path = os.path.join(round_dir, "cv_basis.pt")
    cv_space.save(cv_basis_path)

    # ── 3. Load model for this round ─────────────────────────────────────────
    ckpt = torch.load(current_ckpt, map_location="cpu", weights_only=False)
    net, schedule, update_norm = load_checkpoint(ckpt, args.device)

    # Pre-fill CV buffer from accumulated frames (up to 500 entries)
    cv_buffer = []
    F_acc = accumulated_t.shape[0]
    for i in range(min(500, F_acc)):
        cv_buffer.append(cv_space.project_single(accumulated_t[i]).detach())

    R0 = shard_1f["R"][0].to(args.device)
    t0 = shard_1f["t"][0].to(args.device)

    # ── 4. Generate proposals ────────────────────────────────────────────────
    proposals_dir = os.path.join(round_dir, "proposals")
    os.makedirs(proposals_dir, exist_ok=True)
    seq = shard_1f.get("seq", ["ALA"] * shard_1f["n_res"])

    proposals_ca = []
    for i in range(args.proposals):
        traj = rollout(
            net, schedule, update_norm, R0, t0,
            shard_1f["res_type"].to(args.device),
            shard_1f["chain_id"].to(args.device),
            shard_1f["res_index"].to(args.device),
            steps=50, tau_ps=2000, k=12,
            diff_steps=20, eta=1.0, temp_K=375.0,
            cv_space=cv_space if len(cv_buffer) >= 50 else None,
            cv_buffer=cv_buffer,
            k_guide=0.05, sigma_cv=1.0, guide_warmup=50,
            device=args.device,
        )
        x_final = traj[-1].cpu()
        proposals_ca.append(x_final)
        cv_buffer.append(cv_space.project_single(x_final).detach())

        pdb_path = os.path.join(proposals_dir, f"prop_{i:04d}.pdb")
        write_ca_pdb(x_final, seq, pdb_path)

    # ── 5. Filter novel proposals ─────────────────────────────────────────────
    novel, min_rmsds = _filter_novel(proposals_ca, accumulated_t, args.novel_threshold)
    n_novel = len(novel)

    if n_novel == 0:
        print(f"[round {round_num}] No novel proposals — landscape exhausted; terminating.",
              flush=True)
        _write_summary(round_dir, round_num, args, proposals_ca, novel,
                       md_success=0, new_frames=0, total_md_ns=prev_total_md_ns,
                       novel_fraction=0.0, fes_js=float("nan"), converged=True,
                       prev_accumulated_t=prev_accumulated_t, accumulated_t=accumulated_t)
        Path(done_stamp).touch()
        return None

    # Random selection from novel candidates
    batch_size = min(args.batch_size, n_novel)
    selected_ca = random.sample(novel, batch_size)
    print(f"[round {round_num}] generated={args.proposals} novel={n_novel} selected={batch_size}",
          flush=True)

    # ── 6. Reconstruct all-atom structures ──────────────────────────────────
    allatom_dir = os.path.join(round_dir, "allatom")
    os.makedirs(allatom_dir, exist_ok=True)
    rec = AllAtomReconstructor(args.pdb, args.pdb)  # use input PDB as template
    allatom_pdbs = []
    for j, ca_struct in enumerate(selected_ca):
        xyz = rec.reconstruct_frame_ca(ca_struct)   # numpy [N_heavy, 3]
        import mdtraj as md
        traj_tmp = md.load(args.pdb)
        top_tmp  = traj_tmp.topology
        ha_idx   = top_tmp.select("protein and not type H")
        xyz_nm   = xyz / 10.0                        # Å → nm
        t_out = md.Trajectory(xyz_nm[None], traj_tmp.atom_slice(ha_idx).topology)
        out_pdb = os.path.join(allatom_dir, f"struct_{j:04d}.pdb")
        t_out.save_pdb(out_pdb)
        allatom_pdbs.append(out_pdb)

    # ── 7. Run MD validation (parallel) ──────────────────────────────────────
    md_runs_dir = os.path.join(round_dir, "md_runs")
    os.makedirs(md_runs_dir, exist_ok=True)
    md_run_dirs = []
    def _run_one(j_pdb):
        j, pdb = j_pdb
        run_dir_j = os.path.join(md_runs_dir, f"struct_{j:04d}")
        run_md(pdb, run_dir_j, md_ns=args.md_ns, temp_K=310.0)
        return run_dir_j

    with ThreadPoolExecutor(max_workers=args.n_parallel) as pool:
        md_run_dirs = list(pool.map(_run_one, enumerate(allatom_pdbs)))

    n_md_success = sum(
        1 for d in md_run_dirs
        if os.path.exists(os.path.join(d, "metrics.json")) and
           json.load(open(os.path.join(d, "metrics.json"))).get("error") is None
    )
    print(f"[round {round_num}] MD success: {n_md_success}/{batch_size}", flush=True)

    # ── 8. Extract frames and build replay shard ───────────────────────────
    new_R, new_t = shard_from_md_runs(md_run_dirs, dt_ps=200)
    new_frames = len(new_t) if new_t.shape[0] > 0 else 0

    if new_frames > 0:
        replay_shard = build_replay_shard(
            new_R, new_t, accumulated_pt, protein_meta,
            replay_cap=args.replay_cap, dt_ps=200.0
        )
        replay_shard_path = os.path.join(round_dir, "replay_shard.pt")
        torch.save(replay_shard, replay_shard_path)

        # ── 9. Fine-tune model ────────────────────────────────────────────
        next_ckpt = os.path.join(round_dir, "checkpoint.pt")
        subprocess.run([
            sys.executable, "scripts/train_transfer.py",
            "--shard",   replay_shard_path,
            "--resume",  args.checkpoint,   # always from universal base
            "--steps",   str(args.fine_tune_steps),
            "--lr",      "1e-4",
            "--hidden",  "256",
            "--layers",  "6",
            "--lags_ps", "200", "1000", "5000",
            "--time_reversal",
            "--device",  args.device,
            "--out",     next_ckpt,
        ], check=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    else:
        # No new frames: skip fine-tuning, carry forward previous checkpoint
        next_ckpt = current_ckpt
        replay_shard_path = None
        print(f"[round {round_num}] WARNING: no MD frames extracted; skipping fine-tune",
              flush=True)

    # ── 10. Check stopping criterion ────────────────────────────────────────
    total_md_ns = prev_total_md_ns + n_md_success * args.md_ns
    novel_fraction = n_novel / len(proposals_ca)

    # Load updated accumulated_t for FES criterion
    if os.path.exists(accumulated_pt):
        acc_now = torch.load(accumulated_pt, map_location="cpu", weights_only=False)["t"]
    else:
        acc_now = accumulated_t

    state = {
        "total_md_ns":         total_md_ns,
        "last_novel_fraction": novel_fraction,
        "accumulated_t":       acc_now,
        "prev_accumulated_t":  prev_accumulated_t,
        "round":               round_num,
    }
    converged, metric = check_convergence(args.stop, args.stop_threshold, state)

    # Determine metric label
    fes_js = metric if args.stop == "fes" else float("nan")

    _write_summary(round_dir, round_num, args, proposals_ca, novel,
                   md_success=n_md_success, new_frames=new_frames,
                   total_md_ns=total_md_ns,
                   novel_fraction=novel_fraction, fes_js=fes_js,
                   converged=converged, prev_accumulated_t=prev_accumulated_t,
                   accumulated_t=acc_now)
    Path(done_stamp).touch()

    return {
        "next_ckpt":          next_ckpt,
        "total_md_ns":        total_md_ns,
        "novel_fraction":     novel_fraction,
        "accumulated_t":      acc_now,
        "converged":          converged,
    }


def _write_summary(round_dir, round_num, args, all_proposals, novel_proposals,
                   md_success, new_frames, total_md_ns, novel_fraction, fes_js,
                   converged, prev_accumulated_t, accumulated_t):
    """Write round_summary.json and append to loop_summary.json."""
    n_acc_before = prev_accumulated_t.shape[0] if prev_accumulated_t is not None else 0
    summary = {
        "round":                   round_num,
        "n_proposals_generated":   len(all_proposals),
        "n_novel_filtered":        len(novel_proposals),
        "n_md_attempted":          min(args.batch_size, len(novel_proposals)),
        "n_md_success":            md_success,
        "new_frames_this_round":   new_frames,
        "total_frames_accumulated": n_acc_before + new_frames,
        "total_md_ns":             total_md_ns,
        "last_novel_fraction":     novel_fraction,
        "fes_js":                  None if fes_js != fes_js else fes_js,  # nan → None
        "converged":               converged,
        "stop_criterion":          args.stop,
        "stop_threshold":          args.stop_threshold,
    }
    with open(os.path.join(round_dir, "round_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    loop_path = os.path.join(args.out, "loop_summary.json")
    if os.path.exists(loop_path):
        with open(loop_path) as fh:
            loop_data = json.load(fh)
    else:
        loop_data = []
    loop_data.append(summary)
    with open(loop_path, "w") as fh:
        json.dump(loop_data, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Active learning loop for single-protein conformational exploration."
    )
    ap.add_argument("--pdb",             required=True,
                    help="Input heavy-atom PDB (crystal or AlphaFold structure).")
    ap.add_argument("--checkpoint",      required=True,
                    help="Universal pretrained checkpoint (.pt), e.g. v2_256h_90k.pt.")
    ap.add_argument("--out",             required=True,
                    help="Output directory (created if needed; resume-safe).")
    ap.add_argument("--rounds",          type=int,   default=10)
    ap.add_argument("--proposals",       type=int,   default=100,
                    help="DDIM proposals generated per round.")
    ap.add_argument("--batch-size",      type=int,   default=20,
                    help="Number of novel proposals to validate with MD per round.")
    ap.add_argument("--md-ns",           type=float, default=10.0,
                    help="MD validation length per structure (nanoseconds).")
    ap.add_argument("--replay-cap",      type=int,   default=5000,
                    help="Max frames in replay shard (controls fine-tuning cost).")
    ap.add_argument("--novel-threshold", type=float, default=1.5,
                    help="Min-RMSD (Å) to count a proposal as novel.")
    ap.add_argument("--stop",            choices=["budget", "coverage", "fes"],
                    default="coverage")
    ap.add_argument("--stop-threshold",  type=float, default=0.10,
                    help="Stopping value: ns (budget), fraction (coverage), JS (fes).")
    ap.add_argument("--bootstrap-ns",    type=float, default=10.0,
                    help="Bootstrap MD length (ns) if universal model geometry is poor.")
    ap.add_argument("--fine-tune-steps", type=int,   default=2000)
    ap.add_argument("--n-parallel",      type=int,   default=4,
                    help="Parallel MD worker threads.")
    ap.add_argument("--device",          default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Copy input PDB into output dir for provenance
    shutil.copy(args.pdb, os.path.join(args.out, "input.pdb"))

    accumulated_pt = os.path.join(args.out, "accumulated_frames.pt")

    # Resume: load already-completed rounds
    completed = _load_completed_rounds(args.out)
    loop_summary = [completed[r] for r in sorted(completed)]
    print(f"[active_learning] resuming from round {len(completed)} "
          f"(completed: {sorted(completed.keys())})", flush=True)

    # ── Round 0: bootstrap check ─────────────────────────────────────────────
    bootstrap_shard_path = os.path.join(args.out, "round_0", "bootstrap_shard.pt")
    protein_meta_path    = os.path.join(args.out, "protein_meta.pt")

    if 0 not in completed:
        os.makedirs(os.path.join(args.out, "round_0"), exist_ok=True)
        shard_1f = bootstrap_check(
            pdb_path=args.pdb,
            checkpoint=args.checkpoint,
            device=args.device,
            bootstrap_ns=args.bootstrap_ns,
            out_dir=os.path.join(args.out, "round_0", "bootstrap"),
        )
        torch.save(shard_1f, bootstrap_shard_path)
        protein_meta = {k: shard_1f[k]
                        for k in ("res_type", "chain_id", "res_index", "seq", "n_res")}
        torch.save(protein_meta, protein_meta_path)
    else:
        shard_1f     = torch.load(bootstrap_shard_path, map_location="cpu", weights_only=False)
        protein_meta = torch.load(protein_meta_path,    map_location="cpu", weights_only=False)

    # State carried between rounds
    if completed:
        last = completed[max(completed)]
        total_md_ns     = last["total_md_ns"]
        novel_fraction  = last["last_novel_fraction"]
    else:
        total_md_ns    = 0.0
        novel_fraction = 1.0

    # Previous accumulated_t for FES criterion
    if os.path.exists(accumulated_pt):
        prev_t = torch.load(accumulated_pt, map_location="cpu", weights_only=False)["t"]
    else:
        prev_t = None

    # ── Round loop ────────────────────────────────────────────────────────────
    for round_num in range(args.rounds):
        if round_num in completed:
            if completed[round_num]["converged"]:
                print(f"[active_learning] converged in round {round_num} (loaded from cache)",
                      flush=True)
                break
            continue  # already done, not converged

        # Determine current checkpoint
        if round_num == 0:
            current_ckpt = args.checkpoint
        else:
            prev_ckpt = os.path.join(args.out, f"round_{round_num - 1}", "checkpoint.pt")
            current_ckpt = prev_ckpt if os.path.exists(prev_ckpt) else args.checkpoint

        result = run_round(
            round_num=round_num,
            args=args,
            current_ckpt=current_ckpt,
            protein_meta=protein_meta,
            shard_1f=shard_1f,
            accumulated_pt=accumulated_pt,
            prev_total_md_ns=total_md_ns,
            prev_novel_fraction=novel_fraction,
            prev_accumulated_t=prev_t,
            loop_summary=loop_summary,
        )

        if result is None:
            print("[active_learning] early termination: no novel proposals.", flush=True)
            break

        total_md_ns    = result["total_md_ns"]
        novel_fraction = result["novel_fraction"]
        prev_t         = result["accumulated_t"]
        current_ckpt   = result["next_ckpt"]

        if result["converged"]:
            print(f"[active_learning] stopping criterion '{args.stop}' met at round {round_num}.",
                  flush=True)
            break

    # ── Symlinks to final outputs ─────────────────────────────────────────────
    final_ckpt  = os.path.join(args.out, "final_checkpoint.pt")
    final_shard = os.path.join(args.out, "final_shard.pt")

    # Find last round's checkpoint
    for r in range(args.rounds - 1, -1, -1):
        ckpt_path = os.path.join(args.out, f"round_{r}", "checkpoint.pt")
        if os.path.exists(ckpt_path):
            if os.path.lexists(final_ckpt):
                os.remove(final_ckpt)
            os.symlink(os.path.abspath(ckpt_path), final_ckpt)
            break

    if os.path.exists(accumulated_pt):
        if os.path.lexists(final_shard):
            os.remove(final_shard)
        os.symlink(os.path.abspath(accumulated_pt), final_shard)

    print(f"[active_learning] done. Outputs in {args.out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the integration tests**

```bash
pytest tests/test_active_loop.py::test_active_learning_help \
       tests/test_active_loop.py::test_active_learning_resume_skips_done \
       -xvs
```

Expected: both PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -x -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6: Quick smoke test of CLI parsing**

```bash
python scripts/active_learning.py --help
```

Expected: prints usage with `--pdb`, `--checkpoint`, `--out`, `--stop` etc. and exits 0.

- [ ] **Step 7: Commit**

```bash
git add scripts/active_learning.py tests/test_active_loop.py
git commit -m "feat: scripts/active_learning.py — active learning loop orchestrator"
```

---

## Smoke Test (Optional — Requires GPU + OpenMM)

To verify the full pipeline end-to-end on a real protein:

```bash
python scripts/active_learning.py \
    --pdb             data/kras_wt.pdb \
    --checkpoint      checkpoints/v2_256h_90k.pt \
    --out             smoke_test_loop \
    --rounds          2 \
    --proposals       10 \
    --batch-size      3 \
    --md-ns           1 \
    --replay-cap      500 \
    --novel-threshold 1.5 \
    --stop            budget \
    --stop-threshold  6 \
    --bootstrap-ns    1 \
    --fine-tune-steps 100 \
    --n-parallel      2 \
    --device          cuda
```

Expected: creates `smoke_test_loop/round_0/` and `round_1/` with `.done` stamps; `loop_summary.json` has 2 entries; `final_checkpoint.pt` symlink exists.
