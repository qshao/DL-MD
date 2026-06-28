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


# ---------------------------------------------------------------------------
# Task 2: bootstrap helper tests
# ---------------------------------------------------------------------------
import os
import json
import tempfile
import numpy as np
from lsmd.active_loop import (
    _pdb_to_shard, _geometry_pass_rate, _min_rmsd_kabsch, bootstrap_check,
)


def _write_tiny_pdb(path, n_res=5):
    """Write a minimal backbone PDB (N, CA, C, O per residue)."""
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
                f"ATOM  {serial:5d} {aname} {rn:3s} A{ri+1:4d}    "
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


# ---------------------------------------------------------------------------
# Task 3: shard_from_md_runs and build_replay_shard tests
# ---------------------------------------------------------------------------
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
