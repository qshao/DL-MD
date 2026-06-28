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


# ---------------------------------------------------------------------------
# Task 4: check_convergence tests
# ---------------------------------------------------------------------------
import math
from lsmd.active_loop import check_convergence


def test_convergence_budget_hits():
    """Budget criterion converges when total_md_ns >= threshold."""
    state = {"total_md_ns": 100.0, "last_novel_fraction": 0.5, "round": 1}
    converged, metric = check_convergence("budget", 100.0, state)
    assert converged
    assert abs(metric - 100.0) < 1e-6


def test_convergence_budget_miss():
    """Budget criterion does not converge when total_md_ns < threshold."""
    state = {"total_md_ns": 45.0, "last_novel_fraction": 0.5, "round": 1}
    converged, metric = check_convergence("budget", 100.0, state)
    assert not converged
    assert abs(metric - 45.0) < 1e-6


def test_convergence_coverage_hits():
    """Coverage criterion converges when last_novel_fraction < threshold."""
    state = {"total_md_ns": 10.0, "last_novel_fraction": 0.05, "round": 3}
    converged, metric = check_convergence("coverage", 0.10, state)
    assert converged
    assert abs(metric - 0.05) < 1e-6


def test_convergence_fes_insufficient_data():
    """FES criterion returns (False, nan) when round < 2 or frames < 50."""
    N = 10
    state = {
        "total_md_ns": 10.0,
        "last_novel_fraction": 0.5,
        "round": 0,
        "accumulated_frames": torch.randn(20, N, 3),
        "cv_basis": None,
        "prev_hist": None,
    }
    converged, metric = check_convergence("fes", 0.05, state)
    assert not converged
    assert math.isnan(metric)


def test_convergence_fes_hits():
    """FES converges when JS divergence < threshold (identical PC scores → JS ≈ 0).

    prev_hist is now a raw [F, 2] PC scores array (not a pre-computed 50×50
    histogram).  _check_fes builds histograms with a unified range from both
    the current and previous PC scores, so using the same frames for both
    yields near-zero JS divergence.
    """
    N = 10
    frames = torch.randn(80, N, 3)
    cv = CVSpace(n_pc=2)
    cv.fit(frames)

    # prev_hist is now [F, 2] raw PC scores (not a 50×50 histogram)
    projections = cv.project_batch(frames.float())[:, :2].detach().cpu().numpy()
    prev_pc_scores = projections  # shape [80, 2]

    state = {
        "total_md_ns": 10.0,
        "last_novel_fraction": 0.2,
        "round": 3,
        "accumulated_frames": frames,
        "cv_basis": cv,
        "prev_hist": prev_pc_scores,
    }
    converged, metric = check_convergence("fes", 0.05, state)
    assert converged
    assert metric < 0.05


def test_convergence_unknown_criterion():
    """Unknown criterion raises ValueError."""
    state = {"total_md_ns": 10.0, "last_novel_fraction": 0.5, "round": 1}
    with pytest.raises(ValueError, match="Unknown convergence criterion"):
        check_convergence("unknown", 0.5, state)


# ---------------------------------------------------------------------------
# Task 5: integration tests for scripts/active_learning.py
# ---------------------------------------------------------------------------
import subprocess
import sys
from pathlib import Path


def test_active_learning_help():
    """scripts/active_learning.py --help must exit 0."""
    result = subprocess.run(
        [sys.executable, "scripts/active_learning.py", "--help"],
        capture_output=True, cwd=str(Path(__file__).resolve().parent.parent)
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


def test_active_loop_early_termination(tmp_path):
    """Loop terminates (returns None) when no proposals are novel."""
    from unittest.mock import patch, MagicMock
    from scripts.active_learning import run_round

    N = 5
    pdb_path = str(tmp_path / "input.pdb")
    _write_tiny_pdb(pdb_path, n_res=N)

    # accumulated_pt deliberately absent so run_round falls back to shard_1f["t"]
    accumulated_pt = str(tmp_path / "accumulated_frames.pt")

    shard_1f = {
        "R":        torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, N, -1, -1).clone(),
        "t":        torch.randn(1, N, 3),
        "res_type": torch.zeros(N, dtype=torch.long),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "n_res":    N,
        "dt":       200.0,
        "seq":      ["ALA"] * N,
    }
    protein_meta = {k: shard_1f[k]
                    for k in ("res_type", "chain_id", "res_index", "seq", "n_res")}

    class FakeArgs:
        out            = str(tmp_path)
        pdb            = pdb_path
        proposals      = 3
        batch_size     = 2
        md_ns          = 1.0
        novel_threshold = 1.5
        device         = "cpu"
        stop           = "coverage"
        stop_threshold = 0.1
        n_parallel     = 1
        replay_cap     = 100
        fine_tune_steps = 10
        bootstrap_ns   = 1.0
        rounds         = 1

    args = FakeArgs()

    # rollout returns a 1-element list; traj[-1] is a [N,3] Ca tensor
    fake_traj = [torch.randn(N, 3)]

    with patch("torch.load", return_value={"state_dict": {}}), \
         patch("scripts.active_learning.load_checkpoint",
               return_value=(MagicMock(), MagicMock(), MagicMock())), \
         patch("scripts.active_learning.rollout", return_value=fake_traj), \
         patch("scripts.active_learning._min_rmsd_kabsch", return_value=0.0), \
         patch("scripts.active_learning.write_ca_pdb"):

        result = run_round(
            round_num=0,
            args=args,
            current_ckpt="fake_checkpoint.pt",
            protein_meta=protein_meta,
            shard_1f=shard_1f,
            accumulated_pt=accumulated_pt,
            prev_total_md_ns=0.0,
            prev_novel_fraction=1.0,
            prev_accumulated_t=None,
            prev_hist=None,
        )

    # All proposals have RMSD 0.0 < novel_threshold 1.5 → no novel → early exit
    assert result is None, "Expected early termination (None) when no novel proposals"
