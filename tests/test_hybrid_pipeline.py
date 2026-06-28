import argparse
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


def test_run_analysis_kinetics_dispatches_analyze_kinetics(tmp_path, monkeypatch):
    """run_analysis dispatches analyze_kinetics when objective is kinetics."""
    import lsmd.pipeline_analysis as pa
    called = []
    monkeypatch.setattr(pa, "analyze_kinetics",
                        lambda md_runs_dir, out_dir, **kw: called.append("kinetics") or {})
    args = argparse.Namespace(
        objective="kinetics",
        out=str(tmp_path),
    )
    from scripts.hybrid_pipeline import run_analysis
    run_analysis(args)
    assert called == ["kinetics"]


def test_run_proposals_fes_builds_cv_basis(tmp_path, monkeypatch):
    """run_proposals builds cv_basis.pt for fes objective when absent (Fix 1).

    Stage 1 runs in 'sample' mode via explore_conformations.py which never writes
    cv_basis.pt.  run_proposals must construct it from the shard so Stage 4
    (analyze_fes) can project frames without a FileNotFoundError.
    """
    import torch
    import scripts.hybrid_pipeline as hp
    import lsmd.cv_guidance as cvg

    proposals_dir = tmp_path / "proposals"
    proposals_dir.mkdir()
    # candidates_dir intentionally absent — triggers the "n_candidates == 0" path
    # so the subprocess mock is reached rather than the early-return skip.

    # Stage 1 subprocess is mocked — no real ML model needed
    monkeypatch.setattr(hp, "_run_proposals_subprocess", lambda args: None)

    # torch.load is mocked to return a minimal fake shard (no real file on disk)
    dummy_t = torch.zeros(10, 5, 3)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"t": dummy_t})

    # torch.save is mocked to write a sentinel file; FakeCVSpace is locally
    # defined and therefore not picklable by torch.save's pickle backend.
    saved_paths = []

    def _fake_save(obj, path):
        with open(path, "wb") as fh:
            fh.write(b"dummy_cv_basis")
        saved_paths.append(path)

    monkeypatch.setattr(torch, "save", _fake_save)

    # CVSpace is mocked so no real PCA computation is needed
    class FakeCVSpace:
        def __init__(self, n_pc=5):
            self.n_pc = n_pc

        def fit(self, x):
            pass

    monkeypatch.setattr(cvg, "CVSpace", FakeCVSpace)

    args = argparse.Namespace(
        out=str(tmp_path),
        objective="fes",
        shard="dummy_shard.pt",
        n_proposals=10,
    )
    hp.run_proposals(args)

    cv_basis_path = proposals_dir / "cv_basis.pt"
    assert any("cv_basis.pt" in p for p in saved_paths), (
        "torch.save was not called with cv_basis.pt"
    )
    assert cv_basis_path.exists(), "cv_basis.pt file was not written to proposals_dir"
