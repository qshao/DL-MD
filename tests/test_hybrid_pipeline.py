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
