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
