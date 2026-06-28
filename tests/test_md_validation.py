import json
import pytest
from lsmd.md_validation import run_md


def test_checkpoint_returns_cached(tmp_path):
    """run_md returns cached metrics.json without running MD — only for successful runs.

    A cached result with error=None is treated as a valid checkpoint and returned
    immediately without invoking OpenMM.  A cached result with error set is NOT
    treated as a valid checkpoint; see test_checkpoint_does_not_cache_failed_run.
    """
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


def test_checkpoint_does_not_cache_failed_run(tmp_path, monkeypatch):
    """run_md re-runs when a cached metrics.json contains an error (Fix 2).

    Transient failures (GPU OOM, bad PDB, etc.) must not be cached permanently.
    When metrics.json has error != None the checkpoint is ignored and OpenMM is
    invoked again — detected here by forcing HAS_OPENMM=False so the re-run
    attempt raises ImportError rather than silently returning the stale error dict.
    """
    import lsmd.md_validation as mdv
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    failed_cached = {
        "id": "test", "md_ns": 10, "final_pe_kJ": None,
        "rmsd_initial_A": None, "rmsd_final_A": None,
        "rmsd_mean_A": None, "rmsd_std_A": None,
        "stable": False, "error": "GPU OOM: out of memory",
    }
    (out_dir / "metrics.json").write_text(json.dumps(failed_cached))
    # Force HAS_OPENMM=False to detect that the code attempted a re-run
    monkeypatch.setattr(mdv, "HAS_OPENMM", False)
    with pytest.raises(ImportError, match="openmm is required"):
        run_md("nonexistent.pdb", str(out_dir), md_ns=10)


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
