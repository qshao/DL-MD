import json
import subprocess
import sys
import numpy as np
import pytest
from pathlib import Path


def _write_summary(tmp_path, n=10):
    import random
    random.seed(0)
    records = []
    for i in range(n):
        records.append({
            "id": i,
            "cv": [float(x) for x in np.random.randn(5).tolist()],
            "rmsd_native": round(abs(np.random.randn()) * 3, 3),
            "clashes": 0.0,
            "bond_rmsd": 0.02,
            "md_pass": bool(i % 3 == 0),
            "md_rmsd_final": round(abs(np.random.randn()) * 3, 3) if i % 3 == 0 else None,
            "md_rg_final": round(10.0 + np.random.randn(), 3) if i % 3 == 0 else None,
        })
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(records))
    np.save(str(tmp_path / "cv_coords.npy"),
            np.stack([r["cv"] for r in records]))
    return str(tmp_path)


def test_summarize_runs(tmp_path):
    out_dir = _write_summary(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/summarize_exploration.py",
         "--out", out_dir],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent)
    )
    assert result.returncode == 0, result.stderr
    assert "md_pass" in result.stdout.lower() or "validated" in result.stdout.lower()


def test_summarize_creates_figure(tmp_path):
    out_dir = _write_summary(tmp_path)
    subprocess.run(
        [sys.executable, "scripts/summarize_exploration.py",
         "--out", out_dir],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent), check=True
    )
    assert (tmp_path / "md_summary.png").exists()
