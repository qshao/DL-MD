import json, subprocess, sys
import pytest


def test_compare_modes_runs(tmp_path):
    proteins = {
        "3u7t_A": {
            "structural":    {"rmsf_corr": 0.5, "dist_js": 0.01},
            "thermodynamic": {"fes_js": 0.60, "fes_rmse_kT": 0.8, "pop_tv": 0.4},
            "kinetic":       {"msd_rmse": 1.0, "acf_rmse": 0.05,
                              "relax_model_ps": 4000.0, "relax_md_ps": 3000.0,
                              "relax_ratio": 1.33},
            "reweight": None, "n_res": 46,
        }
    }
    base = {"heldout": False, "proteins": proteins, "summary": {}}
    modeA_proteins = {k: dict(v, kinetic=dict(v["kinetic"], relax_ratio=0.9))
                      for k, v in proteins.items()}
    modeA = {"heldout": False, "proteins": modeA_proteins, "summary": {}}
    modeB_proteins = {k: dict(v,
                               thermodynamic=dict(v["thermodynamic"], fes_js=0.30),
                               kinetic=dict(v["kinetic"], relax_ratio=None))
                      for k, v in proteins.items()}
    modeB = {"heldout": False, "proteins": modeB_proteins, "summary": {}}

    (tmp_path / "base.json").write_text(json.dumps(base))
    (tmp_path / "modeA.json").write_text(json.dumps(modeA))
    (tmp_path / "modeB.json").write_text(json.dumps(modeB))

    result = subprocess.run(
        [sys.executable, "scripts/compare_modes.py",
         str(tmp_path / "base.json"),
         str(tmp_path / "modeA.json"),
         str(tmp_path / "modeB.json")],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "relax_ratio" in result.stdout
    assert "fes_js" in result.stdout
    # Mode B relax_ratio is null → "null" in output
    assert "null" in result.stdout
    # Mode A improved relax_ratio: 1.33 → 0.9 → negative delta
    assert "-" in result.stdout
