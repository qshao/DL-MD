import importlib.util
import json, subprocess, sys, os, pytest

CKPT = "checkpoints/v2_256h_90k.pt"
SHARD = "data/atlas/3u7t_A.pt"

_spec = importlib.util.spec_from_file_location(
    "validate_physics",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "validate_physics.py"))
_validate_physics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validate_physics)


def test_validate_physics_has_noether_flag():
    """--help output lists --noether flag."""
    result = subprocess.run([sys.executable, "scripts/validate_physics.py", "--help"],
                            capture_output=True, text=True)
    assert "--noether" in result.stdout


@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(SHARD)),
                    reason="data not available")
def test_validate_physics_noether_runs(tmp_path):
    out = tmp_path / "modeA.json"
    subprocess.run([sys.executable, "scripts/validate_physics.py",
                    "--checkpoint", CKPT, "--shard", SHARD,
                    "--steps", "3", "--tau_ps", "2000", "--diff_steps", "2",
                    "--noether", "--out", str(out)],
                   check=True)
    report = json.loads(out.read_text())
    assert report["settings"]["noether"] is True
    assert report["proteins"]["3u7t_A"]["reweight"] is None
    # kinetic fields are present (Mode A does not null them)
    assert report["proteins"]["3u7t_A"]["kinetic"]["relax_ratio"] is not None


def test_summarize_skips_none_relax_ratio():
    """summarize() ignores None relax_ratio entries (Mode B proteins)."""
    proteins = {
        "A": {"structural": {"rmsf_corr": 0.5, "dist_js": 0.01},
              "thermodynamic": {"fes_js": 0.4},
              "kinetic": {"relax_ratio": None}},
        "B": {"structural": {"rmsf_corr": 0.7, "dist_js": 0.02},
              "thermodynamic": {"fes_js": 0.6},
              "kinetic": {"relax_ratio": 2.0}},
    }
    s = _validate_physics.summarize(proteins)
    assert abs(s["mean_rmsf_corr"] - 0.6) < 1e-6   # (0.5 + 0.7) / 2
    assert abs(s["mean_relax_ratio"] - 2.0) < 1e-6  # only protein B has non-None


def test_validate_physics_has_reweight_flags():
    result = subprocess.run([sys.executable, "scripts/validate_physics.py", "--help"],
                            capture_output=True, text=True)
    for flag in ["--reweight", "--kT_reweight", "--w_angle", "--w_mj", "--w_wca_cg"]:
        assert flag in result.stdout, f"missing flag {flag}"


@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(SHARD)),
                    reason="data not available")
def test_validate_physics_reweight_nulls_kinetics(tmp_path):
    out = tmp_path / "modeB.json"
    subprocess.run([sys.executable, "scripts/validate_physics.py",
                    "--checkpoint", CKPT, "--shard", SHARD,
                    "--steps", "3", "--tau_ps", "2000", "--diff_steps", "2",
                    "--noether", "--reweight", "--kT_reweight", "0.593",
                    "--out", str(out)],
                   check=True)
    report = json.loads(out.read_text())
    prot = report["proteins"]["3u7t_A"]
    assert prot["reweight"] is not None
    assert "n_eff" in prot["reweight"]
    assert prot["kinetic"]["relax_ratio"] is None
    assert prot["kinetic"]["msd_rmse"] is None
