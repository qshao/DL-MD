# tests/test_validate_physics_cli.py
import torch
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "validate_physics",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "validate_physics.py"))
validate_physics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_physics)


def test_summarize_means():
    proteins = {
        "a": {"structural": {"rmsf_corr": 0.4, "dist_js": 0.01},
              "thermodynamic": {"fes_js": 0.1}, "kinetic": {"relax_ratio": 1.0}},
        "b": {"structural": {"rmsf_corr": 0.6, "dist_js": 0.03},
              "thermodynamic": {"fes_js": 0.3}, "kinetic": {"relax_ratio": 1.2}},
    }
    s = validate_physics.summarize(proteins)
    assert abs(s["mean_rmsf_corr"] - 0.5) < 1e-9
    assert abs(s["mean_dist_js"] - 0.02) < 1e-9
    assert abs(s["mean_fes_js"] - 0.2) < 1e-9
    assert abs(s["mean_relax_ratio"] - 1.1) < 1e-9
