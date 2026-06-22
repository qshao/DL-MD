import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "fit_energy.py")


def test_fit_energy_help_lists_flags():
    out = subprocess.run([sys.executable, SCRIPT, "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0
    for flag in ["--shard", "--sigma", "--kT", "--out", "--gate", "--gate_threshold"]:
        assert flag in out.stdout


SHARD = os.path.join(REPO, "data", "atlas", "3u7t_A.pt")


@pytest.mark.skipif(not os.path.exists(SHARD), reason="atlas shard absent")
def test_fit_energy_runs_and_writes_checkpoint(tmp_path):
    out_path = tmp_path / "energy_theta.pt"
    out = subprocess.run(
        [sys.executable, SCRIPT, "--shard", SHARD, "--steps", "20",
         "--out", str(out_path)],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert out_path.exists()
