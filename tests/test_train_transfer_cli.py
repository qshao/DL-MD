import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "train_transfer.py")


def test_train_transfer_help_lists_phase3_flags():
    out = subprocess.run([sys.executable, SCRIPT, "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0
    for flag in ["--energy_ckpt", "--lam_energy", "--lam_fdt", "--phys_warmup",
                 "--w_hi", "--w_lo"]:
        assert flag in out.stdout
