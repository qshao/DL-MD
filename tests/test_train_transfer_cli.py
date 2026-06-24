import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "train_transfer.py")


def test_train_transfer_help_lists_phase3_flags():
    out = subprocess.run([sys.executable, SCRIPT, "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0
    for flag in ["--lam_fdt", "--phys_warmup"]:
        assert flag in out.stdout


def test_train_transfer_help_lists_shard_flag():
    out = subprocess.run([sys.executable, SCRIPT, "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0
    assert "--shard" in out.stdout


def test_train_transfer_requires_shard_source():
    """Running with neither --shards_dir nor --shard must exit non-zero."""
    out = subprocess.run(
        [sys.executable, SCRIPT, "--lags_ps", "200", "--steps", "1",
         "--out", "/tmp/dummy.pt"],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode != 0
