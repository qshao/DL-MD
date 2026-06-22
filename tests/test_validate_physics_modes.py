import json, subprocess, sys, os, pytest

CKPT = "checkpoints/v2_256h_90k.pt"
SHARD = "data/atlas/3u7t_A.pt"


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
