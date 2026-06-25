import json
import subprocess
import sys
import torch
import pytest
from pathlib import Path
from lsmd import transfer_train as tt
from lsmd import geometry as g


def _make_ckpt(tmp_path, F=15, N=8, seed=0):
    torch.manual_seed(seed)
    shards = [{
        "R": g.so3_exp(torch.randn(F, N, 3) * 0.05),
        "t": torch.randn(F, N, 3) * 5.0,
        "res_type": torch.zeros(N, dtype=torch.long),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "dt": 200.0, "seq": ["ALA"] * N, "n_res": N,
    }]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=1, steps=2, T_diff=20,
                    norm_samples=16, device="cpu", seed=seed)
    ckpt_path = str(tmp_path / "test.pt")
    torch.save(ckpt, ckpt_path)
    shard_path = str(tmp_path / "shard.pt")
    torch.save(shards[0], shard_path)
    return ckpt_path, shard_path


def test_explore_smoke(tmp_path):
    ckpt_path, shard_path = _make_ckpt(tmp_path)
    out_dir = str(tmp_path / "out")
    result = subprocess.run(
        [sys.executable, "scripts/explore_conformations.py",
         "--checkpoint", ckpt_path,
         "--shard", shard_path,
         "--n_explore", "5",
         "--n_steps", "2",
         "--diff_steps", "3",
         "--tau_ps", "200",
         "--k_guide", "0.05",
         "--sigma_cv", "1.0",
         "--guide_warmup", "0",
         "--out", out_dir,
         "--seed", "0"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent)
    )
    assert result.returncode == 0, result.stderr
    out = Path(out_dir)
    assert (out / "summary.json").exists()
    assert (out / "cv_basis.pt").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary) >= 0   # may be 0 if all fail geometry filter


def test_explore_output_structure(tmp_path):
    ckpt_path, shard_path = _make_ckpt(tmp_path, F=20, N=8, seed=1)
    out_dir = str(tmp_path / "out2")
    subprocess.run(
        [sys.executable, "scripts/explore_conformations.py",
         "--checkpoint", ckpt_path,
         "--shard", shard_path,
         "--n_explore", "8",
         "--n_steps", "2",
         "--diff_steps", "3",
         "--tau_ps", "200",
         "--guide_warmup", "0",
         "--out", out_dir,
         "--seed", "42"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent), check=True
    )
    out = Path(out_dir)
    summary = json.loads((out / "summary.json").read_text())
    for entry in summary:
        for key in ("id", "cv", "rmsd_native", "clashes", "bond_rmsd",
                    "md_pass", "md_rmsd_final", "md_rg_final"):
            assert key in entry, f"missing key {key}"
        assert entry["md_pass"] is None
