import json
import os
import numpy as np
import pytest
from pathlib import Path


def _write_fake_md_run(md_runs_dir, run_id, ca_coords_A, stable=True):
    """Write fake metrics.json for a stable or unstable run (no DCD needed for unit tests)."""
    run_dir = Path(md_runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "id": run_id, "md_ns": 10,
        "final_pe_kJ": -1000.0,
        "rmsd_initial_A": 0.0, "rmsd_final_A": 1.0,
        "rmsd_mean_A": 0.8, "rmsd_std_A": 0.2,
        "stable": stable, "error": None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    # Store coords for frame loading mock
    np.save(str(run_dir / "_ca_coords.npy"), ca_coords_A)


def test_pairwise_rmsd_zero_diagonal():
    from lsmd.pipeline_analysis import _pairwise_rmsd
    ca = [np.zeros((5, 3)), np.ones((5, 3))]
    mat = _pairwise_rmsd(ca)
    assert mat.shape == (2, 2)
    assert mat[0, 0] == pytest.approx(0.0)
    assert mat[1, 1] == pytest.approx(0.0)
    assert mat[0, 1] == pytest.approx(mat[1, 0])
    assert mat[0, 1] > 0


def test_pairwise_rmsd_known_value():
    from lsmd.pipeline_analysis import _pairwise_rmsd
    # Two structures: one shifted by 1 Å along x for all 4 residues
    ca_a = np.zeros((4, 3))
    ca_b = np.zeros((4, 3)); ca_b[:, 0] = 1.0
    mat = _pairwise_rmsd([ca_a, ca_b])
    assert mat[0, 1] == pytest.approx(1.0, abs=1e-5)


def test_cluster_structures_two_groups():
    from lsmd.pipeline_analysis import _cluster_structures
    # 4 structures: 2 near (0,0,0), 2 near (20,0,0) — should cluster at 2 Å
    rng = np.random.default_rng(42)
    group_a = [rng.normal(0,  0.1, (10, 3)) for _ in range(2)]
    group_b = [rng.normal(20, 0.1, (10, 3)) for _ in range(2)]
    ca_list = group_a + group_b
    labels, mat = _cluster_structures(ca_list, rmsd_cutoff_A=2.0)
    assert labels.shape == (4,)
    # The two groups should be in different clusters
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_structures_single_structure():
    from lsmd.pipeline_analysis import _cluster_structures
    labels, mat = _cluster_structures([np.zeros((5, 3))], rmsd_cutoff_A=2.0)
    assert len(labels) == 1
    assert labels[0] == 1


def test_analyze_explore_filters_unstable(tmp_path, monkeypatch):
    from lsmd.pipeline_analysis import analyze_explore
    md_runs = tmp_path / "md_runs"
    # 2 stable + 1 unstable
    _write_fake_md_run(md_runs, "00001", np.zeros((10, 3)), stable=True)
    _write_fake_md_run(md_runs, "00002", np.ones((10, 3)) * 0.5, stable=True)
    _write_fake_md_run(md_runs, "00003", np.ones((10, 3)) * 100, stable=False)

    # Monkeypatch the DCD loader to return the saved npy coords
    def _mock_load_frames(md_runs_dir):
        frames, ids = [], []
        for d in sorted(os.listdir(md_runs_dir)):
            m = json.loads((Path(md_runs_dir) / d / "metrics.json").read_text())
            if not m["stable"]:
                continue
            coords = np.load(str(Path(md_runs_dir) / d / "_ca_coords.npy"))
            frames.append(coords)
            ids.append(d)
        return frames, ids

    import lsmd.pipeline_analysis as pa
    monkeypatch.setattr(pa, "_load_stable_ca_frames", _mock_load_frames)

    out_dir = tmp_path / "results" / "explore"
    result = analyze_explore(str(md_runs), str(out_dir))
    assert result["n_proposals_attempted"] == 3
    assert result["n_stable"] == 2
    summary_path = out_dir / "cluster_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["n_stable"] == 2
