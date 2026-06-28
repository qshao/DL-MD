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


def test_analyze_fes_boltzmann_inversion(tmp_path, monkeypatch):
    """FES minimum is 0.0 and values are non-negative."""
    from lsmd.pipeline_analysis import analyze_fes
    from lsmd.cv_guidance import CVSpace
    import torch

    # Build a tiny CVSpace (5 residues, n_pc=2) and save it
    rng = torch.Generator(); rng.manual_seed(0)
    ca_ref = torch.randn(20, 5, 3, generator=rng)
    cv_space = CVSpace(n_pc=2)
    cv_space.fit(ca_ref)
    cv_basis_path = str(tmp_path / "cv_basis.pt")
    cv_space.save(cv_basis_path)

    # Fake MD runs: 3 stable, each with known CA coords
    md_runs = tmp_path / "md_runs"
    n_res = 5

    def _mock_load_all_ca(md_runs_dir, n_frames_per_run=10):
        # Return synthetic frames without loading real DCD files
        all_frames = torch.randn(30, n_res, 3, generator=rng)
        return all_frames, 30

    import lsmd.pipeline_analysis as pa
    monkeypatch.setattr(pa, "_load_all_ca_frames", _mock_load_all_ca)

    out_dir = tmp_path / "fes"
    result = analyze_fes(str(md_runs), cv_basis_path, str(out_dir), temp_K=310.0, n_bins=10)

    assert result["n_frames_stable"] == 30
    assert result["fes_min_kcal"] == pytest.approx(0.0)
    assert result["fes_max_kcal"] >= 0.0
    assert (out_dir / "fes.npy").exists()
    fes = np.load(str(out_dir / "fes.npy"))
    assert fes.shape == (10, 10)
    assert fes.min() == pytest.approx(0.0, abs=1e-6)
    assert (fes >= 0).all()


def test_analyze_kinetics_no_pyemma_raises(tmp_path, monkeypatch):
    """analyze_kinetics raises ImportError when PyEMMA is unavailable."""
    import lsmd.pipeline_analysis as pa
    monkeypatch.setattr(pa, "_HAS_PYEMMA", False)
    with pytest.raises(ImportError, match="pyemma is required"):
        pa.analyze_kinetics(str(tmp_path / "md"), str(tmp_path / "out"))


def test_analyze_kinetics_smoke(tmp_path, monkeypatch):
    """analyze_kinetics runs on synthetic featurised data when PyEMMA available."""
    pytest.importorskip("pyemma")
    from lsmd.pipeline_analysis import analyze_kinetics

    # Provide 5 fake trajectories via monkeypatched featuriser
    n_res = 10
    rng = np.random.default_rng(0)

    def _mock_load_featurised(md_runs_dir):
        # Return list of 5 synthetic [100, n_features] arrays
        n_pairs = n_res * (n_res - 1) // 2
        return [rng.standard_normal((100, n_pairs)).astype(np.float32)
                for _ in range(5)]

    import lsmd.pipeline_analysis as pa
    monkeypatch.setattr(pa, "_load_featurised_trajs", _mock_load_featurised)

    out_dir = tmp_path / "kinetics"
    result = analyze_kinetics(
        str(tmp_path / "md"), str(out_dir),
        tica_lag=5, n_clusters=10, msm_lag=2,
    )
    assert result["n_trajectories"] == 5
    assert (out_dir / "transition_matrix.npy").exists()
    assert (out_dir / "msm_summary.json").exists()
