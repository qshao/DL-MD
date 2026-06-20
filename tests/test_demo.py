import numpy as np
import mdtraj as md
from lsmd import demo


def _tiny_traj(tmp_path, n_res=6, n_frames=60):
    top = md.Topology()
    chain = top.add_chain()
    for i in range(n_res):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        for name, elem in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]:
            top.add_atom(name, md.element.get_by_symbol(elem), res)
    rs = np.random.RandomState(0)
    xyz = np.zeros((n_frames, n_res * 4, 3), np.float32)
    for i in range(n_res):
        base = np.array([i * 0.38, 0, 0], np.float32)
        offs = rs.randn(4, 3).astype(np.float32) * 0.02
        for fr in range(n_frames):
            wobble = rs.randn(4, 3).astype(np.float32) * 0.01
            xyz[fr, i * 4:(i + 1) * 4] = base + offs + wobble
    p = tmp_path / "tiny.pdb"
    md.Trajectory(xyz, top).save_pdb(str(p))
    return str(p)


def test_run_demo_smoke(tmp_path):
    path = _tiny_traj(tmp_path)
    out = tmp_path / "out"
    report = demo.run_demo(
        path, path,
        taus=[3, 5, 8], infer_tau=5,
        out_dir=str(out), K=4, epochs=30, batch_size=8,
        T_diff=20,          # small T for speed
        diff_steps=5,       # few inference steps for speed
    )
    # Structure keys
    assert "model_geometry" in report
    assert "diversity_rmsd" in report
    assert "ramachandran_js" in report
    assert "pca_js" in report
    assert "ensemble_recall" in report
    assert "ensemble_novelty" in report
    assert "n_md_reference" in report
    assert report["taus"] == [3, 5, 8]
    assert report["infer_tau"] == 5
    # PDB files written
    pdbs = list(out.glob("future_*.pdb"))
    assert len(pdbs) == 4
    # Metrics in valid ranges
    assert 0.0 <= report["ramachandran_js"] <= 1.0
    assert 0.0 <= report["ensemble_recall"]  <= 1.0
    assert 0.0 <= report["ensemble_novelty"] <= 1.0
    assert report["diversity_rmsd"] >= 0.0
