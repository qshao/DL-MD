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
    report = demo.run_demo(path, path, tau=5, out_dir=str(out), K=4, epochs=30)
    assert "model_geometry" in report
    assert "diversity" in report
    # K PDB files were written
    pdbs = list(out.glob("future_*.pdb"))
    assert len(pdbs) == 4
