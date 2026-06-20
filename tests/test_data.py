import numpy as np
import mdtraj as md
import torch
from lsmd import data as d


def _tiny_traj(tmp_path, n_res=4, n_frames=10):
    top = md.Topology()
    chain = top.add_chain()
    for i in range(n_res):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        for name, elem in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]:
            top.add_atom(name, md.element.get_by_symbol(elem), res)
    xyz = np.random.RandomState(0).randn(n_frames, n_res * 4, 3).astype(np.float32) * 0.3
    # spread residues out so frames are well-defined
    for i in range(n_res):
        xyz[:, i * 4:(i + 1) * 4, 0] += i * 4.0
    traj = md.Trajectory(xyz, top)
    p = tmp_path / "tiny.pdb"
    traj.save_pdb(str(p))
    return str(p)


def test_load_frames_shapes(tmp_path):
    path = _tiny_traj(tmp_path)
    out = d.load_frames(path, path)
    assert out["R"].shape == (10, 4, 3, 3)
    assert out["t"].shape == (10, 4, 3)
    assert out["res_type"].shape == (4,)
    assert out["chain_id"].shape == (4,)
    assert out["n_types"] >= 1


def test_make_pairs_and_split():
    pairs = d.make_pairs(num_frames=100, tau=10)
    assert pairs.shape[1] == 2
    assert (pairs[:, 1] - pairs[:, 0] == 10).all()
    assert pairs[:, 1].max() < 100
    train, val = d.time_split(pairs, val_frac=0.2)
    # time-ordered: max train start < min val start (no leakage)
    assert train[:, 0].max() < val[:, 0].min()
