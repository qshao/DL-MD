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


def test_make_multi_lag_pairs():
    taus = [5, 10, 20]
    pairs = d.make_multi_lag_pairs(num_frames=50, taus=taus)
    assert pairs.shape[1] == 3, "columns: (start, end, tau)"
    # every pair should have the correct delta
    assert (pairs[:, 1] - pairs[:, 0] == pairs[:, 2]).all()
    # all three tau values present
    assert set(pairs[:, 2].tolist()) == set(taus)
    # more pairs than any single lag alone
    single = d.make_pairs(num_frames=50, tau=10)
    assert pairs.shape[0] > single.shape[0]
    # sorted by start frame → time_split is leakage-free
    assert (pairs[1:, 0] >= pairs[:-1, 0]).all(), "must be sorted by start frame"
    train, val = d.time_split(pairs, val_frac=0.2)
    assert train[:, 0].max() <= val[:, 0].min()


def test_compute_frame_weights_shape_and_mean(tmp_path):
    path = _tiny_traj(tmp_path)
    frames = d.load_frames(path, path)
    weights = d.compute_frame_weights(frames)
    F = frames["R"].shape[0]
    assert weights.shape == (F,)
    assert abs(weights.mean().item() - 1.0) < 0.01
    assert (weights > 0).all()


def test_compute_frame_weights_uniform():
    """Identical CA positions → uniform density → all weights equal."""
    F, N = 10, 4
    # All frames have the same CA positions
    t_identical = torch.zeros(F, N, 3)
    for i in range(N):
        t_identical[:, i, 0] = i * 3.8
    frames = {
        "R": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(F, N, 3, 3).clone(),
        "t": t_identical,
    }
    weights = d.compute_frame_weights(frames)
    assert weights.std().item() < 1e-3
    assert abs(weights.mean().item() - 1.0) < 0.01
