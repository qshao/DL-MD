import torch
import pytest
from lsmd import geometry as g
from lsmd import decoder as dec
from lsmd import validation as val


def _atoms(n_structs, n_res, t_base, rot_scale, seed=0):
    """Build [n_structs, n_res, 4, 3] atom tensor with reproducible noise."""
    torch.manual_seed(seed)
    R = g.so3_exp(torch.randn(n_structs, n_res, 3) * rot_scale)
    t = t_base.unsqueeze(0).expand(n_structs, -1, -1).clone()
    return torch.stack([dec.build_structure(R[k], t[k]) for k in range(n_structs)])


def _t_base(n_res=6):
    t = torch.zeros(n_res, 3)
    for i in range(n_res):
        t[i, 0] = i * 3.8
    return t


# ── existing tests (unchanged) ─────────────────────────────────────────────────

def test_geometry_metrics_keys():
    R = g.so3_exp(torch.randn(5, 3) * 0.1)
    t = torch.arange(5).float().unsqueeze(-1).repeat(1, 3) * 3.8
    atoms = dec.build_structure(R, t)
    mt = val.geometry_metrics(atoms)
    assert {"ca_bond_mean", "peptide_violation", "clash_count"} <= set(mt)


def test_diversity_zero_for_identical():
    R = g.so3_exp(torch.randn(4, 3) * 0.1)
    t = torch.randn(4, 3)
    atoms = dec.build_structure(R, t)
    stacked = atoms.unsqueeze(0).repeat(5, 1, 1, 1)
    assert val.diversity(stacked) < 1e-5


def test_baselines_shapes():
    R = g.so3_exp(torch.randn(6, 3) * 0.1)
    t = torch.randn(6, 3)
    uc = val.baseline_copy(R, t, K=4)
    un = val.baseline_noise(R, t, K=4, sigma=0.2)
    assert uc.shape == (4, 6, 6) and un.shape == (4, 6, 6)
    assert uc.abs().sum() < 1e-5


# ── backbone_torsions ──────────────────────────────────────────────────────────

def test_backbone_torsions_shape():
    t = _t_base(n_res=6)
    atoms = _atoms(1, 6, t, 0.1)[0]    # [6, 4, 3]
    phi, psi = val.backbone_torsions(atoms)
    assert phi.shape == (4,)            # N - 2 = 4
    assert psi.shape == (4,)


def test_backbone_torsions_range():
    t = _t_base(n_res=6)
    atoms = _atoms(1, 6, t, 0.5)[0]
    phi, psi = val.backbone_torsions(atoms)
    assert (phi >= -torch.pi).all() and (phi <= torch.pi).all()
    assert (psi >= -torch.pi).all() and (psi <= torch.pi).all()


# ── ramachandran_js ────────────────────────────────────────────────────────────

def test_ramachandran_js_identical():
    t = _t_base()
    atoms = _atoms(5, 6, t, 0.1)
    js = val.ramachandran_js(atoms, atoms.clone())
    assert js < 1e-5


def test_ramachandran_js_bounded():
    t = _t_base()
    atoms_model = _atoms(5, 6, t, 0.1, seed=0)
    atoms_md    = _atoms(5, 6, t, 1.5, seed=1)  # large rotations → different angles
    js = val.ramachandran_js(atoms_model, atoms_md)
    assert 0.0 <= js <= 1.0


# ── pca_js ─────────────────────────────────────────────────────────────────────

def test_pca_js_returns_dict():
    t = _t_base()
    atoms_model = _atoms(5, 6, t, 0.1, seed=0)
    atoms_md    = _atoms(5, 6, t, 0.1, seed=2)
    result = val.pca_js(atoms_model, atoms_md)
    assert set(result.keys()) == {"js", "var_explained"}
    assert 0.0 <= result["js"] <= 1.0
    assert len(result["var_explained"]) == 2
    assert all(0.0 <= v <= 1.0 for v in result["var_explained"])


# ── ensemble_recall / novelty ─────────────────────────────────────────────────

def test_ensemble_recall_perfect():
    t = _t_base()
    atoms = _atoms(5, 6, t, 0.01, seed=0)
    # Model = MD → every MD frame is within r of itself
    recall = val.ensemble_recall(atoms, atoms.clone(), r_ang=0.01)
    assert recall == 1.0


def test_ensemble_recall_zero():
    t_near = _t_base()
    t_far = t_near.clone(); t_far[:, 0] += 200.0   # shift x by 200 Å
    atoms_model = _atoms(5, 6, t_near, 0.01, seed=0)
    atoms_md    = _atoms(5, 6, t_far,  0.01, seed=1)
    recall = val.ensemble_recall(atoms_model, atoms_md, r_ang=2.0)
    assert recall == 0.0


def test_ensemble_novelty_zero():
    t = _t_base()
    atoms = _atoms(5, 6, t, 0.01, seed=0)
    # Model = MD clone → no sample is novel
    novelty = val.ensemble_novelty(atoms, atoms.clone(), r_ang=2.0)
    assert novelty == 0.0


# ── CA-specific metrics ───────────────────────────────────────────────────────

def test_ca_geometry_keys():
    ca = torch.randn(10, 3)
    out = val.ca_geometry(ca)
    assert set(out) == {"ca_bond_mean", "ca_bond_min", "ca_bond_max", "clash_count"}


def test_distance_matrix_js_identical_is_zero():
    torch.manual_seed(0)
    ca = torch.randn(6, 12, 3)
    js = val.distance_matrix_js(ca, ca)
    assert js < 1e-3


def test_distance_matrix_js_bounded():
    a = torch.randn(5, 12, 3)
    b = torch.randn(5, 12, 3) * 5.0 + 20.0
    js = val.distance_matrix_js(a, b)
    assert 0.0 <= js <= 1.0


def test_rmsf_profile_identical_corr_one():
    torch.manual_seed(0)
    ca = torch.randn(8, 10, 3)
    out = val.rmsf_profile(ca, ca)
    assert len(out["model"]) == 10
    assert abs(out["corr"] - 1.0) < 1e-4


def test_displacement_js_identical_is_zero():
    d = torch.rand(50)
    out = val.displacement_js(d, d)
    assert out["js"] < 1e-3
    assert abs(out["model_mean"] - out["md_mean"]) < 1e-6


def test_pca_js_accepts_ca_pointcloud():
    torch.manual_seed(0)
    ca = torch.randn(6, 12, 3)
    out = val.pca_js(ca, ca)              # [K,P,3] inputs, not [K,N,4,3]
    assert out["js"] < 1e-3


def test_recall_accepts_ca_pointcloud():
    torch.manual_seed(0)
    ca = torch.randn(5, 12, 3)
    assert val.ensemble_recall(ca, ca, r_ang=0.01) == 1.0
    assert val.ensemble_novelty(ca, ca, r_ang=0.01) == 0.0
