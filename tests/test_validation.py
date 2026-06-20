import torch
from lsmd import geometry as g
from lsmd import decoder as dec
from lsmd import validation as val


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


def test_ensemble_overlap_identical_is_high():
    ca = torch.randn(50, 3)
    o = val.ensemble_overlap(ca, ca.clone())
    assert o > 0.95


def test_baselines_shapes():
    R = g.so3_exp(torch.randn(6, 3) * 0.1)
    t = torch.randn(6, 3)
    uc = val.baseline_copy(R, t, K=4)
    un = val.baseline_noise(R, t, K=4, sigma=0.2)
    assert uc.shape == (4, 6, 6) and un.shape == (4, 6, 6)
    assert uc.abs().sum() < 1e-5  # copy = zero update
