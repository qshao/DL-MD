import math
import torch
import pytest
from lsmd.cg_energy import angle_energy


def test_angle_energy_equilibrium():
    """Triplet at exactly theta0=2.094 rad (120°) → energy = 0."""
    # v1 = CA_0 - CA_1 = (-3.8, 0, 0)
    # We need v2 = CA_2 - CA_1 such that angle(v1, v2) = 120°:
    #   cos(120°) = -0.5; v1_hat = (-1,0,0); need dot(v1_hat, v2_hat) = -0.5
    #   → v2_hat = (0.5, 0.866, 0)
    CA_0 = torch.tensor([-3.8, 0.0, 0.0])
    CA_1 = torch.tensor([0.0,  0.0, 0.0])
    CA_2 = torch.tensor([1.9,  3.29, 0.0])   # 3.8 * (0.5, 0.866, 0)
    t = torch.stack([CA_0, CA_1, CA_2])
    chain_id = torch.zeros(3, dtype=torch.long)
    E = angle_energy(t, chain_id, theta0=2.094)
    assert float(E.abs()) < 1e-3


def test_angle_energy_straight_chain():
    """Straight chain (180°) → maximum energy = k * (pi - theta0)^2."""
    CA_0 = torch.tensor([0.0, 0.0, 0.0])
    CA_1 = torch.tensor([3.8, 0.0, 0.0])
    CA_2 = torch.tensor([7.6, 0.0, 0.0])   # 180° angle
    t = torch.stack([CA_0, CA_1, CA_2])
    chain_id = torch.zeros(3, dtype=torch.long)
    E = angle_energy(t, chain_id, k_angle=10.0, theta0=2.094)
    expected = 10.0 * (math.pi - 2.094) ** 2  # ≈ 10.974 kcal/mol
    assert abs(float(E) - expected) < 1e-3


def test_angle_energy_two_triplets():
    """Chain of 4: two triplets each contribute independently."""
    # Use same 120° geometry repeated for both triplets
    # Triplet 0-1-2 at 120°, triplet 1-2-3 at 180°
    CA_0 = torch.tensor([-3.8, 0.0, 0.0])
    CA_1 = torch.tensor([0.0,  0.0, 0.0])
    CA_2 = torch.tensor([1.9,  3.29, 0.0])
    # Extend CA_3 straight from CA_2 direction: angle at CA_2 = 180°
    direction = (CA_2 - CA_1) / (CA_2 - CA_1).norm()
    CA_3 = CA_2 + direction * 3.8
    t = torch.stack([CA_0, CA_1, CA_2, CA_3])
    chain_id = torch.zeros(4, dtype=torch.long)
    E = angle_energy(t, chain_id, k_angle=10.0, theta0=2.094)
    # First triplet ≈ 0, second triplet = 10*(pi-2.094)^2
    expected = 10.0 * (math.pi - 2.094) ** 2
    assert abs(float(E) - expected) < 1e-2


def test_angle_energy_two_chains_no_cross():
    """Two chains of 2 residues each: no triplets → energy = 0."""
    t = torch.randn(4, 3)
    chain_id = torch.tensor([0, 0, 1, 1])
    E = angle_energy(t, chain_id)
    assert float(E.abs()) < 1e-6
