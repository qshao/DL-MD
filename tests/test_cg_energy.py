import math
import torch
from lsmd.cg_energy import angle_energy, mj_contact_energy, total_cg_energy, MJ_MATRIX


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


def test_mj_gly_gly_contact():
    """GLY(7)-GLY(7) pair at 6 Å, seq_sep=4 → energy = MJ_MATRIX[7,7]."""
    N = 5
    t = torch.zeros(N, 3)
    # Residues 1-3 far away (no contacts among themselves or with 0/4)
    t[1] = torch.tensor([100.0, 0.0, 0.0])
    t[2] = torch.tensor([200.0, 0.0, 0.0])
    t[3] = torch.tensor([300.0, 0.0, 0.0])
    t[4] = torch.tensor([6.0, 0.0, 0.0])   # GLY at index 0 and 4, dist=6 Å
    # res_type: GLY=7 at 0 and 4; UNK=20 at 1,2,3 (excluded from MJ)
    res_type = torch.tensor([7, 20, 20, 20, 7])
    chain_id = torch.zeros(N, dtype=torch.long)
    E = mj_contact_energy(t, res_type, chain_id)
    expected = float(MJ_MATRIX[7, 7])   # only one contact pair
    assert abs(float(E) - expected) < 1e-4


def test_mj_beyond_cutoff():
    """Pair at 9 Å (> cutoff 8 Å) → energy = 0."""
    N = 5
    t = torch.zeros(N, 3)
    t[1] = t[2] = t[3] = torch.tensor([500.0, 0.0, 0.0])
    t[4] = torch.tensor([9.1, 0.0, 0.0])  # dist > 8 Å
    res_type = torch.tensor([7, 20, 20, 20, 7])
    chain_id = torch.zeros(N, dtype=torch.long)
    E = mj_contact_energy(t, res_type, chain_id)
    assert float(E.abs()) < 1e-6


def test_mj_bonded_excluded():
    """Pair with seq_sep=2 (≤ 3) → energy = 0 even at 5 Å."""
    t = torch.zeros(3, 3)
    t[2] = torch.tensor([5.0, 0.0, 0.0])  # seq_sep(0,2)=2, dist=5 Å
    res_type = torch.tensor([7, 7, 7])
    chain_id = torch.zeros(3, dtype=torch.long)
    E = mj_contact_energy(t, res_type, chain_id)
    assert float(E.abs()) < 1e-6


def test_mj_unk_excluded():
    """UNK residue (index 20) is excluded from all contacts."""
    N = 5
    t = torch.zeros(N, 3)
    t[4] = torch.tensor([6.0, 0.0, 0.0])
    res_type = torch.tensor([20, 20, 20, 20, 7])  # index 0 is UNK
    chain_id = torch.zeros(N, dtype=torch.long)
    E = mj_contact_energy(t, res_type, chain_id)
    assert float(E.abs()) < 1e-6


def test_mj_matrix_is_symmetric():
    """MJ_MATRIX must be symmetric: MJ[i,j] == MJ[j,i]."""
    diff = (MJ_MATRIX - MJ_MATRIX.T).abs().max()
    assert diff < 1e-5


def test_mj_diagonal_negative():
    """All diagonal entries should be negative (self-contacts are favorable)."""
    diag = MJ_MATRIX.diagonal()
    assert (diag < 0).all()


def test_total_cg_energy_w_mj_zero():
    """With w_mj=0, total_cg_energy = angle + wca only (no MJ)."""
    N = 5
    t = torch.zeros(N, 3)
    for i in range(N):
        t[i, 0] = float(i) * 20.0   # wide spacing → WCA=0
    res_type = torch.zeros(N, dtype=torch.long)
    chain_id = torch.zeros(N, dtype=torch.long)
    E_full  = total_cg_energy(t, res_type, chain_id, w_mj=1.0, w_wca=0.0)
    E_nomj  = total_cg_energy(t, res_type, chain_id, w_mj=0.0, w_wca=0.0)
    # Both should equal the angle energy
    angle_E = angle_energy(t, chain_id)
    assert abs(float(E_nomj) - float(angle_E)) < 1e-4
