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
    # Wide spacing → no MJ contacts, so w_mj=1.0 and w_mj=0.0 produce the same result
    angle_E = angle_energy(t, chain_id)
    assert abs(float(E_nomj) - float(angle_E)) < 1e-4
    assert abs(float(E_full) - float(angle_E)) < 1e-4


# ── Reweighting tests (Task 5) ────────────────────────────────────────────────
from unittest.mock import patch
from lsmd import transfer_modes as tm
import lsmd.cg_energy as cge


def test_reweight_boltzmann_uniform():
    """All frames equal energy → uniform weights, N_eff = F, not degenerate."""
    F, N = 20, 4
    # All UNK, widely spaced → E ≈ 0 for every frame
    traj = torch.zeros(F, N, 3)
    for i in range(F):
        for j in range(N):
            traj[i, j] = torch.tensor([float(j) * 50, float(i) * 50, 0.0])
    res_type = torch.ones(N, dtype=torch.long) * 20   # UNK
    chain_id = torch.zeros(N, dtype=torch.long)
    result = tm.reweight_boltzmann(traj, res_type, chain_id, kT=0.593, w_wca=0.0)
    assert result["weights"].std() < 1e-4
    assert abs(result["n_eff"] - F) < 0.5
    assert not result["degenerate"]


def test_reweight_boltzmann_degenerate():
    """Frame 0 energy -1000 kcal/mol → single dominant frame → degenerate=True."""
    F, N = 100, 3
    traj = torch.zeros(F, N, 3)
    res_type = torch.zeros(N, dtype=torch.long)
    chain_id = torch.zeros(N, dtype=torch.long)
    call_count = [0]
    def mock_energy(t, rt, ci, **kw):
        i = call_count[0]; call_count[0] += 1
        return torch.tensor(-1000.0) if i == 0 else torch.tensor(0.0)
    with patch.object(cge, "total_cg_energy", side_effect=mock_energy):
        result = tm.reweight_boltzmann(traj, res_type, chain_id, kT=0.593)
    assert result["degenerate"]
    assert result["n_eff"] < 0.1 * F


def test_resample_trajectory_shape():
    F, N = 80, 6
    traj = torch.randn(F, N, 3)
    weights = torch.ones(F) / F
    resampled = tm.resample_trajectory(traj, weights, n_samples=50)
    assert resampled.shape == (50, N, 3)


def test_resample_trajectory_concentrated_weights():
    """All weight on frame 0 → all resampled frames equal frame 0."""
    F, N = 20, 4
    traj = torch.randn(F, N, 3)
    weights = torch.zeros(F); weights[0] = 1.0
    resampled = tm.resample_trajectory(traj, weights, n_samples=10)
    assert (resampled - traj[0]).abs().max() < 1e-6


def test_wca_energy_importable_from_both_modules():
    from lsmd.cg_energy import _wca_energy as wca_cge
    from lsmd.transfer_eval import _wca_energy as wca_te
    assert wca_cge is wca_te          # same object (re-export, not a copy)
    t = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    chain_id = torch.zeros(3, dtype=torch.long)
    e = wca_cge(t, chain_id)
    assert torch.isfinite(e) and e.ndim == 0
