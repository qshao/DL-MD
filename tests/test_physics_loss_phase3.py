import torch
from lsmd.physics_loss import energy_match_loss
from lsmd.learned_energy import LearnedCGEnergy
from lsmd import geometry as g


def _identity_frames(N):
    R = torch.eye(3).expand(N, 3, 3).contiguous()
    t = torch.zeros(N, 3)
    # spread CA along x at 3.8 Å so the reference geometry is physical
    t[:, 0] = torch.arange(N).float() * 3.8
    return R, t


def test_energy_match_zero_when_prediction_is_physical():
    N = 10
    R, t = _identity_frames(N)
    res_type = torch.zeros(N, dtype=torch.long)
    chain_id = torch.zeros(N, dtype=torch.long)
    protein_id = torch.zeros(N, dtype=torch.long)
    energy = LearnedCGEnergy()
    # zero update → predicted frame == current physical frame → low energy
    u = torch.zeros(N, 6)
    u_cut = 1e6                          # ceiling far above any physical energy
    loss = energy_match_loss(R, t, u, res_type, protein_id, chain_id, energy,
                             u_cut=u_cut)
    assert float(loss) == 0.0


def test_energy_match_positive_for_clashing_prediction():
    N = 10
    R, t = _identity_frames(N)
    res_type = torch.zeros(N, dtype=torch.long)
    chain_id = torch.zeros(N, dtype=torch.long)
    protein_id = torch.zeros(N, dtype=torch.long)
    energy = LearnedCGEnergy()
    # collapse all residues toward the origin via a large negative-x translation
    u = torch.zeros(N, 6)
    u[:, 0] = -t[:, 0]                   # local_trans cancels the x spread → clash
    u_cut = -10.0                        # low ceiling so the hinge activates
    loss = energy_match_loss(R, t, u, res_type, protein_id, chain_id, energy,
                             u_cut=u_cut)
    assert float(loss) > 0.0
