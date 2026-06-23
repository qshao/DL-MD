import torch

# ── Excluded-volume (WCA) energy ──────────────────────────────────────────────

def _wca_energy(t_pred, chain_id, sigma=4.5, eps=0.3):
    """Weeks–Chandler–Andersen excluded-volume energy for non-bonded CA pairs.

    Only applies to pairs with sequence separation > 2 within the same chain,
    and all cross-chain pairs. Parameterisation from CG-MD literature (CA–CA
    contact radius ~4.5 Å, well depth ~0.3 kcal/mol ≈ 0.5 kT at 300 K).

    Args:
        t_pred:   [N, 3] predicted CA positions (differentiable).
        chain_id: [N] long, chain assignment.
        sigma:    WCA diameter (Å). Cutoff r_cut = 2^(1/6) * sigma ≈ 5.05 Å.
        eps:      Well depth (kcal/mol).

    Returns:
        Scalar energy (differentiable w.r.t. t_pred).
    """
    N = t_pred.shape[0]
    # Use manual pairwise distances instead of torch.cdist so that
    # create_graph=True (second-order gradients) works for score-matching.
    diff = t_pred.unsqueeze(0) - t_pred.unsqueeze(1)         # [N, N, 3]
    d = (diff * diff).sum(-1).clamp_min(1e-8).sqrt()         # [N, N]
    idx = torch.arange(N, device=t_pred.device)
    seq_sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()    # [N, N]
    same_chain = (chain_id.unsqueeze(0) == chain_id.unsqueeze(1))
    # Bonded: same chain AND seq_sep ≤ 2 (already handled by SHAKE)
    bonded = same_chain & (seq_sep <= 2)
    non_bonded = ~bonded & (seq_sep != 0)                    # exclude self

    r_cut = (2.0 ** (1.0 / 6.0)) * sigma                    # ≈ 5.05 Å
    # 3.0 Å is still extremely unphysical for Cα–Cα; clamping here bounds the
    # WCA gradient (∝ r^-13) so backward doesn't overflow during energy-loss training.
    r = d[non_bonded].clamp_min(3.0)
    sr6 = (sigma / r).pow(6)
    v_wca = 4.0 * eps * (sr6 * sr6 - sr6) + eps             # [M]
    in_range = (r < r_cut).to(v_wca.dtype)
    return (v_wca * in_range).sum()


# ── Angle energy ──────────────────────────────────────────────────────────────

def angle_energy(t: torch.Tensor,
                 chain_id: torch.Tensor,
                 k_angle: float = 10.0,
                 theta0: float = 2.094) -> torch.Tensor:
    """Harmonic CA-CA-CA angle energy.

    Args:
        t:        [N, 3] CA positions.
        chain_id: [N] long chain assignment.
        k_angle:  force constant in kcal/mol/rad².
        theta0:   equilibrium angle in radians (2.094 rad = 120°).

    Returns:
        Scalar energy in kcal/mol.
    """
    E = t.new_zeros(())
    for c in chain_id.unique():
        mask = (chain_id == c).nonzero(as_tuple=True)[0]
        if mask.shape[0] < 3:
            continue
        pos = t[mask]                   # [nc, 3]
        v1 = pos[:-2] - pos[1:-1]      # [nc-2, 3]
        v2 = pos[2:]  - pos[1:-1]      # [nc-2, 3]
        norms1 = v1.norm(dim=-1).clamp_min(1e-8)
        norms2 = v2.norm(dim=-1).clamp_min(1e-8)
        cos_theta = (v1 * v2).sum(-1) / (norms1 * norms2)
        # Use atan2 for numerical stability at extreme angles
        sin_cross = torch.cross(v1, v2, dim=-1).norm(dim=-1) / (norms1 * norms2)
        theta = torch.atan2(sin_cross, cos_theta)
        E = E + (k_angle * (theta - theta0) ** 2).sum()
    return E


# ── MJ statistical contact potential ──────────────────────────────────────────
#
# Source: Miyazawa & Jernigan 1996, J. Mol. Biol. 256:623-644, Table 3.
# Original values are in kT units at 298 K; multiplied by 0.592 kcal/mol here.
# Matrix is indexed by lsmd.vocab.CANONICAL residue order (see CANONICAL_TO_PAPER
# mapping below).

def _build_mj_matrix() -> torch.Tensor:
    # Paper residue order: CYS MET PHE ILE LEU VAL TRP TYR ALA GLY THR SER ASN GLN ASP GLU HIS ARG LYS PRO
    # Lower-triangle (row i contains values for paper residues 0..i)
    _lower = [
        [-5.44],
        [-4.99, -5.46],
        [-5.80, -5.74, -7.26],
        [-5.50, -5.53, -6.84, -5.78],
        [-5.83, -6.02, -7.28, -6.67, -5.83],
        [-4.96, -4.91, -6.29, -5.96, -5.83, -5.52],
        [-6.47, -6.34, -9.03, -7.46, -7.68, -6.48, -9.73],
        [-6.20, -6.05, -7.80, -6.98, -7.08, -6.29, -8.80, -6.36],
        [-3.57, -3.94, -4.81, -4.91, -4.96, -4.04, -5.06, -4.66, -2.72],
        [-3.16, -3.39, -4.13, -3.78, -4.16, -3.38, -4.65, -4.13, -2.31, -3.02],
        [-3.11, -3.40, -4.28, -4.21, -4.34, -3.71, -4.70, -4.18, -2.78, -2.88, -3.64],
        [-2.86, -3.05, -4.02, -3.52, -3.92, -3.05, -4.20, -4.00, -2.36, -2.64, -2.99, -3.05],
        [-2.59, -3.07, -4.20, -3.76, -3.74, -3.14, -4.53, -3.75, -2.17, -2.83, -3.17, -2.73, -3.54],
        [-3.07, -3.11, -4.66, -4.19, -4.21, -3.49, -5.49, -4.31, -2.57, -2.93, -3.01, -2.57, -3.07, -4.27],
        [-2.57, -2.89, -4.43, -3.52, -3.28, -2.97, -4.48, -3.62, -1.95, -2.42, -2.48, -2.37, -2.96, -3.10, -2.30],
        [-2.89, -2.92, -4.20, -3.65, -3.31, -3.05, -4.66, -3.82, -2.01, -2.44, -2.69, -2.27, -2.84, -3.07, -3.20, -2.89],
        [-3.60, -3.98, -4.77, -4.63, -4.37, -3.90, -5.39, -4.85, -2.41, -3.01, -3.23, -2.87, -3.11, -3.62, -3.16, -3.06, -4.77],
        [-2.57, -3.12, -4.77, -4.34, -4.26, -3.63, -5.56, -4.50, -2.27, -2.64, -2.88, -2.42, -2.59, -3.33, -2.87, -2.99, -3.98, -3.98],
        [-1.95, -2.48, -3.36, -3.37, -3.48, -3.05, -3.82, -3.36, -1.62, -1.72, -2.03, -1.64, -2.14, -2.57, -2.48, -2.57, -2.85, -2.69, -3.37],
        [-3.07, -3.45, -4.25, -4.04, -4.20, -3.32, -4.65, -4.10, -2.03, -2.48, -2.75, -2.53, -2.84, -3.23, -2.41, -2.90, -3.73, -3.44, -2.40, -4.93],
    ]
    m = torch.zeros(20, 20)
    for i, row in enumerate(_lower):
        for j, val in enumerate(row):
            m[i, j] = val
            m[j, i] = val
    # Reindex: paper order → CANONICAL order
    # CANONICAL: ALA=0 ARG=1 ASN=2 ASP=3 CYS=4 GLN=5 GLU=6 GLY=7
    #            HIS=8 ILE=9 LEU=10 LYS=11 MET=12 PHE=13 PRO=14
    #            SER=15 THR=16 TRP=17 TYR=18 VAL=19
    # Paper: CYS(p0) MET(p1) PHE(p2) ILE(p3) LEU(p4) VAL(p5) TRP(p6) TYR(p7)
    #        ALA(p8) GLY(p9) THR(p10) SER(p11) ASN(p12) GLN(p13) ASP(p14) GLU(p15)
    #        HIS(p16) ARG(p17) LYS(p18) PRO(p19)
    canon_to_paper = torch.tensor([8, 17, 12, 14, 0, 13, 15, 9,
                                   16,  3,  4, 18, 1,  2, 19, 11,
                                   10,  6,  7,  5])
    m = m[canon_to_paper][:, canon_to_paper]
    return m * 0.592   # kT(298K) → kcal/mol


MJ_MATRIX: torch.Tensor = _build_mj_matrix()


def mj_contact_energy(t: torch.Tensor,
                      res_type: torch.Tensor,
                      chain_id: torch.Tensor,
                      cutoff: float = 8.0) -> torch.Tensor:
    """Miyazawa-Jernigan statistical contact energy.

    Sums MJ[res_i, res_j] for all pairs (i<j) where:
      - CA-CA distance < cutoff
      - |i - j| > 3  (exclude bonded neighbors)
      - neither residue is UNK (index 20)

    Args:
        t:        [N, 3] CA positions.
        res_type: [N] long, CANONICAL residue indices (0-19; 20=UNK excluded).
        chain_id: [N] long (unused beyond UNK filter - seq_sep is global index).
        cutoff:   contact distance in Angstroms (default 8.0).

    Returns:
        Scalar energy in kcal/mol (negative = favorable).
    """
    N = t.shape[0]
    idx = torch.arange(N, device=t.device)

    diff = t.unsqueeze(0) - t.unsqueeze(1)          # [N, N, 3]
    dist2 = (diff * diff).sum(-1)                    # [N, N]

    upper_tri  = idx.unsqueeze(1) < idx.unsqueeze(0)
    in_contact = dist2 < cutoff * cutoff
    seq_sep_ok = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() > 3
    not_unk    = (res_type < 20).unsqueeze(1) & (res_type < 20).unsqueeze(0)

    mask = upper_tri & in_contact & seq_sep_ok & not_unk   # [N, N]

    ri = res_type.clamp(max=19)
    energies = MJ_MATRIX.to(t.device)[ri.unsqueeze(1), ri.unsqueeze(0)]  # [N, N]
    return (energies * mask.float()).sum()


# ── Combined CG energy ────────────────────────────────────────────────────────

def total_cg_energy(t: torch.Tensor,
                    res_type: torch.Tensor,
                    chain_id: torch.Tensor,
                    *,
                    wca_sigma: float = 4.5,
                    wca_eps: float = 0.3,
                    k_angle: float = 10.0,
                    theta0: float = 2.094,
                    mj_cutoff: float = 8.0,
                    w_wca: float = 1.0,
                    w_angle: float = 1.0,
                    w_mj: float = 1.0) -> torch.Tensor:
    """Sum of WCA + angle + MJ contact energies (kcal/mol).

    Args:
        t:         [N, 3] CA positions.
        res_type:  [N] long CANONICAL indices.
        chain_id:  [N] long chain assignment.
        w_wca/w_angle/w_mj: per-term weights (default 1.0).

    Returns:
        Scalar energy in kcal/mol.
    """
    E = t.new_zeros(())
    if w_wca != 0.0:
        E = E + w_wca  * _wca_energy(t, chain_id, sigma=wca_sigma, eps=wca_eps) / 2
    if w_angle != 0.0:
        E = E + w_angle * angle_energy(t, chain_id, k_angle=k_angle, theta0=theta0)
    if w_mj != 0.0:
        E = E + w_mj   * mj_contact_energy(t, res_type, chain_id, cutoff=mj_cutoff)
    return E
