import datetime
import math
import torch
from lsmd import decoder as dec


def _ca(x):
    """Return CA coords [.,P,3]. Accepts a CA point cloud [.,P,3] (used as-is)
    or a full backbone tensor [.,N,4,3] (CA = atom index 1 extracted)."""
    return x if x.dim() == 3 else x[:, :, 1, :]


def geometry_metrics(atoms):
    """Compute CA bond length, peptide bond violation, and clash count.

    Args:
        atoms: Atom coordinates [N, 4, 3]

    Returns:
        dict with keys: ca_bond_mean, peptide_violation, clash_count
    """
    ca = atoms[:, 1, :]
    ca_bonds = (ca[1:] - ca[:-1]).norm(dim=-1)
    d = torch.cdist(ca, ca)
    n = ca.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=ca.device)
    for i in range(n - 1):
        mask[i, i + 1] = mask[i + 1, i] = False
    clash_count = ((d < 2.0) & mask).sum().item() / 2
    return {
        "ca_bond_mean": ca_bonds.mean().item(),
        "peptide_violation": dec.peptide_bond_violation(atoms).item(),
        "clash_count": clash_count,
    }


def diversity(atoms_K):
    """Compute mean pairwise CA-RMSD across K structures.

    Args:
        atoms_K: Atom coordinates [K, N, 4, 3]

    Returns:
        float: Mean pairwise CA-RMSD
    """
    ca = atoms_K[:, :, 1, :]  # [K, N, 3]
    K = ca.shape[0]
    total, count = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            total += (ca[i] - ca[j]).norm(dim=-1).pow(2).mean().sqrt().item()
            count += 1
    return total / max(count, 1)


def baseline_copy(R_t, t_t, K):
    """Zero-update baseline (copy of current state).

    Args:
        R_t: Target rotations [N, 3, 3]
        t_t: Target translations [N, 3]
        K: Number of samples

    Returns:
        u_samples [K, N, 6]: All zeros (no update)
    """
    n = R_t.shape[0]
    return torch.zeros(K, n, 6)


def baseline_noise(R_t, t_t, K, sigma):
    """Random Gaussian noise baseline.

    Args:
        R_t: Target rotations [N, 3, 3]
        t_t: Target translations [N, 3]
        K: Number of samples
        sigma: Noise standard deviation

    Returns:
        u_samples [K, N, 6]: Random Gaussian noise scaled by sigma
    """
    n = R_t.shape[0]
    return torch.randn(K, n, 6) * sigma


def backbone_torsions(atoms):
    """Compute backbone dihedral angles for interior residues.

    Args:
        atoms: [N, 4, 3] — atom order per residue: N, CA, C, O

    Returns:
        phi: [N-2] tensor in (-π, π]
        psi: [N-2] tensor in (-π, π]
    """
    def _dihedral(a, b, c, d):
        b1 = b - a; b2 = c - b; b3 = d - c
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        b2n = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        m1 = torch.cross(n1, b2n, dim=-1)
        return torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    # phi_i = dihedral(C(i-1), N(i), CA(i), C(i))  for i = 1..N-2
    phi = _dihedral(
        atoms[:-2, 2, :], atoms[1:-1, 0, :],
        atoms[1:-1, 1, :], atoms[1:-1, 2, :]
    )
    # psi_i = dihedral(N(i), CA(i), C(i), N(i+1))  for i = 1..N-2
    psi = _dihedral(
        atoms[1:-1, 0, :], atoms[1:-1, 1, :],
        atoms[1:-1, 2, :], atoms[2:, 0, :]
    )
    return phi, psi


def _batch_torsions(atoms_batch):
    """Vectorised backbone torsions over a batch of structures.

    Args:
        atoms_batch: [K, N, 4, 3]

    Returns:
        (phi, psi) each [K*(N-2)]
    """
    def _dihedral(a, b, c, d):  # each [K, N-2, 3]
        b1 = b - a; b2 = c - b; b3 = d - c
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        b2n = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        m1 = torch.cross(n1, b2n, dim=-1)
        return torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    phi = _dihedral(
        atoms_batch[:, :-2, 2, :], atoms_batch[:, 1:-1, 0, :],
        atoms_batch[:, 1:-1, 1, :], atoms_batch[:, 1:-1, 2, :]
    ).reshape(-1)
    psi = _dihedral(
        atoms_batch[:, 1:-1, 0, :], atoms_batch[:, 1:-1, 1, :],
        atoms_batch[:, 1:-1, 2, :], atoms_batch[:, 2:,   0, :]
    ).reshape(-1)
    return phi, psi


def ramachandran_js(atoms_model, atoms_md, bins=36):
    """Jensen-Shannon divergence between Ramachandran distributions.

    Pools φ,ψ from all K×(N-2) model angles and M×(N-2) MD angles, builds
    36×36 histograms over [-π, π]², and computes JS divergence.

    Args:
        atoms_model: [K, N, 4, 3]
        atoms_md:    [M, N, 4, 3]
        bins:        Grid resolution (10° at bins=36).

    Returns:
        JS divergence in [0, 1].  0 = identical, 1 = disjoint.
    """
    def _hist2d(phi, psi, bins):
        lo, hi = -torch.pi, torch.pi
        phi_b = ((phi - lo) / (hi - lo) * bins).long().clamp(0, bins - 1)
        psi_b = ((psi - lo) / (hi - lo) * bins).long().clamp(0, bins - 1)
        idx = phi_b * bins + psi_b
        h = torch.zeros(bins * bins, device=phi.device)
        h.scatter_add_(0, idx, torch.ones(len(phi), device=phi.device))
        h = h + 1e-8
        return h / h.sum()

    phi_m, psi_m = _batch_torsions(atoms_model)
    phi_d, psi_d = _batch_torsions(atoms_md)
    p = _hist2d(phi_m, psi_m, bins)
    q = _hist2d(phi_d, psi_d, bins)
    mix = 0.5 * (p + q)
    js = 0.5 * (p * torch.log(p / mix)).sum() + \
         0.5 * (q * torch.log(q / mix)).sum()
    return (js / math.log(2)).clamp(0.0, 1.0).item()


def pca_js(atoms_model, atoms_md, n_components=2, bins=20):
    """Jensen-Shannon divergence of 2D PCA density between two ensembles.

    Fits PCA on the MD CA ensemble, projects both ensembles, and computes
    JS divergence of the 2D density.

    Args:
        atoms_model:  [K, N, 4, 3]
        atoms_md:     [M, N, 4, 3]
        n_components: Number of PCA components (only first 2 used for JS).
        bins:         Histogram bins per axis.

    Returns:
        dict with keys:
            js:            JS divergence ∈ [0, 1]
            var_explained: [float, float] — per-component variance fraction
    """
    ca_model = _ca(atoms_model).float()   # [K, P, 3]
    ca_md    = _ca(atoms_md).float()      # [M, P, 3]
    K, M = ca_model.shape[0], ca_md.shape[0]

    mu = ca_md.mean(0)                            # [N, 3]
    cm = (ca_model - mu).reshape(K, -1)
    cd = (ca_md    - mu).reshape(M, -1)

    _, s_vals, Vt = torch.linalg.svd(cd, full_matrices=False)
    n_components = min(n_components, Vt.shape[0], cd.shape[1])
    total_var = (s_vals ** 2).sum().clamp_min(1e-8)
    var_explained = [(s_vals[i] ** 2 / total_var).item()
                     for i in range(min(n_components, len(s_vals)))]
    while len(var_explained) < 2:
        var_explained.append(0.0)

    V = Vt[:n_components].T                      # [N*3, n_components]
    pm = cm @ V                                   # [K, n_components]
    pd = cd @ V                                   # [M, n_components]

    # Pad to at least 2 columns so proj[:,1] never raises IndexError
    if pm.shape[1] < 2:
        pm = torch.cat([pm, torch.zeros(K, 2 - pm.shape[1], device=pm.device)], dim=1)
        pd = torch.cat([pd, torch.zeros(M, 2 - pd.shape[1], device=pd.device)], dim=1)

    lo = pd[:, :2].min(0).values
    hi = pd[:, :2].max(0).values
    span = (hi - lo).clamp_min(1e-8)

    def _idx(proj):
        xb = ((proj[:, 0] - lo[0]) / span[0] * bins).long().clamp(0, bins - 1)
        yb = ((proj[:, 1] - lo[1]) / span[1] * bins).long().clamp(0, bins - 1)
        return (xb * bins + yb).cpu()

    def _hist(idx, n):
        h = torch.zeros(bins * bins)
        h.scatter_add_(0, idx, torch.ones(n))
        h = h + 1e-8
        return h / h.sum()

    p = _hist(_idx(pm), K)
    q = _hist(_idx(pd), M)
    mix = 0.5 * (p + q)
    js = (0.5 * (p * torch.log(p / mix)).sum() +
          0.5 * (q * torch.log(q / mix)).sum()) / math.log(2)
    return {"js": js.clamp(0.0, 1.0).item(), "var_explained": var_explained}


def ensemble_recall(atoms_model, atoms_md, r_ang=2.0):
    """Fraction of MD frames covered by at least one model sample within r_ang Å.

    Measures whether the model reproduces all conformational states the MD visits.
    recall = 1.0 → no mode collapse; recall < 0.8 → model missing states.

    Args:
        atoms_model: [K, N, 4, 3]
        atoms_md:    [M, N, 4, 3]
        r_ang:       CA-RMSD coverage radius in Angstrom.

    Returns:
        float in [0, 1].
    """
    ca_model = _ca(atoms_model)   # [K, P, 3]
    ca_md    = _ca(atoms_md)      # [M, P, 3]
    M = ca_md.shape[0]
    covered = 0
    for m_idx in range(M):
        diff = ca_model - ca_md[m_idx].unsqueeze(0)    # [K, P, 3]
        rmsd = diff.norm(dim=-1).pow(2).mean(dim=-1).sqrt()   # [K]
        if rmsd.min().item() < r_ang:
            covered += 1
    return covered / M


def ensemble_novelty(atoms_model, atoms_md, r_ang=2.0):
    """Fraction of model samples with no MD neighbor within r_ang Å.

    Measures generalization beyond the training trajectory.
    High novelty + good geometry = beneficial extrapolation.
    High novelty + bad geometry  = hallucination.

    Args:
        atoms_model: [K, N, 4, 3]
        atoms_md:    [M, N, 4, 3]
        r_ang:       CA-RMSD novelty radius in Angstrom.

    Returns:
        float in [0, 1].
    """
    ca_model = _ca(atoms_model)   # [K, P, 3]
    ca_md    = _ca(atoms_md)      # [M, P, 3]
    K = ca_model.shape[0]
    novel = 0
    for k_idx in range(K):
        diff = ca_md - ca_model[k_idx].unsqueeze(0)    # [M, P, 3]
        rmsd = diff.norm(dim=-1).pow(2).mean(dim=-1).sqrt()   # [M]
        if rmsd.min().item() >= r_ang:
            novel += 1
    return novel / K


def ca_geometry(ca):
    """Sequential CA-CA bond statistics and clash count for one CA trace.

    Args:
        ca: CA coordinates [P, 3]

    Returns:
        dict: ca_bond_mean, ca_bond_min, ca_bond_max (Å), clash_count
              (non-adjacent CA pairs closer than 2.0 Å).
    """
    bonds = (ca[1:] - ca[:-1]).norm(dim=-1)
    d = torch.cdist(ca, ca)
    n = ca.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=ca.device)
    for i in range(n - 1):
        mask[i, i + 1] = mask[i + 1, i] = False
    clash_count = ((d < 2.0) & mask).sum().item() / 2
    return {
        "ca_bond_mean": bonds.mean().item(),
        "ca_bond_min": bonds.min().item(),
        "ca_bond_max": bonds.max().item(),
        "clash_count": clash_count,
    }


def _pairwise_dists(ca):
    """Pooled upper-triangle CA-CA distances over an ensemble [K,P,3] → 1-D."""
    K, P, _ = ca.shape
    iu = torch.triu_indices(P, P, offset=1)
    d = torch.cdist(ca, ca)                 # [K,P,P]
    return d[:, iu[0], iu[1]].reshape(-1)   # [K * P(P-1)/2]


def _hist_js(a, b, bins, lo=None, hi=None):
    """JS divergence (bits) between two 1-D samples via shared-range histograms."""
    if lo is None:
        lo = torch.min(a.min(), b.min())
    if hi is None:
        hi = torch.max(a.max(), b.max())
    span = (hi - lo).clamp_min(1e-8)

    def _h(x):
        idx = ((x - lo) / span * bins).long().clamp(0, bins - 1)
        h = torch.zeros(bins, device=x.device)
        h.scatter_add_(0, idx, torch.ones_like(x))
        h = h + 1e-8
        return h / h.sum()

    p, q = _h(a), _h(b)
    mix = 0.5 * (p + q)
    js = 0.5 * (p * torch.log(p / mix)).sum() + 0.5 * (q * torch.log(q / mix)).sum()
    return (js / math.log(2)).clamp(0.0, 1.0).item()


def distance_matrix_js(ca_model, ca_md, bins=30):
    """JS divergence between pooled pairwise CA-CA distance distributions.

    Captures whether the model reproduces the overall conformational geometry
    (contact distances) of the MD ensemble.

    Args:
        ca_model: [K, P, 3]
        ca_md:    [M, P, 3]
        bins:     histogram bins.

    Returns:
        JS divergence in [0, 1]. 0 = identical distance distributions.
    """
    a = _pairwise_dists(ca_model)
    b = _pairwise_dists(ca_md)
    return _hist_js(a, b, bins)


def rmsf_profile(ca_model, ca_md):
    """Per-residue CA positional fluctuation (RMSF) for both ensembles.

    Args:
        ca_model: [K, P, 3]
        ca_md:    [M, P, 3]

    Returns:
        dict: model [P], md [P] (per-residue std magnitude, Å),
              corr (Pearson correlation of the two profiles).
    """
    def _rmsf(ca):
        mu = ca.mean(0, keepdim=True)               # [1,P,3]
        return (ca - mu).pow(2).sum(-1).mean(0).sqrt()   # [P]

    rm = _rmsf(ca_model)
    rd = _rmsf(ca_md)
    rmc = rm - rm.mean()
    rdc = rd - rd.mean()
    denom = (rmc.norm() * rdc.norm()).clamp_min(1e-8)
    corr = (rmc * rdc).sum() / denom
    return {"model": rm.tolist(), "md": rd.tolist(), "corr": corr.item()}


def displacement_js(disp_model, disp_md, bins=30):
    """JS divergence between two displacement-magnitude distributions.

    disp_* are per-sample RMSD magnitudes (Å): for the model, ‖Δ‖ of sampled
    displacements; for MD, per-pair ‖Δ‖ at the chosen lag. Separates the
    fluctuation bulk (small ‖Δ‖) from the transition tail (large ‖Δ‖).

    Args:
        disp_model: 1-D tensor of model displacement magnitudes.
        disp_md:    1-D tensor of MD displacement magnitudes.
        bins:       histogram bins.

    Returns:
        dict: js (in [0,1]), model_mean, md_mean.
    """
    js = _hist_js(disp_model, disp_md, bins)
    return {
        "js": js,
        "model_mean": disp_model.mean().item(),
        "md_mean": disp_md.mean().item(),
    }


# ---------------------------------------------------------------------------
# 4-bead physical validity check
# ---------------------------------------------------------------------------

# Bond ranges [lo, hi] in Å for each bond type in the 4-bead model
_4B_BOND_RANGES = {
    "N-CA":  (1.35, 1.62),
    "CA-C":  (1.40, 1.65),
    "CA-CB": (1.38, 1.68),
    "C-N":   (1.20, 1.52),   # peptide bond
}
_4B_CLASH_DIST = 2.0   # Å — minimum non-bonded heavy-atom distance


def check_4bead_conformation(beads, gly_mask=None):
    """Check whether a 4-bead (N, CA, C, CB) conformation is physically meaningful.

    Checks four bond types:
        N−CA  (ideal 1.46 Å)
        CA−C  (ideal 1.52 Å)
        CA−CB (ideal 1.52 Å; skipped for Gly if gly_mask provided)
        C−N   (peptide bond, ideal 1.33 Å)

    and non-bonded steric clashes between all atom pairs not connected by one
    of the above bonds (threshold 2.0 Å).

    Args:
        beads    : [P, 4, 3] bead coordinates in Å, order (N, CA, C, CB)
        gly_mask : optional bool tensor [P], True for Gly residues (no real CB)

    Returns:
        dict with keys:
            valid           bool
            bond_ok         bool
            clash_free      bool
            rg_ok           bool  (Rg of CA vs Flory scaling)
            n_bond_violations int
            n_clashes       int
            bond_violations list  of (res_i, res_j, bond_type, dist_A)
            clashes         list  of (atom_i_flat, atom_j_flat, dist_A)
            rg_A            float
            rg_expected_A   float
    """
    P = beads.shape[0]
    device = beads.device
    N_a, CA, C_a, CB = beads[:, 0], beads[:, 1], beads[:, 2], beads[:, 3]

    bond_violations = []

    def _check_bonds(a, b, name, res_a, res_b):
        dists = (b - a).norm(dim=-1)
        lo, hi = _4B_BOND_RANGES[name]
        bad = ((dists < lo) | (dists > hi)).nonzero(as_tuple=False).squeeze(1)
        for k in bad:
            bond_violations.append((int(res_a[k]), int(res_b[k]),
                                    name, round(dists[k].item(), 3)))

    r = torch.arange(P, device=device)
    _check_bonds(N_a,  CA,       "N-CA",  r,    r)
    _check_bonds(CA,   C_a,      "CA-C",  r,    r)
    _check_bonds(C_a[:-1], N_a[1:], "C-N", r[:-1], r[1:])
    # CA-CB: skip Gly
    if gly_mask is None:
        _check_bonds(CA, CB, "CA-CB", r, r)
    else:
        not_gly = ~gly_mask
        _check_bonds(CA[not_gly], CB[not_gly], "CA-CB",
                     r[not_gly], r[not_gly])
    bond_ok = len(bond_violations) == 0

    # Clashes: build flat [4P, 3] and exclude bonded pairs
    n_atoms = P * 4
    flat = beads.reshape(n_atoms, 3)

    # Bonded pairs (flat indices): N-CA, CA-C, CA-CB, C-N(next)
    bi_list, bj_list = [], []
    for res in range(P):
        bi_list += [res*4, res*4+1, res*4+1]    # N-CA, CA-C, CA-CB
        bj_list += [res*4+1, res*4+2, res*4+3]
        if res < P - 1:
            bi_list.append(res*4+2)              # C-N(next)
            bj_list.append((res+1)*4)
    # Gly: remove CA-CB bonds
    if gly_mask is not None:
        pairs_filtered = [(i, j) for i, j in zip(bi_list, bj_list)
                          if not (j == (i//4)*4+3 and gly_mask[i//4])]
        bi_list = [p[0] for p in pairs_filtered]
        bj_list = [p[1] for p in pairs_filtered]

    bonded_adj = torch.zeros(n_atoms, n_atoms, dtype=torch.bool, device=device)
    if bi_list:
        bi_t = torch.tensor(bi_list, device=device)
        bj_t = torch.tensor(bj_list, device=device)
        bonded_adj[bi_t, bj_t] = True
        bonded_adj[bj_t, bi_t] = True

    ii, jj = torch.triu_indices(n_atoms, n_atoms, offset=1, device=device)
    nb_mask = ~bonded_adj[ii, jj]
    nb_i, nb_j = ii[nb_mask], jj[nb_mask]

    dists_nb = (flat[nb_i] - flat[nb_j]).norm(dim=-1)
    clash_mask = dists_nb < _4B_CLASH_DIST
    clashes = [
        (int(nb_i[k]), int(nb_j[k]), round(dists_nb[k].item(), 3))
        for k in clash_mask.nonzero(as_tuple=False).squeeze(1)
    ]
    clash_free = len(clashes) == 0

    # Rg on CA positions (same Flory scaling as before)
    centroid = CA.mean(0)
    rg = ((CA - centroid).pow(2).sum(-1).mean()).sqrt().item()
    rg_expected = 2.2 * (P ** 0.38)
    rg_ok = (0.5 * rg_expected) <= rg <= (2.0 * rg_expected)

    return {
        "valid":             bond_ok and clash_free and rg_ok,
        "bond_ok":           bond_ok,
        "clash_free":        clash_free,
        "rg_ok":             rg_ok,
        "n_bond_violations": len(bond_violations),
        "n_clashes":         len(clashes),
        "bond_violations":   bond_violations,
        "clashes":           clashes,
        "rg_A":              round(rg, 3),
        "rg_expected_A":     round(rg_expected, 3),
    }


def _build_4bead_bond_tensors(P, gly_mask, device):
    """Pre-build bond index tensors and non-bonded pair indices for a P-residue
    4-bead system.  Returns (bond_i, bond_j, bond_targets, nb_i, nb_j) as tensors.
    Designed to be called once and reused across many minimize_energy_4bead calls."""
    # Target distances (Å) for each bond type
    TARGETS = {"N-CA": 1.458, "CA-C": 1.525, "CA-CB": 1.521, "C-N": 1.329}
    bi, bj, bt = [], [], []
    for r in range(P):
        bi += [r*4+0, r*4+1, r*4+1]          # N-CA, CA-C, CA-CB
        bj += [r*4+1, r*4+2, r*4+3]
        bt += [TARGETS["N-CA"], TARGETS["CA-C"], TARGETS["CA-CB"]]
        if r < P - 1:
            bi.append(r*4+2); bj.append((r+1)*4); bt.append(TARGETS["C-N"])
    # Remove CA-CB for Gly
    if gly_mask is not None:
        keep = [(i, j, t) for i, j, t in zip(bi, bj, bt)
                if not (j == (i//4)*4+3 and gly_mask[i//4])]
        bi, bj, bt = zip(*keep) if keep else ([], [], [])

    bond_i = torch.tensor(bi, dtype=torch.long,  device=device)
    bond_j = torch.tensor(bj, dtype=torch.long,  device=device)
    bond_t = torch.tensor(bt, dtype=torch.float32, device=device)

    n = P * 4
    adj = torch.zeros(n, n, dtype=torch.bool, device=device)
    if len(bi):
        adj[bond_i, bond_j] = True
        adj[bond_j, bond_i] = True
    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    nb_mask = ~adj[ii, jj]
    return bond_i, bond_j, bond_t, ii[nb_mask], jj[nb_mask]


# ---------------------------------------------------------------------------
# Energy minimization
# ---------------------------------------------------------------------------

def minimize_energy(ca, bond_target=3.8, clash_dist=3.0,
                    k_bond=10.0, k_clash=1.0, n_steps=100):
    """Minimize a CA pseudo-energy to simultaneously fix bond lengths and clashes.

    Minimizes:
        E = k_bond  × Σ_{consecutive}  (|r_{i+1} − r_i| − bond_target)²
          + k_clash × Σ_{non-adjacent} max(0, clash_dist − |r_i − r_j|)²

    using L-BFGS with strong Wolfe line search.  Both bond and clash terms
    compete in a single gradient step, so fixing one does not worsen the other
    (unlike sequential SHAKE-style projection).

    Args:
        ca          : [P, 3] CA coordinates (any device, float32 or float64)
        bond_target : ideal CA-CA bond length in Å (default 3.8)
        clash_dist  : minimum non-bonded CA-CA distance in Å (default 3.0)
        k_bond      : bond spring constant — higher value enforces bonds more strictly
        k_clash     : clash penalty weight
        n_steps     : maximum L-BFGS iterations (default 100)

    Returns:
        [P, 3] minimized CA coordinates on the same device as input
    """
    P = ca.shape[0]
    device = ca.device

    x = ca.detach().clone().float()
    x.requires_grad_(True)

    # Non-adjacent upper-triangle pair indices: j >= i+2 (excludes diagonal + bonds)
    idx_i, idx_j = torch.triu_indices(P, P, offset=2, device=device)

    opt = torch.optim.LBFGS([x], lr=1.0, max_iter=n_steps,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        bonds  = (x[1:] - x[:-1]).norm(dim=-1)                        # [P-1]
        e_bond = k_bond * ((bonds - bond_target) ** 2).sum()

        diff   = x[idx_i] - x[idx_j]                                  # [n_pairs, 3]
        dists  = diff.norm(dim=-1)                                     # [n_pairs]
        e_clash = k_clash * torch.clamp(clash_dist - dists, min=0.0).pow(2).sum()

        loss = e_bond + e_clash
        loss.backward()
        return loss

    opt.step(closure)
    return x.detach().to(ca.dtype)


# ---------------------------------------------------------------------------
# 2-bead (CA + CB) physical validity and energy minimization
# ---------------------------------------------------------------------------

_2B_CACB_RANGE  = (1.38, 1.68)   # Å — CA-CB covalent bond
_2B_CACA_RANGE  = (3.5,  4.2)    # Å — consecutive CA pseudo-bond
_2B_CLASH_DIST  = 2.5            # Å — minimum non-bonded heavy-atom distance
_2B_CACB_TARGET = 1.521          # Å — ideal CA-CB
_2B_CACA_TARGET = 3.8            # Å — ideal consecutive CA-CA


def check_2bead_conformation(beads, gly_mask=None):
    """Check whether a 2-bead (CA, CB) conformation is physically meaningful.

    Checks:
        CA-CA consecutive pseudo-bonds (3.5–4.2 Å)
        CA-CB covalent bonds (1.38–1.68 Å; skipped for Gly)
        Non-bonded steric clashes (< 2.5 Å)
        Rg vs Flory scaling

    Args:
        beads    : [P, 2, 3] — order (CA, CB)
        gly_mask : optional bool [P], True for Glycine
    """
    P      = beads.shape[0]
    device = beads.device
    CA, CB = beads[:, 0], beads[:, 1]
    r      = torch.arange(P, device=device)

    bond_violations = []

    def _chk(a, b, name, lo, hi, ra, rb):
        d = (b - a).norm(dim=-1)
        bad = ((d < lo) | (d > hi)).nonzero(as_tuple=False).squeeze(1)
        for k in bad:
            bond_violations.append((int(ra[k]), int(rb[k]), name, round(d[k].item(), 3)))

    _chk(CA[:-1], CA[1:], "CA-CA", *_2B_CACA_RANGE, r[:-1], r[1:])
    if gly_mask is None:
        _chk(CA, CB, "CA-CB", *_2B_CACB_RANGE, r, r)
    else:
        not_gly = ~gly_mask
        _chk(CA[not_gly], CB[not_gly], "CA-CB", *_2B_CACB_RANGE,
             r[not_gly], r[not_gly])
    bond_ok = len(bond_violations) == 0

    # Clashes over flat [2P, 3] excluding bonded pairs
    n_atoms = P * 2
    flat = beads.reshape(n_atoms, 3)

    bi_list, bj_list = [], []
    for res in range(P):
        bi_list.append(res * 2);     bj_list.append(res * 2 + 1)   # CA-CB (intra)
    if gly_mask is not None:
        pairs_filtered = [(i, j) for i, j in zip(bi_list, bj_list)
                          if not (j == i + 1 and gly_mask[i // 2])]
        bi_list = [p[0] for p in pairs_filtered]
        bj_list = [p[1] for p in pairs_filtered]

    bonded_adj = torch.zeros(n_atoms, n_atoms, dtype=torch.bool, device=device)
    if bi_list:
        bi_t = torch.tensor(bi_list, device=device)
        bj_t = torch.tensor(bj_list, device=device)
        bonded_adj[bi_t, bj_t] = True
        bonded_adj[bj_t, bi_t] = True
    # Consecutive CA-CA are pseudo-bonds — don't mark as bonded (allow clash check)

    ii, jj = torch.triu_indices(n_atoms, n_atoms, offset=1, device=device)
    nb_mask  = ~bonded_adj[ii, jj]
    nb_i, nb_j = ii[nb_mask], jj[nb_mask]
    dists_nb = (flat[nb_i] - flat[nb_j]).norm(dim=-1)
    clash_mask = dists_nb < _2B_CLASH_DIST
    clashes = [
        (int(nb_i[k]), int(nb_j[k]), round(dists_nb[k].item(), 3))
        for k in clash_mask.nonzero(as_tuple=False).squeeze(1)
    ]
    clash_free = len(clashes) == 0

    centroid  = CA.mean(0)
    rg        = ((CA - centroid).pow(2).sum(-1).mean()).sqrt().item()
    rg_expect = 2.2 * (P ** 0.38)
    rg_ok     = (0.5 * rg_expect) <= rg <= (2.0 * rg_expect)

    return {
        "valid":             bond_ok and clash_free and rg_ok,
        "bond_ok":           bond_ok,
        "clash_free":        clash_free,
        "rg_ok":             rg_ok,
        "n_bond_violations": len(bond_violations),
        "n_clashes":         len(clashes),
        "bond_violations":   bond_violations,
        "clashes":           clashes,
        "rg_A":              round(rg, 3),
        "rg_expected_A":     round(rg_expect, 3),
    }


def _build_2bead_bond_tensors(P, gly_mask, device):
    """Pre-build bond + non-bonded index tensors for a P-residue 2-bead system."""
    bi, bj, bt = [], [], []
    for r in range(P):
        bi.append(r * 2);     bj.append(r * 2 + 1); bt.append(_2B_CACB_TARGET)   # CA-CB
        if r < P - 1:
            bi.append(r * 2); bj.append((r + 1) * 2); bt.append(_2B_CACA_TARGET) # CA-CA
    if gly_mask is not None:
        keep = [(i, j, t) for i, j, t in zip(bi, bj, bt)
                if not (j == i + 1 and gly_mask[i // 2])]
        bi, bj, bt = (zip(*keep) if keep else ([], [], []))

    bond_i = torch.tensor(bi, dtype=torch.long,    device=device)
    bond_j = torch.tensor(bj, dtype=torch.long,    device=device)
    bond_t = torch.tensor(bt, dtype=torch.float32, device=device)

    n = P * 2
    adj = torch.zeros(n, n, dtype=torch.bool, device=device)
    if len(bi):
        adj[bond_i, bond_j] = True
        adj[bond_j, bond_i] = True
    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    nb_mask = ~adj[ii, jj]
    return bond_i, bond_j, bond_t, ii[nb_mask], jj[nb_mask]


def minimize_energy_2bead(beads, gly_mask=None,
                           k_bond=10.0, k_clash=5.0, n_steps=100,
                           clash_dist=_2B_CLASH_DIST,
                           _cache={}):
    """L-BFGS energy minimization for a 2-bead (CA, CB) conformation.

    Args:
        beads      : [P, 2, 3] in order (CA, CB)
        gly_mask   : bool [P], True for Glycine
        k_bond     : bond spring constant
        k_clash    : clash penalty weight (default 5.0 — stronger than 4-bead default)
        n_steps    : max L-BFGS iterations
        clash_dist : minimum non-bonded distance in Å

    Returns:
        [P, 2, 3] minimized bead coordinates
    """
    P      = beads.shape[0]
    device = beads.device
    gly_key   = tuple(gly_mask.tolist()) if gly_mask is not None else None
    cache_key = (P, gly_key, str(device))
    if cache_key not in _cache:
        _cache[cache_key] = _build_2bead_bond_tensors(P, gly_mask, device)
    bond_i, bond_j, bond_t, nb_i, nb_j = _cache[cache_key]

    n_atoms = P * 2
    x = beads.detach().clone().float().reshape(n_atoms, 3)
    x.requires_grad_(True)

    opt = torch.optim.LBFGS([x], lr=1.0, max_iter=n_steps,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        b_dist  = (x[bond_j] - x[bond_i]).norm(dim=-1)
        e_bond  = k_bond  * ((b_dist - bond_t) ** 2).sum()
        nb_dist = (x[nb_i] - x[nb_j]).norm(dim=-1)
        e_clash = k_clash * torch.clamp(clash_dist - nb_dist, min=0.0).pow(2).sum()
        loss = e_bond + e_clash
        loss.backward()
        return loss

    opt.step(closure)
    return x.detach().reshape(P, 2, 3).to(beads.dtype)


def minimize_energy_4bead(beads, gly_mask=None, bond_target=None,
                           k_bond=10.0, k_clash=1.0, n_steps=100,
                           clash_dist=2.0,
                           _cache={}):
    """L-BFGS energy minimization for a 4-bead (N, CA, C, CB) conformation.

    Minimizes:
        E = k_bond  × Σ_bonds   (|r_j − r_i| − d_ideal)²
          + k_clash × Σ_{non-bonded} max(0, clash_dist − |r_j − r_i|)²

    Bond ideal lengths (Å): N-CA 1.458, CA-C 1.525, CA-CB 1.521, C-N 1.329.
    Non-bonded pairs exclude all 1-2 connected atoms.  Bond and clash terms
    compete in the same gradient step so fixing one does not worsen the other.

    Args:
        beads      : [P, 4, 3] in order (N, CA, C, CB)
        gly_mask   : bool [P], True for Glycine (no real CB; CA-CB bond skipped)
        bond_target: ignored (kept for API consistency); targets are hardcoded
        k_bond     : bond spring constant (default 10.0)
        k_clash    : clash penalty weight (default 1.0)
        n_steps    : max L-BFGS iterations (default 100)
        clash_dist : minimum non-bonded distance in Å (default 2.0)

    Returns:
        [P, 4, 3] minimized bead coordinates on the same device/dtype as input
    """
    P      = beads.shape[0]
    device = beads.device

    # Cache bond/non-bonded pair tensors keyed by (P, gly_mask fingerprint, device)
    gly_key = tuple(gly_mask.tolist()) if gly_mask is not None else None
    cache_key = (P, gly_key, str(device))
    if cache_key not in _cache:
        _cache[cache_key] = _build_4bead_bond_tensors(P, gly_mask, device)
    bond_i, bond_j, bond_t, nb_i, nb_j = _cache[cache_key]

    n_atoms = P * 4
    x = beads.detach().clone().float().reshape(n_atoms, 3)
    x.requires_grad_(True)

    opt = torch.optim.LBFGS([x], lr=1.0, max_iter=n_steps,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        b_vec  = x[bond_j] - x[bond_i]
        b_dist = b_vec.norm(dim=-1)
        e_bond = k_bond * ((b_dist - bond_t) ** 2).sum()

        nb_dist = (x[nb_i] - x[nb_j]).norm(dim=-1)
        e_clash = k_clash * torch.clamp(clash_dist - nb_dist, min=0.0).pow(2).sum()

        loss = e_bond + e_clash
        loss.backward()
        return loss

    opt.step(closure)
    return x.detach().reshape(P, 4, 3).to(beads.dtype)


# ---------------------------------------------------------------------------
# Physical validity check
# ---------------------------------------------------------------------------

def check_conformation(ca,
                       bond_lo=3.5, bond_hi=4.2,
                       clash_dist=3.0,
                       rg_lo_factor=0.5, rg_hi_factor=2.0):
    """Check whether a CA conformation is physically meaningful.

    Three independent criteria:

    1. **Bond lengths** — every consecutive CA-CA distance must lie in
       [bond_lo, bond_hi] Å (default 3.5–4.2 Å; ideal ≈ 3.8 Å).
    2. **Steric clashes** — no non-adjacent CA pair may be closer than
       clash_dist Å (default 3.0 Å; appropriate for CA-sized pseudo-atoms).
    3. **Radius of gyration** — Rg must fall in
       [rg_lo_factor, rg_hi_factor] × Rg_expected, where
       Rg_expected = 2.2 × P^0.38 Å (Flory scaling for globular proteins).

    Args:
        ca:            CA coordinates [P, 3] in Angstrom.
        bond_lo:       Lower CA-CA bond length threshold (Å).
        bond_hi:       Upper CA-CA bond length threshold (Å).
        clash_dist:    CA-CA non-bonded clash distance (Å).
        rg_lo_factor:  Lower Rg multiplier relative to expected.
        rg_hi_factor:  Upper Rg multiplier relative to expected.

    Returns:
        dict with keys:
            valid             bool  — passes ALL three checks
            bond_ok           bool
            clash_free        bool
            rg_ok             bool
            n_bond_violations int   — number of bonds outside [bond_lo, bond_hi]
            n_clashes         int   — number of non-bonded CA pairs < clash_dist
            bond_violations   list  — [(i, j, dist_A), ...]
            clashes           list  — [(i, j, dist_A), ...]
            bond_mean_A       float
            bond_min_A        float
            bond_max_A        float
            rg_A              float — actual radius of gyration
            rg_expected_A     float — Flory-scaling expected Rg
    """
    P = ca.shape[0]

    # --- 1. CA-CA bond lengths ---
    bonds = (ca[1:] - ca[:-1]).norm(dim=-1)            # [P-1]
    bad_mask = (bonds < bond_lo) | (bonds > bond_hi)
    bad_idx = bad_mask.nonzero(as_tuple=False).squeeze(1)
    bond_violations = [
        (int(i), int(i) + 1, round(bonds[i].item(), 3)) for i in bad_idx
    ]
    bond_ok = len(bond_violations) == 0

    # --- 2. Steric clashes (non-adjacent CA pairs) ---
    d = torch.cdist(ca.unsqueeze(0), ca.unsqueeze(0))[0]   # [P, P]
    non_bonded = ~torch.eye(P, dtype=torch.bool, device=ca.device)
    adj = torch.arange(P - 1, device=ca.device)
    non_bonded[adj, adj + 1] = False
    non_bonded[adj + 1, adj] = False
    clash_mat = (d < clash_dist) & non_bonded
    clash_ij = clash_mat.triu(diagonal=1).nonzero(as_tuple=False)
    clashes = [
        (int(r[0]), int(r[1]), round(d[r[0], r[1]].item(), 3))
        for r in clash_ij
    ]
    clash_free = len(clashes) == 0

    # --- 3. Radius of gyration ---
    centroid = ca.mean(0)
    rg = ((ca - centroid).pow(2).sum(-1).mean()).sqrt().item()
    rg_expected = 2.2 * (P ** 0.38)
    rg_ok = (rg_lo_factor * rg_expected) <= rg <= (rg_hi_factor * rg_expected)

    return {
        "valid":             bond_ok and clash_free and rg_ok,
        "bond_ok":           bond_ok,
        "clash_free":        clash_free,
        "rg_ok":             rg_ok,
        "n_bond_violations": len(bond_violations),
        "n_clashes":         len(clashes),
        "bond_violations":   bond_violations,
        "clashes":           clashes,
        "bond_mean_A":       round(bonds.mean().item(), 3),
        "bond_min_A":        round(bonds.min().item(), 3),
        "bond_max_A":        round(bonds.max().item(), 3),
        "rg_A":              round(rg, 3),
        "rg_expected_A":     round(rg_expected, 3),
    }


# ---------------------------------------------------------------------------
# Timing report
# ---------------------------------------------------------------------------

def timing_report(tau, time_per_step_s, out_path,
                  target_ns=1000, ps_per_frame=200,
                  md_ns_per_day=100, md_step_fs=2):
    """Estimate wall-clock time to simulate target_ns using generative MD and
    write a human-readable timing report to out_path.

    Args:
        tau:              Lag per generative step (trajectory frames).
        time_per_step_s:  Measured wall-clock seconds per DDPM sampling call.
        out_path:         File path for the written report.
        target_ns:        Target simulation duration in ns (default 1000).
        ps_per_frame:     Trajectory save interval in ps (default 200).
        md_ns_per_day:    Typical GPU classical MD throughput (ns/day).
        md_step_fs:       Classical MD integration step in fs (default 2).

    Returns:
        dict with the same key/value pairs written to the report file.
    """
    ps_per_step    = tau * ps_per_frame
    ns_per_step    = ps_per_step / 1000.0
    steps_needed   = math.ceil(target_ns / ns_per_step)
    total_s        = steps_needed * time_per_step_s
    total_min      = total_s / 60
    total_h        = total_s / 3600

    md_steps       = int(target_ns * 1e6 / md_step_fs)   # ns → fs → steps
    md_days        = target_ns / md_ns_per_day
    md_hours       = md_days * 24
    speedup        = (md_days * 86400) / max(total_s, 1e-9)

    result = {
        "target_ns":          target_ns,
        "tau_frames":         tau,
        "ps_per_step":        ps_per_step,
        "ns_per_step":        ns_per_step,
        "steps_needed":       steps_needed,
        "time_per_step_s":    round(time_per_step_s, 4),
        "total_s":            round(total_s, 2),
        "total_min":          round(total_min, 2),
        "total_h":            round(total_h, 4),
        "md_step_fs":         md_step_fs,
        "md_steps":           md_steps,
        "md_days":            round(md_days, 1),
        "md_hours":           round(md_hours, 1),
        "speedup_vs_classical_md": round(speedup, 0),
    }

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Generative MD Timing Report",
        "=" * 52,
        f"Generated : {now}",
        f"Target    : {target_ns} ns",
        "",
        "── Generative MD (this model) ─────────────────────",
        f"  τ per step           : {tau} frames = {ps_per_step} ps = {ns_per_step:.2f} ns",
        f"  Steps required       : {steps_needed:,}",
        f"  Time per step        : {time_per_step_s:.4f} s  (measured)",
        f"  Total wall-clock     : {total_s:.1f} s"
                                f"  /  {total_min:.1f} min"
                                f"  /  {total_h:.2f} h",
        "",
        "── Classical MD (reference GPU estimate) ───────────",
        f"  Integration step     : {md_step_fs} fs",
        f"  Steps required       : {md_steps:,}",
        f"  Typical throughput   : ~{md_ns_per_day} ns/day on GPU",
        f"  Total wall-clock     : ~{md_days:.0f} days  /  ~{md_hours:.0f} h",
        "",
        "── Speedup ─────────────────────────────────────────",
        f"  Generative MD vs classical MD : {speedup:.0f}×  faster",
        "=" * 52,
    ]

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    return result
