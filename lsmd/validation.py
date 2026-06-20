import math
import torch
from lsmd import decoder as dec


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
    ca_model = atoms_model[:, :, 1, :].float()   # [K, N, 3]
    ca_md    = atoms_md[:,    :, 1, :].float()   # [M, N, 3]
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
    ca_model = atoms_model[:, :, 1, :]   # [K, N, 3]
    ca_md    = atoms_md[:,    :, 1, :]   # [M, N, 3]
    M = ca_md.shape[0]
    covered = 0
    for m_idx in range(M):
        diff = ca_model - ca_md[m_idx].unsqueeze(0)    # [K, N, 3]
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
    ca_model = atoms_model[:, :, 1, :]   # [K, N, 3]
    ca_md    = atoms_md[:,    :, 1, :]   # [M, N, 3]
    K = ca_model.shape[0]
    novel = 0
    for k_idx in range(K):
        diff = ca_md - ca_model[k_idx].unsqueeze(0)    # [M, N, 3]
        rmsd = diff.norm(dim=-1).pow(2).mean(dim=-1).sqrt()   # [M]
        if rmsd.min().item() >= r_ang:
            novel += 1
    return novel / K
