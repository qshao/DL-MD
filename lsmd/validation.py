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


def ensemble_overlap(ca_gen, ca_md, bins=30):
    """Compute histogram overlap of CA pairwise distance distributions.

    Args:
        ca_gen: Generated CA coordinates [M, 3]
        ca_md: MD CA coordinates [M, 3]
        bins: Number of histogram bins (default 30)

    Returns:
        float: Overlap in [0, 1]
    """
    if ca_gen.shape[0] < 2 or ca_md.shape[0] < 2:
        return 0.0
    dg = torch.pdist(ca_gen)
    dm = torch.pdist(ca_md)
    lo = min(dg.min(), dm.min()).item()
    hi = max(dg.max(), dm.max()).item()
    hg = torch.histc(dg, bins=bins, min=lo, max=hi)
    hm = torch.histc(dm, bins=bins, min=lo, max=hi)
    hg, hm = hg / hg.sum(), hm / hm.sum()
    return torch.minimum(hg, hm).sum().item()


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
