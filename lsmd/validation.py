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
