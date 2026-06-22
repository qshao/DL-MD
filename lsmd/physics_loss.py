"""Physics-aware terms for the transferable propagator (Plan 4).

A single chain-aware geometric penalty decodes a predicted per-residue SE(3)
update onto the current frames and scores Ca-Ca chain connectivity, steric
clashes, and an optional Ramachandran prior. It is shared by C1 (soft training
loss) and C2 (sampling guidance).
"""
import torch

from lsmd import featurize as feat
from lsmd.transfer_model import _scatter_mean

# Cache triu indices by (n, device) — index tensors depend only on size and
# device, so reusing them across forward passes avoids repeated O(n²) allocations.
_triu_cache: dict = {}


def _triu_idx(n, device):
    key = (n, str(device))
    if key not in _triu_cache:
        _triu_cache[key] = torch.triu_indices(n, n, offset=2, device=device)
    return _triu_cache[key]


def geometric_penalty(R_cur, t_cur, u_denorm, global_chain, protein_id=None,
                      rama_pot=None, w_bond=1.0, w_clash=1.0, w_rama=0.1,
                      bond_target=3.8, clash_dist=3.0):
    """Geometric energy of the frames obtained by applying u_denorm to (R_cur, t_cur).

    Args:
        R_cur:        [ΣN,3,3] current rotations.
        t_cur:        [ΣN,3] current CA positions.
        u_denorm:     [ΣN,6] de-normalized predicted update.
        global_chain: [ΣN] long, globally-unique chain id (same value = same
                      protein and same chain; see collate_physics).
        protein_id:   [ΣN] long, globally-unique protein id (same value = same
                      protein; used for clash exclusion so inter-chain intra-protein
                      pairs are still penalized). If None, falls back to global_chain
                      (clash restricted to same-chain pairs).
        rama_pot:     optional validation.RamachandranPotential.
        w_bond, w_clash, w_rama: term weights.
        bond_target:  ideal Ca-Ca distance (Angstrom).
        clash_dist:   minimum non-bonded Ca-Ca distance (Angstrom).

    Returns:
        scalar energy, differentiable w.r.t. u_denorm.
    """
    R_next, t_next = feat.apply_update(R_cur, t_cur, u_denorm)
    ca = t_next                                            # [ΣN,3]

    # same-chain consecutive Ca-Ca bonds
    same = global_chain[1:] == global_chain[:-1]           # [ΣN-1]
    bonds = (ca[1:] - ca[:-1]).norm(dim=-1)                # [ΣN-1]
    if same.any():
        e_bond = ((bonds - bond_target) ** 2)[same].mean()
    else:
        e_bond = ca.new_zeros(())

    # non-adjacent Ca-Ca clashes within each protein (local index gap >= 2).
    # unique_consecutive(return_counts=True) gives group sizes in O(ΣN) without
    # per-protein equality scans; contiguous layout (union_collate) lets us
    # slice ca directly instead of gathering, and sum/count avoids cat allocation.
    pid = protein_id if protein_id is not None else global_chain
    _, counts = torch.unique_consecutive(pid, return_counts=True)
    clash_parts = []
    offset = 0
    for n in counts.tolist():
        if n < 3:  # triu offset=2 yields empty pairs; mean() on empty → NaN
            offset += n
            continue
        ca_p = ca[offset:offset + n]   # slice — view, no copy
        li, lj = _triu_idx(n, ca.device)
        d = (ca_p[li] - ca_p[lj]).norm(dim=-1)
        clash_parts.append(torch.clamp(clash_dist - d, min=0.0).pow(2))
        offset += n
    if clash_parts:
        total = sum(t.sum() for t in clash_parts)
        e_clash = total / sum(t.numel() for t in clash_parts)
    else:
        e_clash = ca.new_zeros(())

    e = w_bond * e_bond + w_clash * e_clash

    if rama_pot is not None:
        from lsmd import decoder
        beads = decoder.build_structure(R_next, t_next)    # [ΣN,4,3] (N,CA,C,O)
        e = e + w_rama * rama_pot.energy(beads)
    return e


def collate_physics(examples):
    """Collate current-frame extras for the physics term (mirrors union order).

    Returns:
        R_cur [ΣN,3,3], t_cur [ΣN,3],
        global_chain [ΣN]  — graph_idx*10_000 + chain_id, unique per chain,
        protein_id   [ΣN]  — graph_idx, unique per protein (used for clash),
        res_type     [ΣN]  — canonical residue type indices,
        chain_id     [ΣN]  — local chain id within each protein,
        u_cut        [G]   — per-protein energy ceiling (Phase 3; 0.0 if absent),
        sigma_md_tau [G]   — per-protein MD step variance (Phase 3; 0.0 if absent).
    """
    R_cur, t_cur, chains, pids = [], [], [], []
    res_types, local_chains, u_cuts, sig_taus = [], [], [], []
    for gi, ex in enumerate(examples):
        R_cur.append(ex["R_cur"])
        t_cur.append(ex["t_cur"])
        cid = ex["chain_id"].long()
        if cid.numel() > 0 and int(cid.max()) >= 10_000:
            raise ValueError(
                f"chain_id values must be < 10_000 for global_chain encoding; "
                f"got max={int(cid.max())} in example {gi}"
            )
        chains.append(gi * 10_000 + cid)
        local_chains.append(cid)
        pids.append(torch.full((cid.shape[0],), gi, dtype=torch.long))
        rt = ex.get("res_type")
        if rt is None:
            rt = torch.zeros(cid.shape[0], dtype=torch.long)
        res_types.append(rt.long())
        u_cuts.append(float(ex.get("u_cut", 0.0)))
        sig_taus.append(float(ex.get("sigma_md_tau", 0.0)))
    return {
        "R_cur": torch.cat(R_cur, dim=0),
        "t_cur": torch.cat(t_cur, dim=0),
        "global_chain": torch.cat(chains, dim=0),
        "protein_id": torch.cat(pids, dim=0),
        "res_type": torch.cat(res_types, dim=0),
        "chain_id": torch.cat(local_chains, dim=0),
        "u_cut": torch.tensor(u_cuts),
        "sigma_md_tau": torch.tensor(sig_taus),
    }


def lambda_schedule(step, warmup_steps, lam_max):
    """Linear ramp 0 -> lam_max over warmup_steps, then constant lam_max."""
    if warmup_steps <= 0:
        return float(lam_max)
    return float(lam_max) * min(1.0, step / warmup_steps)


def ddpm_physics_loss(net, union, physics, scale, schedule, *, rama_pot=None,
                      lam=0.0, w_bond=1.0, w_clash=1.0, w_rama=0.1):
    """Union DDPM score loss + lam * geometric_penalty on the clean estimate.

    Mirrors transfer_model.ddpm_loss_union's noising (same RNG order), recovers
    the model's clean-update estimate x0_hat, de-normalizes it, and adds the
    chain-aware geometric penalty. lam=0 reproduces ddpm_loss_union exactly.
    """
    # cast to device only — keep float32 so AMP doesn't quantize scale to fp16
    scale_dev = scale.to(device=union["u_target"].device)
    u_target = union["u_target"] / scale_dev
    node_feats = union["node_feats"]
    edge_index = union["edge_index"]
    edge_feats = union["edge_feats"]
    tau = union["tau"]
    batch = union["batch"].to(u_target.device)

    G = tau.shape[0]
    T = schedule.T
    t_min = max(1, T // 20)
    t_idx = torch.randint(t_min, T + 1, (G,), device=u_target.device)
    t_nodes = t_idx[batch]
    eps = torch.randn_like(u_target)

    sqrt_ab = schedule.sqrt_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    sqrt_1mab = schedule.sqrt_one_minus_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    noisy = sqrt_ab * u_target + sqrt_1mab * eps

    s = (t_idx.float() / T).to(u_target.dtype)
    temp_K = union.get("temp_K")
    pred = net(noisy, s, node_feats, edge_index, edge_feats, tau, batch,
               temp_K=temp_K)

    node_se = ((pred - eps) ** 2).mean(dim=-1)
    score_loss = _scatter_mean(node_se, batch, G).mean()

    if lam == 0.0:
        return score_loss

    u0_hat = (noisy - sqrt_1mab * pred) / sqrt_ab.clamp_min(1e-8)
    u_denorm = u0_hat * scale_dev
    dev = u_denorm.device
    _pid = physics.get("protein_id")
    pen = geometric_penalty(physics["R_cur"].to(dev),
                            physics["t_cur"].to(dev),
                            u_denorm,
                            physics["global_chain"].to(dev),
                            protein_id=_pid.to(dev) if _pid is not None else None,
                            rama_pot=rama_pot, w_bond=w_bond, w_clash=w_clash,
                            w_rama=w_rama)
    return score_loss + lam * pen


def recover_u_denorm(net, union, scale, schedule):
    """Return (u0_hat, u_denorm): the model's clean-update estimate and its
    de-normalized form, for use by the Phase 3 energy/FDT terms.

    Samples a fresh random diffusion timestep (independent of ddpm_physics_loss)
    and uses the score network to recover the clean update estimate x0_hat, then
    de-normalizes by scale. The physics terms act on the clean estimate, not the
    score, so an independent noise draw is acceptable.
    """
    scale_dev = scale.to(device=union["u_target"].device)
    u_target = union["u_target"] / scale_dev
    batch = union["batch"].to(u_target.device)
    G = union["tau"].shape[0]
    T = schedule.T
    t_min = max(1, T // 20)
    t_idx = torch.randint(t_min, T + 1, (G,), device=u_target.device)
    t_nodes = t_idx[batch]
    eps = torch.randn_like(u_target)
    sqrt_ab = schedule.sqrt_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    sqrt_1mab = schedule.sqrt_one_minus_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    noisy = sqrt_ab * u_target + sqrt_1mab * eps
    s = (t_idx.float() / T).to(u_target.dtype)
    pred = net(noisy, s, union["node_feats"], union["edge_index"],
               union["edge_feats"], union["tau"], batch,
               temp_K=union.get("temp_K"))
    u0_hat = (noisy - sqrt_1mab * pred) / sqrt_ab.clamp_min(1e-8)
    return u0_hat, u0_hat * scale_dev


def energy_match_loss(R_cur, t_cur, u_denorm, res_type, protein_id, chain_id,
                      energy, *, u_cut, u_denorm_target=None, w_hi=1.0, w_lo=0.05):
    """Soft, mostly one-sided energy-consistency loss (Phase 3 Stage 2).

    For each protein in the union batch:
      - hinge term  w_hi · relu(U_θ(x_pred)/N - u_cut)   penalizes ONLY
        high-energy / unphysical predicted frames (zero below the ceiling, so
        novel low-energy basins are free), and
      - weak term   w_lo · relu(U_θ(x_pred)/N - U_θ(x_true)/N)  gently
        discourages predictions higher-energy than the real MD transition.

    Args:
        R_cur, t_cur:    [ΣN,3,3], [ΣN,3] current frames.
        u_denorm:        [ΣN,6] de-normalized predicted update.
        res_type:        [ΣN] CANONICAL residue indices.
        protein_id:      [ΣN] per-protein id (groups atoms; energy is per-protein).
        chain_id:        [ΣN] local chain id within each protein.
        energy:          frozen LearnedCGEnergy (no grad on its params).
        u_cut:           per-residue energy ceiling (scalar; from frame_energy_cut).
        u_denorm_target: [ΣN,6] de-normalized TRUE update (enables the weak term).
    Returns:
        scalar tensor (mean over proteins).
    """
    R_pred, t_pred = feat.apply_update(R_cur, t_cur, u_denorm)
    if u_denorm_target is not None:
        _, t_true = feat.apply_update(R_cur, t_cur, u_denorm_target)
    total = t_pred.new_zeros(())
    pids = protein_id.unique()
    for pid in pids:
        m = protein_id == pid
        n = int(m.sum())
        rt = res_type[m]
        cid = chain_id[m]
        u_pred = energy(t_pred[m], rt, cid) / max(n, 1)
        total = total + w_hi * torch.relu(u_pred - u_cut)
        if u_denorm_target is not None:
            u_tru = energy(t_true[m], rt, cid).detach() / max(n, 1)
            total = total + w_lo * torch.relu(u_pred - u_tru)
    return total / max(pids.numel(), 1)


def fdt_loss(u_denorm, protein_id, sigma_md_tau):
    """Fluctuation-dissipation step-variance matching (Phase 3 Stage 2).

    Matches the per-protein translational step variance of the predicted update
    to the MD one-step displacement variance at the same lag τ. Because
    apply_update gives t_f = R·local_trans + t and R is orthogonal, the CA
    displacement variance equals Var(u_denorm[:, :3]).

    Args:
        u_denorm:     [ΣN,6] de-normalized predicted update.
        protein_id:   [ΣN] per-protein id.
        sigma_md_tau: [G] MD target variances, aligned to sorted unique pids.
    Returns:
        scalar tensor (mean squared error over proteins).
    """
    pids = protein_id.unique()                      # sorted ascending
    total = u_denorm.new_zeros(())
    for gi, pid in enumerate(pids):
        m = protein_id == pid
        u_m = u_denorm[m][:, :3]
        var_model = (u_m - u_m.mean(0, keepdim=True)).pow(2).mean()
        var_target = sigma_md_tau[gi].to(var_model.dtype).to(var_model.device)
        total = total + (var_model - var_target) ** 2
    return total / max(pids.numel(), 1)
