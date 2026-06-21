"""Physics-aware terms for the transferable propagator (Plan 4).

A single chain-aware geometric penalty decodes a predicted per-residue SE(3)
update onto the current frames and scores Ca-Ca chain connectivity, steric
clashes, and an optional Ramachandran prior. It is shared by C1 (soft training
loss) and C2 (sampling guidance).
"""
import torch

from lsmd import featurize as feat


def geometric_penalty(R_cur, t_cur, u_denorm, global_chain, rama_pot=None,
                      w_bond=1.0, w_clash=1.0, w_rama=0.1,
                      bond_target=3.8, clash_dist=3.0):
    """Geometric energy of the frames obtained by applying u_denorm to (R_cur, t_cur).

    Args:
        R_cur:        [ΣN,3,3] current rotations.
        t_cur:        [ΣN,3] current CA positions.
        u_denorm:     [ΣN,6] de-normalized predicted update.
        global_chain: [ΣN] long, globally-unique chain id (same value = same
                      protein and same chain; see collate_physics).
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
        e_bond = ((bonds - bond_target) ** 2)[same].sum()
    else:
        e_bond = ca.new_zeros(())

    # non-adjacent Ca-Ca clashes (all pairs with index gap >= 2)
    P = ca.shape[0]
    ii, jj = torch.triu_indices(P, P, offset=2, device=ca.device)
    d = (ca[ii] - ca[jj]).norm(dim=-1)
    e_clash = torch.clamp(clash_dist - d, min=0.0).pow(2).sum()

    e = w_bond * e_bond + w_clash * e_clash

    if rama_pot is not None:
        from lsmd import decoder
        beads = decoder.build_structure(R_next, t_next)    # [ΣN,4,3] (N,CA,C,O)
        e = e + w_rama * rama_pot.energy(beads)
    return e


def collate_physics(examples):
    """Collate current-frame extras for the physics term (mirrors union order).

    Returns R_cur [ΣN,3,3], t_cur [ΣN,3], global_chain [ΣN] where global_chain
    = graph_idx*1000 + chain_id so chains never merge across proteins.
    """
    R_cur, t_cur, chains = [], [], []
    for gi, ex in enumerate(examples):
        R_cur.append(ex["R_cur"])
        t_cur.append(ex["t_cur"])
        chains.append(gi * 1000 + ex["chain_id"].long())
    return {
        "R_cur": torch.cat(R_cur, dim=0),
        "t_cur": torch.cat(t_cur, dim=0),
        "global_chain": torch.cat(chains, dim=0),
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
    from lsmd.transfer_model import _scatter_mean

    u_target = union["u_target"]
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
    pred = net(noisy, s, node_feats, edge_index, edge_feats, tau, batch)

    node_se = ((pred - eps) ** 2).mean(dim=-1)
    score_loss = _scatter_mean(node_se, batch, G).mean()

    if lam == 0.0:
        return score_loss

    u0_hat = (noisy - sqrt_1mab * pred) / sqrt_ab.clamp_min(1e-8)
    u_denorm = u0_hat * scale.to(u0_hat)
    pen = geometric_penalty(physics["R_cur"].to(u_denorm.device),
                            physics["t_cur"].to(u_denorm.device),
                            u_denorm, physics["global_chain"].to(u_denorm.device),
                            rama_pot=rama_pot, w_bond=w_bond, w_clash=w_clash,
                            w_rama=w_rama)
    return score_loss + lam * pen
