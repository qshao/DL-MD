import torch
import torch.nn.functional as F_nn
from lsmd import geometry as g


def relative_update(R_t, t_t, R_f, t_f):
    """Compute relative update from target to frame in target's local frame.

    Args:
        R_t: Target rotation [..., 3, 3]
        t_t: Target translation [..., 3]
        R_f: Frame rotation [..., 3, 3]
        t_f: Frame translation [..., 3]

    Returns:
        u: Relative update [..., 6] = [local_trans(3), axis_angle(3)]

    The update is E(3)-invariant: it's computed in the source frame's local coordinate system.
    """
    Rt_inv = R_t.transpose(-1, -2)
    local_trans = (Rt_inv @ (t_f - t_t).unsqueeze(-1)).squeeze(-1)
    rel_R = Rt_inv @ R_f
    axis_angle = g.so3_log(rel_R)
    return torch.cat([local_trans, axis_angle], dim=-1)


def apply_update(R_t, t_t, u):
    """Apply relative update to get frame.

    Args:
        R_t: Target rotation [..., 3, 3]
        t_t: Target translation [..., 3]
        u: Relative update [..., 6] = [local_trans(3), axis_angle(3)]

    Returns:
        (R_f, t_f): Frame rotation [..., 3, 3] and translation [..., 3]

    Inverse of relative_update.
    """
    local_trans, axis_angle = u[..., :3], u[..., 3:]
    t_f = (R_t @ local_trans.unsqueeze(-1)).squeeze(-1) + t_t
    R_f = R_t @ g.so3_exp(axis_angle)
    return R_f, t_f


def knn_graph(t, k):
    """Build k-nearest neighbor graph.

    Args:
        t: Node positions [N, 3]
        k: Number of neighbors per node

    Returns:
        edge_index: Edge indices [2, E], row 0 = src, row 1 = dst
    """
    n = t.shape[0]
    d = torch.cdist(t, t)
    d.fill_diagonal_(float("inf"))
    k = min(k, n - 1)
    idx = d.topk(k, largest=False).indices  # [n,k]
    src = torch.arange(n, device=t.device).unsqueeze(1).expand(n, k).reshape(-1)
    dst = idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


def edge_features(R, t, edge_index):
    """Compute invariant edge features.

    Args:
        R: Node rotations [N, 3, 3]
        t: Node positions [N, 3]
        edge_index: Edge indices [2, E]

    Returns:
        feats: Edge features [E, 13] = [rel_pos(3), dist(1), rel_R(9)]

    All features are E(3)-invariant.
    """
    src, dst = edge_index
    Rs_inv = R[src].transpose(-1, -2)
    rel_pos = (Rs_inv @ (t[dst] - t[src]).unsqueeze(-1)).squeeze(-1)  # [E,3] invariant
    dist = (t[dst] - t[src]).norm(dim=-1, keepdim=True)               # [E,1]
    rel_R = (Rs_inv @ R[dst]).reshape(-1, 9)                          # [E,9] invariant
    return torch.cat([rel_pos, dist, rel_R], dim=-1)                  # [E,13]


def node_features(res_type, chain_id, res_index, n_types):
    """Compute node features.

    Args:
        res_type: Residue type [N] (long)
        chain_id: Chain ID [N] (long)
        res_index: Residue index [N] (long)
        n_types: Number of residue types

    Returns:
        feats: Node features [N, F] where F = n_types + 1 + 2
    """
    rt = F_nn.one_hot(res_type, num_classes=n_types).float()
    ch = chain_id.float().unsqueeze(-1)
    # smooth positional encoding of residue index
    pos = res_index.float().unsqueeze(-1)
    pe = torch.cat([torch.sin(pos / 100.0), torch.cos(pos / 100.0)], dim=-1)
    return torch.cat([rt, ch, pe], dim=-1)


def ca_displacement(X_i, X_j):
    """Per-pair Kabsch-aligned CA displacement Δ = align(X_j→X_i) − X_i.

    Removes whole-protein tumbling so Δ reflects internal conformational change.

    Args:
        X_i: source CA coords [P,3] or [B,P,3]
        X_j: target CA coords [P,3] or [B,P,3]

    Returns:
        Δ: same shape as inputs.
    """
    R, t = g.kabsch(X_i, X_j)                       # align X_j onto X_i
    X_j_aligned = X_j @ R.transpose(-1, -2) + t.unsqueeze(-2)
    return X_j_aligned - X_i


def ca_graph(X, k):
    """kNN graph + invariant edge features from CA positions.

    Args:
        X: CA coords [P,3] (reference structure, frame-0 orientation).
        k: neighbours per node.

    Returns:
        edge_index [2,E], edge_feats [E,4] = [rel_pos(3), dist(1)].
        rel_pos is in the (canonicalized) frame-0 orientation.
    """
    edge_index = knn_graph(X, k)
    src, dst = edge_index
    rel_pos = X[dst] - X[src]                        # [E,3]
    dist = rel_pos.norm(dim=-1, keepdim=True)        # [E,1]
    edge_feats = torch.cat([rel_pos, dist], dim=-1)  # [E,4]
    return edge_index, edge_feats
