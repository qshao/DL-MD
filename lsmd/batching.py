"""Disjoint-union batching of per-protein graphs.

Concatenates several proteins into one flat graph so variable-size proteins
train together in a single forward pass. Edges are offset into the union so
no edge ever crosses a protein boundary; a `batch` vector records each node's
protein index for broadcasting per-graph scalars (flow-time, tau).
"""
import torch


def union_collate(graphs):
    """Collate a list of per-protein graph dicts into one union graph.

    Args:
        graphs: list of dicts with keys node_feats [N,F], edge_index [2,E],
                edge_feats [E,De], u_target [N,P], tau (float).
                Optional: temp_K (float) — simulation temperature per graph.

    Returns:
        dict with node_feats [ΣN,F], edge_index [2,ΣE], edge_feats [ΣE,De],
        u_target [ΣN,P], batch [ΣN] long, tau [G] float.
        Optional: temp_K [G] float when present in any graph.
    """
    node_feats, edge_feats, u_target = [], [], []
    edge_index, batch, taus, temps = [], [], [], []
    has_any_temp = any("temp_K" in gr for gr in graphs)
    offset = 0
    for i, gr in enumerate(graphs):
        n = gr["node_feats"].shape[0]
        node_feats.append(gr["node_feats"])
        edge_feats.append(gr["edge_feats"])
        u_target.append(gr["u_target"])
        edge_index.append(gr["edge_index"] + offset)
        batch.append(torch.full((n,), i, dtype=torch.long))
        taus.append(float(gr["tau"]))
        if has_any_temp:
            temps.append(float(gr.get("temp_K", 300.0)))
        offset += n
    out = {
        "node_feats": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_index, dim=1),
        "edge_feats": torch.cat(edge_feats, dim=0),
        "u_target": torch.cat(u_target, dim=0),
        "batch": torch.cat(batch, dim=0),
        "tau": torch.tensor(taus, dtype=torch.float32),
    }
    if temps:
        out["temp_K"] = torch.tensor(temps, dtype=torch.float32)
    return out
