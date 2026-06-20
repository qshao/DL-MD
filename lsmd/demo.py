"""End-to-end demo: train FlowNet on a trajectory, sample futures, write PDBs, report metrics."""
import os
import torch
from lsmd import data, featurize as f, model as m, decoder as dec, validation as val


def _build_ctx(frames, k):
    """Build graph context from frame 0.

    Args:
        frames: dict from data.load_frames
        k: number of nearest neighbours

    Returns:
        (node_feats, edge_index, edge_feats) built from frame 0
    """
    R0, t0 = frames["R"][0], frames["t"][0]
    edge_index = f.knn_graph(t0, k=k)
    edge_feats = f.edge_features(R0, t0, edge_index)
    node_feats = f.node_features(
        frames["res_type"], frames["chain_id"],
        frames["res_index"], frames["n_types"],
    )
    return node_feats, edge_index, edge_feats


def train(frames, tau, epochs, k, hidden, layers, sigma, lr):
    """Train FlowNet on a loaded trajectory.

    Args:
        frames: dict from data.load_frames
        tau: time lag between paired frames
        epochs: number of training epochs
        k: number of nearest neighbours for graph construction
        hidden: hidden dimension of FlowNet
        layers: number of message-passing layers
        sigma: prior scale for conditional flow matching
        lr: Adam learning rate

    Returns:
        (net, ctx) where ctx = (node_feats, edge_index, edge_feats) from frame 0
    """
    pairs = data.make_pairs(frames["R"].shape[0], tau)
    train_pairs, _ = data.time_split(pairs, val_frac=0.2)
    if train_pairs.shape[0] == 0:
        raise ValueError("No training pairs: tau too large relative to trajectory length.")

    node_feats, edge_index, edge_feats = _build_ctx(frames, k)

    net = m.FlowNet(
        node_dim=node_feats.shape[1],
        edge_dim=edge_feats.shape[1],
        hidden=hidden,
        layers=layers,
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for _ in range(epochs):
        perm = train_pairs[torch.randperm(train_pairs.shape[0])]
        for i, j in perm.tolist():
            R_t, t_t = frames["R"][i], frames["t"][i]
            R_f, t_f = frames["R"][j], frames["t"][j]
            u_target = f.relative_update(R_t, t_t, R_f, t_f)
            opt.zero_grad()
            loss = m.cfm_loss(net, u_target, node_feats, edge_index, edge_feats, sigma=sigma)
            loss.backward()
            opt.step()

    return net, (node_feats, edge_index, edge_feats)


def run_demo(traj_path, top_path, tau, out_dir, K=8, epochs=50,
             k=8, hidden=64, layers=3, sigma=0.1, lr=1e-3):
    """Full demo: load trajectory, train, sample K futures, write PDBs, return metrics.

    Args:
        traj_path: path to trajectory file
        top_path: path to topology file
        tau: time lag between paired frames
        out_dir: output directory for PDB files
        K: number of future structures to sample
        epochs: number of training epochs
        k: number of nearest neighbours
        hidden: FlowNet hidden dimension
        layers: number of message-passing layers
        sigma: prior scale for CFM
        lr: Adam learning rate

    Returns:
        dict with keys: model_geometry, diversity, ensemble_overlap_vs_true,
                        n_residues, tau
    """
    os.makedirs(out_dir, exist_ok=True)

    frames = data.load_frames(traj_path, top_path)
    net, (node_feats, edge_index, edge_feats) = train(
        frames, tau, epochs, k, hidden, layers, sigma, lr,
    )

    # Identify the first validation pair to use as the query frame
    pairs = data.make_pairs(frames["R"].shape[0], tau)
    _, val_pairs = data.time_split(pairs, val_frac=0.2)
    i0 = int(val_pairs[0, 0])
    R_t, t_t = frames["R"][i0], frames["t"][i0]

    # Sample K future structures
    u = m.sample(net, node_feats, edge_index, edge_feats, K=K, sigma=sigma)
    R_f, t_f = dec.decode_frames(R_t, t_t, u)

    # Build, idealize, and write each sample
    res_names = ["ALA"] * frames["R"].shape[1]  # cosmetic; backbone-only demo
    atoms_K = []
    for kk in range(K):
        atoms = dec.idealize(dec.build_structure(R_f[kk], t_f[kk]))
        atoms_K.append(atoms)
        dec.write_pdb(atoms, res_names, os.path.join(out_dir, f"future_{kk}.pdb"))
    atoms_K = torch.stack(atoms_K, 0)  # [K, N, 4, 3]

    # True future CA positions for ensemble overlap
    md_ca = frames["t"][int(val_pairs[0, 1])]  # [N, 3]

    report = {
        "model_geometry": val.geometry_metrics(atoms_K[0]),
        "diversity": val.diversity(atoms_K),
        "ensemble_overlap_vs_true": val.ensemble_overlap(atoms_K[0][:, 1, :], md_ca),
        "n_residues": frames["R"].shape[1],
        "tau": tau,
    }
    return report


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Long-stride protein MD demo CLI")
    ap.add_argument("--traj", required=True, help="Trajectory file path")
    ap.add_argument("--top", required=True, help="Topology file path")
    ap.add_argument("--tau", type=int, required=True, help="Time lag (frames)")
    ap.add_argument("--out", default="demo_out", help="Output directory")
    ap.add_argument("--K", type=int, default=8, help="Number of sampled futures")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    ap.add_argument("--k", type=int, default=8, help="KNN neighbours")
    ap.add_argument("--hidden", type=int, default=64, help="FlowNet hidden dim")
    ap.add_argument("--layers", type=int, default=3, help="FlowNet layers")
    ap.add_argument("--sigma", type=float, default=0.1, help="Prior scale")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = ap.parse_args()

    rep = run_demo(
        args.traj, args.top, args.tau, args.out,
        K=args.K, epochs=args.epochs, k=args.k,
        hidden=args.hidden, layers=args.layers,
        sigma=args.sigma, lr=args.lr,
    )
    print(json.dumps(rep, indent=2))
