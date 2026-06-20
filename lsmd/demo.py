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


def train(frames, taus, epochs, k, hidden, layers, sigma, lr, clip=1.0, device=None):
    """Train FlowNet on a loaded trajectory using multi-lag pairs.

    Trains across multiple lag times simultaneously so the model learns what
    the protein looks like at different time horizons.  At inference you can
    request any tau (including ones not in the training schedule).

    Args:
        frames: dict from data.load_frames
        taus: list of integer lag values (frames) — e.g. [10, 25, 50, 100, 200]
        epochs: number of training epochs
        k: number of nearest neighbours for graph construction
        hidden: hidden dimension of FlowNet
        layers: number of message-passing layers
        sigma: prior scale for conditional flow matching
        lr: Adam learning rate
        clip: max gradient norm (0 to disable)
        device: torch device; auto-selects CUDA if available when None

    Returns:
        (net, ctx) where ctx = (node_feats, edge_index, edge_feats) from frame 0
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}  taus={taus}")

    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)  # [P, 3]: (i, j, tau)
    train_pairs, _ = data.time_split(pairs, val_frac=0.2)
    if train_pairs.shape[0] == 0:
        raise ValueError("No training pairs: taus too large relative to trajectory length.")
    print(f"  {train_pairs.shape[0]} training pairs across {len(taus)} lag values")

    node_feats, edge_index, edge_feats = _build_ctx(frames, k)
    node_feats = node_feats.to(device)
    edge_index  = edge_index.to(device)
    edge_feats  = edge_feats.to(device)
    # pre-move all frames to device once (avoids per-step transfer overhead)
    R_all = frames["R"].to(device)
    t_all = frames["t"].to(device)

    net = m.FlowNet(
        node_dim=node_feats.shape[1],
        edge_dim=edge_feats.shape[1],
        hidden=hidden,
        layers=layers,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    import time
    for epoch in range(epochs):
        perm = train_pairs[torch.randperm(train_pairs.shape[0])]
        epoch_loss, t0 = 0.0, time.time()
        for i, j, tau in perm.tolist():
            u_target = f.relative_update(R_all[i], t_all[i], R_all[j], t_all[j])
            opt.zero_grad()
            loss = m.cfm_loss(net, u_target, node_feats, edge_index, edge_feats, tau, sigma=sigma)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}  loss={epoch_loss/len(perm):.4f}  t={time.time()-t0:.1f}s")

    return net, (node_feats, edge_index, edge_feats)


def run_demo(traj_path, top_path, taus, infer_tau, out_dir, K=8, epochs=50,
             k=8, hidden=64, layers=3, sigma=0.1, lr=1e-3, clip=1.0, device=None):
    """Full demo: load trajectory, train on multi-lag pairs, sample K futures, write PDBs.

    Args:
        traj_path:  path to trajectory file
        top_path:   path to topology file
        taus:       list of training lag values (frames) — e.g. [10, 25, 50, 100, 200]
        infer_tau:  lag time to request at inference (can be any value, not just in taus)
        out_dir:    output directory for PDB files
        K:          number of future structures to sample
        epochs:     number of training epochs
        k:          number of nearest neighbours
        hidden:     FlowNet hidden dimension
        layers:     number of message-passing layers
        sigma:      prior scale for CFM
        lr:         Adam learning rate
        clip:       gradient clip norm
        device:     torch device (auto-detected if None)

    Returns:
        dict with keys: model_geometry, diversity, ensemble_overlap_vs_true,
                        n_residues, taus, infer_tau
    """
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = data.load_frames(traj_path, top_path)
    net, (node_feats, edge_index, edge_feats) = train(
        frames, taus, epochs, k, hidden, layers, sigma, lr, clip=clip, device=device,
    )

    # Use the first validation pair (from the largest tau) as the query frame
    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)
    _, val_pairs = data.time_split(pairs, val_frac=0.2)
    i0 = int(val_pairs[0, 0])
    R_t = frames["R"][i0].to(device)
    t_t = frames["t"][i0].to(device)

    # Sample K futures conditioned on the requested inference tau
    u = m.sample(net, node_feats, edge_index, edge_feats, K=K, tau=infer_tau, sigma=sigma)
    R_f, t_f = dec.decode_frames(R_t, t_t, u)

    # Build, idealize, and write each sample (move back to CPU for geometry ops)
    res_names = ["ALA"] * frames["R"].shape[1]  # cosmetic; backbone-only demo
    atoms_K = []
    for kk in range(K):
        atoms = dec.idealize(dec.build_structure(R_f[kk].cpu(), t_f[kk].cpu()))
        atoms_K.append(atoms)
        dec.write_pdb(atoms, res_names, os.path.join(out_dir, f"future_{kk}.pdb"))
    atoms_K = torch.stack(atoms_K, 0)  # [K, N, 4, 3]

    # True future CA positions for ensemble overlap (CPU, using closest tau from val set)
    md_ca = frames["t"][int(val_pairs[0, 1])]  # [N, 3]

    report = {
        "model_geometry": val.geometry_metrics(atoms_K[0]),
        "diversity": val.diversity(atoms_K),
        "ensemble_overlap_vs_true": val.ensemble_overlap(atoms_K[0][:, 1, :], md_ca),
        "n_residues": frames["R"].shape[1],
        "taus": taus,
        "infer_tau": infer_tau,
    }
    return report


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Long-stride protein MD demo CLI")
    ap.add_argument("--traj", required=True, help="Trajectory file path")
    ap.add_argument("--top", required=True, help="Topology file path")
    ap.add_argument("--taus", type=int, nargs="+", default=[10, 25, 50, 100, 200],
                    help="Training lag schedule (frames). Default: 10 25 50 100 200")
    ap.add_argument("--infer_tau", type=int, default=None,
                    help="Lag time for inference (frames). Defaults to max(taus).")
    ap.add_argument("--out", default="demo_out", help="Output directory")
    ap.add_argument("--K", type=int, default=8, help="Number of sampled futures")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    ap.add_argument("--k", type=int, default=8, help="KNN neighbours")
    ap.add_argument("--hidden", type=int, default=64, help="FlowNet hidden dim")
    ap.add_argument("--layers", type=int, default=3, help="FlowNet layers")
    ap.add_argument("--sigma", type=float, default=0.1, help="Prior scale")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--clip", type=float, default=1.0, help="Gradient clip norm (0=off)")
    ap.add_argument("--device", default=None, help="Device: cuda / cpu (auto if omitted)")
    args = ap.parse_args()

    infer_tau = args.infer_tau if args.infer_tau is not None else max(args.taus)
    dev = torch.device(args.device) if args.device else None
    rep = run_demo(
        args.traj, args.top, args.taus, infer_tau, args.out,
        K=args.K, epochs=args.epochs, k=args.k,
        hidden=args.hidden, layers=args.layers,
        sigma=args.sigma, lr=args.lr, clip=args.clip, device=dev,
    )
    print(json.dumps(rep, indent=2))
