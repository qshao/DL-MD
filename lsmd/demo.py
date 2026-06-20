"""End-to-end demo: train FlowNet on a trajectory, sample futures, write PDBs, report metrics."""
import os
import time
import torch
from lsmd import data, featurize as f, model as m, decoder as dec, validation as val


def _build_ctx(frames, k):
    """Build graph context from frame 0.

    Returns (node_feats, edge_index, edge_feats) — all on CPU.
    """
    R0, t0 = frames["R"][0], frames["t"][0]
    edge_index = f.knn_graph(t0, k=k)
    edge_feats = f.edge_features(R0, t0, edge_index)
    node_feats = f.node_features(
        frames["res_type"], frames["chain_id"],
        frames["res_index"], frames["n_types"],
    )
    return node_feats, edge_index, edge_feats


def train(frames, taus, epochs, k, hidden, layers, sigma, lr,
          clip=1.0, batch_size=32, device=None):
    """Train FlowNet on a loaded trajectory using multi-lag mini-batch training.

    Trains across multiple lag times simultaneously so the model learns what
    the protein looks like at different time horizons.  At inference you can
    request any tau (including ones not in the training schedule).

    Mini-batch training groups `batch_size` pairs per optimizer step, which
    dramatically reduces Python overhead and improves GPU utilisation.

    Args:
        frames:     dict from data.load_frames
        taus:       list of integer lag values (frames) — e.g. [10, 25, 50, 100, 200]
        epochs:     number of training epochs
        k:          number of nearest neighbours for graph construction
        hidden:     hidden dimension of FlowNet
        layers:     number of message-passing layers
        sigma:      prior scale for conditional flow matching
        lr:         Adam learning rate
        clip:       max gradient norm (0 to disable)
        batch_size: pairs per optimizer step
        device:     torch device; auto-selects CUDA if available when None

    Returns:
        (net, ctx) where ctx = (node_feats, edge_index, edge_feats) on device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}  taus={taus}  batch_size={batch_size}")

    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)  # [P, 3]
    train_pairs, _ = data.time_split(pairs, val_frac=0.2)
    if train_pairs.shape[0] == 0:
        raise ValueError("No training pairs: taus too large relative to trajectory length.")
    n_batches = (train_pairs.shape[0] + batch_size - 1) // batch_size
    print(f"  {train_pairs.shape[0]} training pairs → {n_batches} steps/epoch")

    node_feats, edge_index, edge_feats = _build_ctx(frames, k)
    node_feats = node_feats.to(device)
    edge_index  = edge_index.to(device)
    edge_feats  = edge_feats.to(device)
    # Pre-move all frames to device once — avoids per-step host→device transfer
    R_all = frames["R"].to(device)
    t_all = frames["t"].to(device)

    net = m.FlowNet(
        node_dim=node_feats.shape[1],
        edge_dim=edge_feats.shape[1],
        hidden=hidden,
        layers=layers,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = train_pairs[torch.randperm(train_pairs.shape[0])]  # shuffle on CPU
        epoch_loss, n_steps, t0 = 0.0, 0, time.time()

        for start in range(0, perm.shape[0], batch_size):
            batch = perm[start:start + batch_size]          # [B, 3] CPU LongTensor
            i_idx = batch[:, 0]                             # [B] start frames
            j_idx = batch[:, 1]                             # [B] end frames
            tau_b = batch[:, 2].to(device=device, dtype=R_all.dtype)  # [B] float on device

            # Vectorised: one GPU call builds [B, N, 6] in one shot
            u_batch = f.relative_update(
                R_all[i_idx], t_all[i_idx], R_all[j_idx], t_all[j_idx]
            )  # [B, N, 6]

            opt.zero_grad()
            loss = m.cfm_loss(net, u_batch, node_feats, edge_index, edge_feats,
                              tau_b, sigma=sigma)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1

        print(f"Epoch {epoch+1}/{epochs}  loss={epoch_loss/n_steps:.4f}  t={time.time()-t0:.1f}s")

    return net, (node_feats, edge_index, edge_feats)


def run_demo(traj_path, top_path, taus, infer_tau, out_dir, K=8, epochs=50,
             k=8, hidden=64, layers=3, sigma=0.1, lr=1e-3, clip=1.0,
             batch_size=32, device=None):
    """Full demo: load trajectory, train on multi-lag pairs, sample K futures, write PDBs.

    Args:
        traj_path:  path to trajectory file
        top_path:   path to topology file
        taus:       list of training lag values (frames) — e.g. [10, 25, 50, 100, 200]
        infer_tau:  lag time to request at inference (any value, not restricted to taus)
        out_dir:    output directory for PDB files
        K:          number of future structures to sample
        epochs:     number of training epochs
        k:          number of nearest neighbours
        hidden:     FlowNet hidden dimension
        layers:     number of message-passing layers
        sigma:      prior scale for CFM
        lr:         Adam learning rate
        clip:       gradient clip norm
        batch_size: pairs per optimizer step
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
        frames, taus, epochs, k, hidden, layers, sigma, lr,
        clip=clip, batch_size=batch_size, device=device,
    )

    # Pick a validation pair whose tau matches infer_tau for a fair comparison
    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)
    _, val_pairs = data.time_split(pairs, val_frac=0.2)
    # Filter to pairs with matching tau; fall back to any val pair if not found
    matching = val_pairs[val_pairs[:, 2] == infer_tau]
    ref_pair = matching[0] if matching.shape[0] > 0 else val_pairs[0]
    i0, j0 = int(ref_pair[0]), int(ref_pair[1])

    R_t = frames["R"][i0].to(device)
    t_t = frames["t"][i0].to(device)

    # Sample K futures conditioned on infer_tau — all K run in one batched pass
    u = m.sample(net, node_feats, edge_index, edge_feats, K=K, tau=infer_tau, sigma=sigma)
    R_f, t_f = dec.decode_frames(R_t, t_t, u)

    # Build, idealize, and write each sample (geometry ops on CPU)
    res_names = ["ALA"] * frames["R"].shape[1]
    atoms_K = []
    for kk in range(K):
        atoms = dec.idealize(dec.build_structure(R_f[kk].cpu(), t_f[kk].cpu()))
        atoms_K.append(atoms)
        dec.write_pdb(atoms, res_names, os.path.join(out_dir, f"future_{kk}.pdb"))
    atoms_K = torch.stack(atoms_K, 0)  # [K, N, 4, 3]

    # True future: frame j0 is exactly infer_tau steps ahead of i0
    md_ca = frames["t"][j0]  # [N, 3] on CPU

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
    ap.add_argument("--batch_size", type=int, default=32, help="Pairs per optimizer step")
    ap.add_argument("--device", default=None, help="Device: cuda / cpu (auto if omitted)")
    args = ap.parse_args()

    infer_tau = args.infer_tau if args.infer_tau is not None else max(args.taus)
    dev = torch.device(args.device) if args.device else None
    rep = run_demo(
        args.traj, args.top, args.taus, infer_tau, args.out,
        K=args.K, epochs=args.epochs, k=args.k,
        hidden=args.hidden, layers=args.layers,
        sigma=args.sigma, lr=args.lr, clip=args.clip,
        batch_size=args.batch_size, device=dev,
    )
    print(json.dumps(rep, indent=2))
