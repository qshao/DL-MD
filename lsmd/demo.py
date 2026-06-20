"""End-to-end demo: train FlowNet on a trajectory, sample futures, write PDBs, report metrics."""
import os
import time
import torch
from lsmd import data, featurize as f, model as m, decoder as dec, validation as val


def _build_ctx(frames, k):
    R0, t0 = frames["R"][0], frames["t"][0]
    edge_index = f.knn_graph(t0, k=k)
    edge_feats = f.edge_features(R0, t0, edge_index)
    node_feats = f.node_features(
        frames["res_type"], frames["chain_id"],
        frames["res_index"], frames["n_types"],
    )
    return node_feats, edge_index, edge_feats


def train(frames, taus, epochs, k, hidden, layers, lr,
          clip=1.0, batch_size=32, T_diff=200, sigma_aug=0.05,
          density_clip=10.0, device=None):
    """Train FlowNet with DDPM score matching on a loaded trajectory.

    Uses multi-lag mini-batch training with inverse-density reweighting
    (upweights rare conformations) and target augmentation (smooths discrete
    MD frames into a continuous distribution).

    Args:
        frames:       dict from data.load_frames
        taus:         list of integer lag values (frames)
        epochs:       number of training epochs
        k:            KNN neighbours for graph construction
        hidden:       FlowNet hidden dimension
        layers:       number of message-passing layers
        lr:           Adam learning rate
        clip:         max gradient norm (0 to disable)
        batch_size:   pairs per optimizer step
        T_diff:       DDPM noise schedule length
        sigma_aug:    target augmentation noise scale (0 to disable)
        density_clip: max density weight relative to mean
        device:       torch device; auto-selects CUDA if None

    Returns:
        (net, schedule, ctx) where ctx = (node_feats, edge_index, edge_feats)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}  taus={taus}  batch_size={batch_size}  T_diff={T_diff}")

    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)
    train_pairs, _ = data.time_split(pairs, val_frac=0.2)
    if train_pairs.shape[0] == 0:
        raise ValueError("No training pairs: taus too large relative to trajectory length.")

    # Inverse-density reweighting: upweight rare conformations
    frame_weights = data.compute_frame_weights(frames, density_clip=density_clip)  # [F]
    pair_weights_all = frame_weights[train_pairs[:, 0]]                             # [P]

    n_batches = (train_pairs.shape[0] + batch_size - 1) // batch_size
    print(f"  {train_pairs.shape[0]} training pairs → {n_batches} steps/epoch")

    node_feats, edge_index, edge_feats = _build_ctx(frames, k)
    node_feats  = node_feats.to(device)
    edge_index  = edge_index.to(device)
    edge_feats  = edge_feats.to(device)
    R_all = frames["R"].to(device)
    t_all = frames["t"].to(device)

    schedule = m.NoiseSchedule(T=T_diff).to(device)
    net = m.FlowNet(
        node_dim=node_feats.shape[1],
        edge_dim=edge_feats.shape[1],
        hidden=hidden,
        layers=layers,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        perm_idx = torch.randperm(train_pairs.shape[0])
        perm = train_pairs[perm_idx]
        perm_w = pair_weights_all[perm_idx]
        epoch_loss, n_steps, t0 = 0.0, 0, time.time()

        for start in range(0, perm.shape[0], batch_size):
            batch   = perm[start:start + batch_size]              # [B, 3] CPU
            batch_w = perm_w[start:start + batch_size].to(device) # [B]
            i_idx   = batch[:, 0]
            j_idx   = batch[:, 1]
            tau_b   = batch[:, 2].to(device=device, dtype=R_all.dtype)

            u_batch = f.relative_update(
                R_all[i_idx], t_all[i_idx], R_all[j_idx], t_all[j_idx]
            )   # [B, N, 6]

            opt.zero_grad()
            loss = m.ddpm_loss(
                net, u_batch, node_feats, edge_index, edge_feats,
                tau_b, schedule, pair_weights=batch_w, sigma_aug=sigma_aug,
            )
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1

        print(f"Epoch {epoch+1}/{epochs}  loss={epoch_loss/n_steps:.4f}  "
              f"t={time.time()-t0:.1f}s")

    return net, schedule, (node_feats, edge_index, edge_feats)


def run_demo(traj_path, top_path, taus, infer_tau, out_dir, K=8, epochs=50,
             k=8, hidden=64, layers=3, lr=1e-3, clip=1.0, batch_size=32,
             T_diff=200, diff_steps=50, eta=1.0, sigma_init=1.0,
             sigma_aug=0.05, density_clip=10.0, device=None):
    """Full demo: load trajectory, train DDPM on multi-lag pairs, sample K futures,
    write PDBs, compute distributional metrics.

    Args:
        traj_path:    path to trajectory file
        top_path:     path to topology file
        taus:         list of training lag values (frames)
        infer_tau:    lag time for inference (any value)
        out_dir:      output directory for PDB files
        K:            number of future structures to sample
        epochs:       number of training epochs
        k:            KNN neighbours
        hidden:       FlowNet hidden dimension
        layers:       number of message-passing layers
        lr:           Adam learning rate
        clip:         gradient clip norm
        batch_size:   pairs per optimizer step
        T_diff:       DDPM noise schedule length
        diff_steps:   reverse-process denoising steps
        eta:          DDPM stochasticity (1=full DDPM, 0=DDIM)
        sigma_init:   prior scale for reverse-process start
        sigma_aug:    target augmentation noise scale
        density_clip: max density weight relative to mean
        device:       torch device (auto-detected if None)

    Returns:
        dict with keys: model_geometry, diversity_rmsd, ramachandran_js,
                        pca_js, pca_var_explained, ensemble_recall,
                        ensemble_novelty, n_residues, n_md_reference,
                        taus, infer_tau
    """
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = data.load_frames(traj_path, top_path)
    net, schedule, (node_feats, edge_index, edge_feats) = train(
        frames, taus, epochs, k, hidden, layers, lr,
        clip=clip, batch_size=batch_size, T_diff=T_diff,
        sigma_aug=sigma_aug, density_clip=density_clip, device=device,
    )

    # Pick a source frame from val set with matching tau
    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)
    _, val_pairs = data.time_split(pairs, val_frac=0.2)
    matching = val_pairs[val_pairs[:, 2] == infer_tau]
    ref_pair = matching[0] if matching.shape[0] > 0 else val_pairs[0]
    i0 = int(ref_pair[0])

    R_t = frames["R"][i0].to(device)
    t_t = frames["t"][i0].to(device)

    # Sample K futures
    u = m.sample_ddpm(net, node_feats, edge_index, edge_feats, K=K,
                      tau=infer_tau, schedule=schedule,
                      steps=diff_steps, eta=eta, sigma_init=sigma_init)
    R_f, t_f = dec.decode_frames(R_t, t_t, u)

    # Build and write each sample
    res_names = ["ALA"] * frames["R"].shape[1]
    atoms_K = []
    for kk in range(K):
        atoms = dec.idealize(dec.build_structure(R_f[kk].cpu(), t_f[kk].cpu()))
        atoms_K.append(atoms)
        dec.write_pdb(atoms, res_names, os.path.join(out_dir, f"future_{kk}.pdb"))
    atoms_K = torch.stack(atoms_K, 0)   # [K, N, 4, 3]

    # Assemble MD reference ensemble from val end-frames matching infer_tau
    ref_end_frames = matching[:, 1][:128] if matching.shape[0] > 0 \
                     else val_pairs[:, 1][:128]
    md_atoms = torch.stack([
        dec.build_structure(frames["R"][int(j)].cpu(), frames["t"][int(j)].cpu())
        for j in ref_end_frames
    ])   # [M, N, 4, 3]

    pca_result = val.pca_js(atoms_K, md_atoms)
    report = {
        "model_geometry":    val.geometry_metrics(atoms_K[0]),
        "diversity_rmsd":    val.diversity(atoms_K),
        "ramachandran_js":   val.ramachandran_js(atoms_K, md_atoms),
        "pca_js":            pca_result["js"],
        "pca_var_explained": pca_result["var_explained"],
        "ensemble_recall":   val.ensemble_recall(atoms_K, md_atoms),
        "ensemble_novelty":  val.ensemble_novelty(atoms_K, md_atoms),
        "n_residues":        frames["R"].shape[1],
        "n_md_reference":    md_atoms.shape[0],
        "taus":              taus,
        "infer_tau":         infer_tau,
    }
    return report


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Long-stride protein MD demo CLI")
    ap.add_argument("--traj",        required=True,  help="Trajectory file path")
    ap.add_argument("--top",         required=True,  help="Topology file path")
    ap.add_argument("--taus",        type=int, nargs="+", default=[10, 25, 50, 100, 200],
                    help="Training lag schedule (frames).")
    ap.add_argument("--infer_tau",   type=int, default=None,
                    help="Lag time for inference. Defaults to max(taus).")
    ap.add_argument("--out",         default="demo_out", help="Output directory")
    ap.add_argument("--K",           type=int,   default=8)
    ap.add_argument("--epochs",      type=int,   default=50)
    ap.add_argument("--k",           type=int,   default=8,    help="KNN neighbours")
    ap.add_argument("--hidden",      type=int,   default=64)
    ap.add_argument("--layers",      type=int,   default=3)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--clip",        type=float, default=1.0,  help="Gradient clip norm")
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--T_diff",      type=int,   default=200,  help="DDPM noise levels")
    ap.add_argument("--diff_steps",  type=int,   default=50,   help="Reverse-process steps")
    ap.add_argument("--eta",         type=float, default=1.0,  help="DDPM stochasticity")
    ap.add_argument("--sigma_init",  type=float, default=1.0,  help="Prior scale")
    ap.add_argument("--sigma_aug",   type=float, default=0.05, help="Target augmentation noise")
    ap.add_argument("--density_clip",type=float, default=10.0, help="Max density weight")
    ap.add_argument("--device",      default=None,  help="cuda / cpu (auto if omitted)")
    args = ap.parse_args()

    infer_tau = args.infer_tau if args.infer_tau is not None else max(args.taus)
    dev = torch.device(args.device) if args.device else None
    rep = run_demo(
        args.traj, args.top, args.taus, infer_tau, args.out,
        K=args.K, epochs=args.epochs, k=args.k,
        hidden=args.hidden, layers=args.layers,
        lr=args.lr, clip=args.clip, batch_size=args.batch_size,
        T_diff=args.T_diff, diff_steps=args.diff_steps,
        eta=args.eta, sigma_init=args.sigma_init,
        sigma_aug=args.sigma_aug, density_clip=args.density_clip,
        device=dev,
    )
    print(json.dumps(rep, indent=2))
