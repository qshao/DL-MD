"""End-to-end demo: train a CA-displacement DDPM on a trajectory, sample
future CA conformations, write CA-trace PDBs, and report distributional metrics."""
import os
import time
import torch
from lsmd import data, featurize as f, model as m, decoder as dec, validation as val


def _build_ctx(frames, k):
    """Static reference graph (frame-0 CA) + residue node features.

    Returns (node_feats [P,F], edge_index [2,E], edge_feats [E,4])."""
    X0 = frames["t"][0]                                  # [P,3] CA of frame 0
    edge_index, edge_feats = f.ca_graph(X0, k=k)
    node_feats = f.node_features(
        frames["res_type"], frames["chain_id"],
        frames["res_index"], frames["n_types"],
    )
    return node_feats, edge_index, edge_feats


def train(frames, taus, epochs, k, hidden, layers, lr,
          clip=1.0, batch_size=32, T_diff=200, sigma_aug=0.05,
          density_clip=10.0, device=None):
    """Train a CA-displacement DDPM with multi-lag pairs, inverse-density
    reweighting, and target augmentation.

    Returns (net, schedule, ctx) where ctx = (node_feats, edge_index, edge_feats).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}  taus={taus}  batch_size={batch_size}  T_diff={T_diff}")

    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)
    train_pairs, _ = data.time_split(pairs, val_frac=0.2)
    if train_pairs.shape[0] == 0:
        raise ValueError("No training pairs: taus too large relative to trajectory length.")

    frame_weights = data.compute_frame_weights(frames, density_clip=density_clip)  # [F]
    pair_weights_all = frame_weights[train_pairs[:, 0]]                             # [P]

    n_batches = (train_pairs.shape[0] + batch_size - 1) // batch_size
    print(f"  {train_pairs.shape[0]} training pairs → {n_batches} steps/epoch")

    node_feats, edge_index, edge_feats = _build_ctx(frames, k)
    node_feats = node_feats.to(device)
    edge_index = edge_index.to(device)
    edge_feats = edge_feats.to(device)
    X_all = frames["t"].to(device)                       # [F,P,3] CA coords

    schedule = m.NoiseSchedule(T=T_diff).to(device)
    net = m.FlowNet(
        node_dim=node_feats.shape[1],
        edge_dim=edge_feats.shape[1],
        hidden=hidden,
        layers=layers,
        point_dim=3,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        perm_idx = torch.randperm(train_pairs.shape[0])
        perm = train_pairs[perm_idx]
        perm_w = pair_weights_all[perm_idx]
        epoch_loss, n_steps, t0 = 0.0, 0, time.time()

        for start in range(0, perm.shape[0], batch_size):
            batch = perm[start:start + batch_size]
            batch_w = perm_w[start:start + batch_size].to(device)
            i_idx = batch[:, 0]
            j_idx = batch[:, 1]
            tau_b = batch[:, 2].to(device=device, dtype=X_all.dtype)

            u_batch = f.ca_displacement(X_all[i_idx], X_all[j_idx])   # [B,P,3]

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
    """Load trajectory, train CA-displacement DDPM, sample K future CA traces,
    write PDBs, and compute CA distributional metrics."""
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = data.load_frames(traj_path, top_path)
    net, schedule, (node_feats, edge_index, edge_feats) = train(
        frames, taus, epochs, k, hidden, layers, lr,
        clip=clip, batch_size=batch_size, T_diff=T_diff,
        sigma_aug=sigma_aug, density_clip=density_clip, device=device,
    )
    net.eval()

    X_all = frames["t"]                                   # [F,P,3] CA coords (CPU)

    # Source frame from the val split with matching tau
    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)
    _, val_pairs = data.time_split(pairs, val_frac=0.2)
    matching = val_pairs[val_pairs[:, 2] == infer_tau]
    ref_pair = matching[0] if matching.shape[0] > 0 else val_pairs[0]
    i0 = int(ref_pair[0])
    x_init = X_all[i0]                                     # [P,3]

    # Sample K future displacements and apply to the source CA structure
    delta = m.sample_ddpm(net, node_feats, edge_index, edge_feats, K=K,
                          tau=infer_tau, schedule=schedule,
                          steps=diff_steps, eta=eta, sigma_init=sigma_init)   # [K,P,3]
    ca_model = x_init.to(delta.device).unsqueeze(0) + delta                   # [K,P,3]
    ca_model = ca_model.cpu()

    res_names = ["ALA"] * frames["R"].shape[1]   # CA-trace residue labels (placeholder)
    for kk in range(K):
        dec.write_ca_pdb(ca_model[kk], res_names, os.path.join(out_dir, f"future_{kk}.pdb"))

    # MD reference ensemble: val end-frames matching infer_tau
    ref_end_frames = matching[:, 1][:128] if matching.shape[0] > 0 \
                     else val_pairs[:, 1][:128]
    ca_md = X_all[ref_end_frames.long()]                  # [M,P,3]

    # Displacement-magnitude distributions (fluctuation bulk vs transition tail)
    disp_model = (ca_model - x_init.unsqueeze(0)).norm(dim=-1).pow(2).mean(-1).sqrt()  # [K]
    md_src = matching[:, 0][:128] if matching.shape[0] > 0 else val_pairs[:, 0][:128]
    md_disp = f.ca_displacement(X_all[md_src.long()], X_all[ref_end_frames.long()])    # [M,P,3]
    disp_md = md_disp.norm(dim=-1).pow(2).mean(-1).sqrt()                              # [M]

    pca_result = val.pca_js(ca_model, ca_md)
    rmsf = val.rmsf_profile(ca_model, ca_md)
    disp = val.displacement_js(disp_model, disp_md)
    report = {
        "ca_geometry":       val.ca_geometry(ca_model[0]),
        "pca_js":            pca_result["js"],
        "pca_var_explained": pca_result["var_explained"],
        "ensemble_recall":   val.ensemble_recall(ca_model, ca_md),
        "ensemble_novelty":  val.ensemble_novelty(ca_model, ca_md),
        "distance_matrix_js": val.distance_matrix_js(ca_model, ca_md),
        "rmsf_corr":         rmsf["corr"],
        "displacement_js":   disp["js"],
        "displacement_model_mean": disp["model_mean"],
        "displacement_md_mean":    disp["md_mean"],
        "n_residues":        frames["R"].shape[1],
        "n_md_reference":    ca_md.shape[0],
        "taus":              taus,
        "infer_tau":         infer_tau,
    }
    return report


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="CA-Cartesian long-stride protein MD demo CLI")
    ap.add_argument("--traj",        required=True,  help="Trajectory file path")
    ap.add_argument("--top",         required=True,  help="Topology file path")
    ap.add_argument("--taus",        type=int, nargs="+", default=[1, 2, 5],
                    help="Training lag schedule (frames). 200 ps/frame.")
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
