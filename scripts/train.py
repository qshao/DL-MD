"""Train a CA-displacement DDPM on preprocessed trajectory frames and save a
checkpoint for later inference.

Usage
-----
python scripts/train.py \\
    --frames  data/wt_frames.pt \\
    --taus    1 2 5 \\
    --epochs  200 \\
    --out     checkpoints/wt_200ep.pt

The checkpoint stores everything needed for inference:
    net_state   — FlowNet weights
    schedule    — NoiseSchedule object
    node_feats  — [P, F] reference node features
    edge_index  — [2, E] reference kNN graph
    edge_feats  — [E, 4] reference edge features
    hparams     — dict of training hyperparameters
"""
import argparse
import os
import time
import torch
from lsmd import data, featurize as f, model as m


def build_ctx(frames, k, device):
    X0 = frames["t"][0]
    edge_index, edge_feats = f.ca_graph(X0, k=k)
    node_feats = f.node_features(
        frames["res_type"], frames["chain_id"],
        frames["res_index"], frames["n_types"],
    )
    return (node_feats.to(device), edge_index.to(device), edge_feats.to(device))


def train(frames, taus, epochs, k, hidden, layers, lr,
          clip, batch_size, T_diff, sigma_aug, density_clip, device):

    pairs = data.make_multi_lag_pairs(frames["t"].shape[0], taus)
    train_pairs, _ = data.time_split(pairs, val_frac=0.2)
    if train_pairs.shape[0] == 0:
        raise ValueError("No training pairs — taus too large for this trajectory.")

    frame_weights = data.compute_frame_weights(frames, density_clip=density_clip)
    pair_weights  = frame_weights[train_pairs[:, 0]]

    node_feats, edge_index, edge_feats = build_ctx(frames, k, device)
    X_all = frames["t"].to(device)

    schedule = m.NoiseSchedule(T=T_diff).to(device)
    net = m.FlowNet(
        node_dim=node_feats.shape[1],
        edge_dim=edge_feats.shape[1],
        hidden=hidden,
        layers=layers,
        point_dim=3,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    n_batches = (train_pairs.shape[0] + batch_size - 1) // batch_size
    print(f"Training on {device}  taus={taus}  "
          f"{train_pairs.shape[0]} pairs  {n_batches} steps/epoch")

    for epoch in range(epochs):
        perm_idx = torch.randperm(train_pairs.shape[0])
        perm   = train_pairs[perm_idx]
        perm_w = pair_weights[perm_idx]
        total_loss, n_steps, t0 = 0.0, 0, time.time()

        for start in range(0, perm.shape[0], batch_size):
            batch   = perm[start:start + batch_size]
            batch_w = perm_w[start:start + batch_size].to(device)
            tau_b   = batch[:, 2].to(device=device, dtype=X_all.dtype)

            u_batch = f.ca_displacement(X_all[batch[:, 0]], X_all[batch[:, 1]])

            opt.zero_grad()
            loss = m.ddpm_loss(
                net, u_batch, node_feats, edge_index, edge_feats,
                tau_b, schedule, pair_weights=batch_w, sigma_aug=sigma_aug,
            )
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            total_loss += loss.item()
            n_steps += 1

        print(f"Epoch {epoch+1:4d}/{epochs}  "
              f"loss={total_loss/n_steps:.4f}  t={time.time()-t0:.1f}s")

    return net, schedule, (node_feats, edge_index, edge_feats)


def main():
    ap = argparse.ArgumentParser(description="Train CA-displacement DDPM")
    ap.add_argument("--frames",       required=True,  help="Preprocessed .pt file from preprocess.py")
    ap.add_argument("--taus",         type=int, nargs="+", default=[1, 2, 5],
                    help="Training lag schedule (frames, 200 ps/frame)")
    ap.add_argument("--epochs",       type=int,   default=200)
    ap.add_argument("--out",          default="checkpoint.pt", help="Output checkpoint path")
    ap.add_argument("--k",            type=int,   default=8,    help="kNN neighbours")
    ap.add_argument("--hidden",       type=int,   default=64)
    ap.add_argument("--layers",       type=int,   default=3)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--clip",         type=float, default=1.0)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--T_diff",       type=int,   default=200,  help="DDPM noise levels")
    ap.add_argument("--sigma_aug",    type=float, default=0.05, help="Target augmentation noise")
    ap.add_argument("--density_clip", type=float, default=10.0)
    ap.add_argument("--device",       default=None, help="cuda / cpu (auto if omitted)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device \
             else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = torch.load(args.frames, map_location="cpu")
    print(f"Loaded {args.frames}  —  {frames['t'].shape[0]} frames, "
          f"{frames['t'].shape[1]} residues")

    net, schedule, (node_feats, edge_index, edge_feats) = train(
        frames, args.taus, args.epochs, args.k,
        args.hidden, args.layers, args.lr,
        args.clip, args.batch_size, args.T_diff,
        args.sigma_aug, args.density_clip, device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    checkpoint = {
        "net_state":      net.state_dict(),
        "schedule_state": schedule.state_dict(),
        "node_feats":     node_feats.cpu(),
        "edge_index":     edge_index.cpu(),
        "edge_feats":     edge_feats.cpu(),
        "hparams": {
            "taus":        args.taus,
            "node_dim":    node_feats.shape[1],
            "edge_dim":    edge_feats.shape[1],
            "hidden":      args.hidden,
            "layers":      args.layers,
            "point_dim":   3,
            "T_diff":      args.T_diff,
        },
    }
    torch.save(checkpoint, args.out)
    print(f"Checkpoint saved → {args.out}")


if __name__ == "__main__":
    main()
