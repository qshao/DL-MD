"""Load a saved checkpoint and sample future CA conformations.

Usage
-----
python scripts/infer.py \\
    --checkpoint  checkpoints/wt_200ep.pt \\
    --frames      data/wt_frames.pt \\
    --tau         5 \\
    --K           8 \\
    --out         ca_run_200ep

Outputs
-------
  <out>/future_{0..K-1}.pdb   CA-trace PDB files (one per sample)
  <out>/metrics.json          Distributional metrics vs MD reference
"""
import argparse
import json
import os
import torch
from lsmd import data, featurize as f, model as m, decoder as dec, validation as val


def main():
    ap = argparse.ArgumentParser(description="Infer future CA conformations from checkpoint")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint from train.py")
    ap.add_argument("--frames",     required=True, help="Preprocessed frames from preprocess.py")
    ap.add_argument("--tau",        type=int,   default=5,
                    help="Inference lag (frames, 200 ps/frame). Must be in training taus.")
    ap.add_argument("--K",          type=int,   default=8,   help="Samples to generate")
    ap.add_argument("--out",        default="infer_out",     help="Output directory")
    ap.add_argument("--diff_steps", type=int,   default=50,  help="Reverse diffusion steps")
    ap.add_argument("--eta",        type=float, default=1.0, help="DDPM stochasticity")
    ap.add_argument("--sigma_init", type=float, default=1.0, help="Prior scale")
    ap.add_argument("--source_frame", type=int, default=None,
                    help="Index of source frame (default: first val frame at --tau)")
    ap.add_argument("--device",     default=None, help="cuda / cpu (auto if omitted)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device \
             else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    hp   = ckpt["hparams"]
    net  = m.FlowNet(
        node_dim=hp["node_dim"], edge_dim=hp["edge_dim"],
        hidden=hp["hidden"], layers=hp["layers"], point_dim=hp["point_dim"],
    ).to(device)
    net.load_state_dict(ckpt["net_state"])
    net.eval()

    schedule = m.NoiseSchedule(T=hp["T_diff"]).to(device)
    schedule.load_state_dict(ckpt["schedule_state"])
    node_feats = ckpt["node_feats"].to(device)
    edge_index = ckpt["edge_index"].to(device)
    edge_feats = ckpt["edge_feats"].to(device)

    taus = hp["taus"]
    if args.tau not in taus:
        print(f"Warning: --tau {args.tau} was not in training taus {taus}. "
              f"Model may generalise poorly.")

    # Load frames
    frames = torch.load(args.frames, map_location="cpu")
    X_all  = frames["t"]                           # [F, P, 3]
    F, P   = X_all.shape[:2]
    print(f"Trajectory: {F} frames, {P} residues")

    # Pick source frame
    pairs = data.make_multi_lag_pairs(F, taus)
    _, val_pairs = data.time_split(pairs, val_frac=0.2)
    matching = val_pairs[val_pairs[:, 2] == args.tau]
    if args.source_frame is not None:
        i0 = args.source_frame
    elif matching.shape[0] > 0:
        i0 = int(matching[0, 0])
    else:
        i0 = int(val_pairs[0, 0])
    print(f"Source frame: {i0}  (lag τ={args.tau} → target frame ≈{i0 + args.tau})")

    x_init = X_all[i0]                             # [P, 3]

    # Sample
    with torch.no_grad():
        delta = m.sample_ddpm(
            net, node_feats, edge_index, edge_feats,
            K=args.K, tau=args.tau, schedule=schedule,
            steps=args.diff_steps, eta=args.eta, sigma_init=args.sigma_init,
        )                                           # [K, P, 3]
    mode     = hp.get("mode", "ca")
    gly_mask = ckpt.get("gly_mask")
    if gly_mask is not None:
        gly_mask = gly_mask.cpu()

    delta    = delta.cpu()
    x_init_c = x_init.cpu()

    os.makedirs(args.out, exist_ok=True)
    uniq_res  = sorted(set(r.item() for r in frames["res_type"]))
    res_names = ["ALA"] * P   # placeholder

    if mode == "4bead":
        beads_init = x_init_c                          # [P, 4, 3]
        delta_4b   = delta.reshape(args.K, P, 4, 3)   # [K, P, 4, 3]
        model_out  = beads_init.unsqueeze(0) + delta_4b
        for k in range(args.K):
            dec.write_4bead_pdb(model_out[k], res_names,
                                os.path.join(args.out, f"future_{k}.pdb"),
                                gly_mask=gly_mask)
    else:
        ca_model = x_init_c.unsqueeze(0) + delta      # [K, P, 3]
        model_out = ca_model
        for k in range(args.K):
            dec.write_ca_pdb(model_out[k], res_names,
                             os.path.join(args.out, f"future_{k}.pdb"))
    print(f"Wrote {args.K} PDB files to {args.out}/")

    # Compute metrics against MD reference
    ref_end = matching[:, 1][:128] if matching.shape[0] > 0 else val_pairs[:, 1][:128]
    ca_md   = X_all[ref_end.long()]                 # [M, P, 3]

    md_src  = matching[:, 0][:128] if matching.shape[0] > 0 else val_pairs[:, 0][:128]
    if mode == "4bead":
        md_disp = f.four_bead_displacement(X_all[md_src.long()], X_all[ref_end.long()])
    else:
        md_disp = f.ca_displacement(X_all[md_src.long()], X_all[ref_end.long()])
    disp_md    = md_disp.reshape(md_disp.shape[0], -1).norm(dim=-1)
    disp_model = (model_out - x_init_c.unsqueeze(0)).reshape(args.K, -1).norm(dim=-1)

    pca_r  = val.pca_js(ca_model, ca_md)
    rmsf   = val.rmsf_profile(ca_model, ca_md)
    disp   = val.displacement_js(disp_model, disp_md)
    report = {
        "source_frame":      i0,
        "infer_tau":         args.tau,
        "n_samples":         args.K,
        "n_residues":        P,
        "n_md_reference":    int(ca_md.shape[0]),
        "ca_geometry":       val.ca_geometry(ca_model[0]),
        "pca_js":            pca_r["js"],
        "pca_var_explained": pca_r["var_explained"],
        "ensemble_recall":   val.ensemble_recall(ca_model, ca_md),
        "ensemble_novelty":  val.ensemble_novelty(ca_model, ca_md),
        "distance_matrix_js": val.distance_matrix_js(ca_model, ca_md),
        "rmsf_corr":         rmsf["corr"],
        "displacement_js":   disp["js"],
        "displacement_model_mean": disp["model_mean"],
        "displacement_md_mean":    disp["md_mean"],
    }

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
