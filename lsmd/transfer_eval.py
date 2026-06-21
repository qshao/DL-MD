"""Zero-shot rollout and evaluation for the transferable propagator.

Rolls the trained state-conditional propagator out from a reference structure
(rebuilding the dynamic graph each step) and scores the generated CA ensemble
against reference MD with RMSF-profile correlation, Cα-distance JS, and geometry
validity.
"""
import torch

from lsmd import featurize as feat
from lsmd.transfer_model import PropagatorNet, sample_ddpm_union
from lsmd.normalize import UpdateNorm
from lsmd.model import NoiseSchedule
from lsmd import validation as val


def load_checkpoint(ckpt, device):
    """Rebuild (net, schedule, update_norm) from a Task-4 checkpoint dict."""
    hp = ckpt["hparams"]
    net = PropagatorNet(node_dim=hp["node_dim"], edge_dim=hp["edge_dim"],
                        hidden=hp["hidden"], layers=hp["layers"],
                        point_dim=hp["point_dim"]).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    schedule = NoiseSchedule(T=ckpt["T_diff"]).to(device)
    update_norm = UpdateNorm.from_state_dict(ckpt["update_norm"])
    return net, schedule, update_norm


@torch.no_grad()
def rollout(net, schedule, update_norm, R0, t0, res_type, chain_id, res_index,
            *, steps, tau_ps, k, diff_steps=50, device="cpu"):
    """Autoregressive CA trajectory from a reference structure.

    The graph is rebuilt from current (R, t) each step (state-conditional).
    Node features are fixed (computed once from sequence/chain/residue info).
    The sampled normalized update is de-normalized via update_norm.scale, then
    apply_update advances the frames.

    Args:
        net:         PropagatorNet instance (eval mode).
        schedule:    NoiseSchedule instance.
        update_norm: UpdateNorm instance.
        R0:          [N, 3, 3] per-residue rotation matrices at t=0.
        t0:          [N, 3] CA positions at t=0.
        res_type:    [N] long, residue type indices.
        chain_id:    [N] long, chain assignment.
        res_index:   [N] long, sequential residue index.
        steps:       Number of autoregressive steps.
        tau_ps:      Physical lag in picoseconds.
        k:           Number of kNN neighbors for graph building.
        diff_steps:  Number of DDPM reverse steps (default 50).
        device:      Target device (default "cpu").

    Returns:
        [steps+1, N, 3] CA positions (frame 0 = reference t0).
    """
    device = torch.device(device)
    R = R0.to(device)
    t = t0.to(device)
    res_type = res_type.to(device)
    chain_id = chain_id.to(device)
    res_index = res_index.to(device)
    N = t.shape[0]

    # Node features are fixed throughout the trajectory (sequence-based)
    node_feats = feat.frame_node_features(res_type, chain_id, res_index)

    # De-normalization scale for sampled updates
    scale = update_norm.scale.to(device)

    # Single-graph batch vector and tau tensor
    batch = torch.zeros(N, dtype=torch.long, device=device)
    tau = torch.tensor([float(tau_ps)], device=device)

    traj = [t.clone()]
    for _ in range(steps):
        # Rebuild graph from current frames
        edge_index, edge_feats = feat.frame_graph(R, t, k)
        # Sample normalized update via reverse DDPM
        u = sample_ddpm_union(net, node_feats, edge_index, edge_feats,
                              tau, batch, schedule, steps=diff_steps)
        # De-normalize update
        u = u * scale
        # Advance frames
        R, t = feat.apply_update(R, t, u)
        traj.append(t.clone())

    return torch.stack(traj, dim=0)
