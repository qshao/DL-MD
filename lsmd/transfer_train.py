"""Cross-protein trainer for the transferable propagator.

Samples proteins, builds state-conditional training examples at physical lags,
packs them into node-capped disjoint-union minibatches, normalizes targets with
a corpus-level UpdateNorm, and optimizes the union-graph DDPM loss under AMP +
gradient accumulation on a single GPU.
"""
import random
import torch

from lsmd import data
from lsmd import batching


def sample_example(shard, rng, lags_ps, k):
    """Sample one state-conditional training example from a shard, or None."""
    num_frames = shard["t"].shape[0]
    pairs = data.physical_lag_pairs(num_frames, shard["dt"], lags_ps)
    if pairs.shape[0] == 0:
        return None
    row = pairs[rng.randrange(pairs.shape[0])]
    start, _end, tau_frames = (int(row[0]), int(row[1]), int(row[2]))
    return data.build_training_example(shard, start, tau_frames, k)


def iter_union_batches(shards, rng, lags_ps, k, max_union_nodes, n_batches):
    """Yield n_batches node-capped union-collated minibatches."""
    produced = 0
    while produced < n_batches:
        group, n_nodes = [], 0
        while True:
            shard = shards[rng.randrange(len(shards))]
            ex = sample_example(shard, rng, lags_ps, k)
            if ex is None:
                continue
            n = ex["node_feats"].shape[0]
            if group and n_nodes + n > max_union_nodes:
                # would overflow; defer this example by re-sampling next batch
                break
            group.append(ex)
            n_nodes += n
            if n_nodes >= max_union_nodes:
                break
        if not group:
            continue
        yield batching.union_collate(group)
        produced += 1


from lsmd.normalize import UpdateNorm
from lsmd.transfer_model import PropagatorNet, ddpm_loss_union
from lsmd.model import NoiseSchedule


def fit_update_norm(shards, rng, lags_ps, k, n_samples):
    """Fit corpus-level UpdateNorm from sampled training updates."""
    cols = []
    while len(cols) < n_samples:
        shard = shards[rng.randrange(len(shards))]
        ex = sample_example(shard, rng, lags_ps, k)
        if ex is not None:
            cols.append(ex["u_target"])
    return UpdateNorm.fit(torch.cat(cols, dim=0))


def train(shards, *, lags_ps, k=12, hidden=128, layers=4, lr=1e-3,
          max_union_nodes=2000, accum=4, steps=1000, T_diff=200,
          norm_samples=256, device="cpu", seed=0):
    """Train the union-graph propagator across proteins; return a checkpoint."""
    device = torch.device(device)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    update_norm = fit_update_norm(shards, rng, lags_ps, k, norm_samples)
    scale = update_norm.scale.to(device)

    net = PropagatorNet(hidden=hidden, layers=layers).to(device)
    schedule = NoiseSchedule(T=T_diff).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    batches = iter_union_batches(shards, rng, lags_ps, k,
                                 max_union_nodes, n_batches=steps * accum)
    opt.zero_grad()
    for i, b in enumerate(batches):
        node_feats = b["node_feats"].to(device)
        edge_index = b["edge_index"].to(device)
        edge_feats = b["edge_feats"].to(device)
        u_target = b["u_target"].to(device) / scale
        tau = b["tau"].to(device)
        batch = b["batch"].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = ddpm_loss_union(net, u_target, node_feats, edge_index,
                                   edge_feats, tau, batch, schedule) / accum
        scaler.scale(loss).backward()
        if (i + 1) % accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

    return {
        "model_state": {kk: vv.cpu() for kk, vv in net.state_dict().items()},
        "T_diff": T_diff,
        "update_norm": update_norm.state_dict(),
        "n_aa_types": 21,
        "hparams": {"hidden": hidden, "layers": layers, "k": k,
                    "lags_ps": list(lags_ps), "point_dim": 6,
                    "node_dim": 24, "edge_dim": 13},
    }
