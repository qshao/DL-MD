"""Cross-protein trainer for the transferable propagator.

Samples proteins, builds state-conditional training examples at physical lags,
packs them into node-capped disjoint-union minibatches, normalizes targets with
a corpus-level UpdateNorm, and optimizes the union-graph DDPM loss under AMP +
gradient accumulation on a single GPU.
"""
import bisect
import random
import time
import warnings
import torch

from lsmd import data
from lsmd import batching
from lsmd.normalize import UpdateNorm
from lsmd.transfer_model import PropagatorNet, ddpm_loss_union
from lsmd.model import NoiseSchedule
from lsmd import physics_loss as pl


def _max_misses(shards):
    return max(100, 10 * len(shards))


def _build_sampler(shards):
    """Cumulative frame counts for frame-proportional shard sampling."""
    cum, total = [], 0
    for s in shards:
        total += s["t"].shape[0]
        cum.append(total)
    return cum, total


def _sample_shard(shards, cum, total, rng):
    """Pick a shard with probability proportional to its frame count."""
    idx = bisect.bisect_right(cum, rng.randrange(total))
    return shards[min(idx, len(shards) - 1)]


def sample_example(shard, rng, lags_ps, k):
    """Sample one state-conditional training example from a shard, or None."""
    # Cache lag pairs in the shard dict to avoid recomputing on every call
    cache_key = "_pairs_" + "_".join(str(int(lag)) for lag in sorted(lags_ps))
    if cache_key not in shard:
        shard[cache_key] = data.physical_lag_pairs(
            shard["t"].shape[0], shard["dt"], lags_ps,
            traj_breaks=shard.get("traj_breaks"))
    pairs = shard[cache_key]
    if pairs.shape[0] == 0:
        return None
    row = pairs[rng.randrange(pairs.shape[0])]
    start, _end, tau_frames = (int(row[0]), int(row[1]), int(row[2]))
    return data.build_training_example(shard, start, tau_frames, k)


def iter_union_batches(shards, rng, lags_ps, k, max_union_nodes, n_batches,
                       cum_frames=None, total_frames=None):
    """Yield n_batches (union_batch, example_group) pairs.

    union_batch is the union_collate dict; example_group is the raw list of
    examples before collation (needed by collate_physics for the C1 loss).
    If cum_frames/total_frames are provided, sampling is frame-proportional;
    otherwise uniform over shards.
    """
    use_weighted = cum_frames is not None and total_frames is not None
    miss_limit = _max_misses(shards)
    produced = 0
    while produced < n_batches:
        group, n_nodes, misses = [], 0, 0
        while True:
            if use_weighted:
                shard = _sample_shard(shards, cum_frames, total_frames, rng)
            else:
                shard = shards[rng.randrange(len(shards))]
            ex = sample_example(shard, rng, lags_ps, k)
            if ex is None:
                misses += 1
                if misses > miss_limit:
                    raise RuntimeError(
                        f"iter_union_batches: {misses} consecutive misses "
                        "— lags_ps may be too large for the available shards"
                    )
                continue
            n = ex["node_feats"].shape[0]
            if group and n_nodes + n > max_union_nodes:
                # would overflow; defer this example by re-sampling next batch
                break
            group.append(ex)
            n_nodes += n
            misses = 0
            if n_nodes >= max_union_nodes:
                break
        yield batching.union_collate(group), group
        produced += 1


def fit_update_norm(shards, rng, lags_ps, k, n_samples, norm_shards=None,
                    cum_frames=None, total_frames=None):
    """Fit corpus-level UpdateNorm from sampled training updates.

    Args:
        norm_shards:  If provided, sample from this pool for norm fitting
                      (e.g. ATLAS-only to avoid high-T mdCATH outliers).
        cum_frames / total_frames:  Frame-proportional sampling weights for
                      the norm pool.  Ignored when norm_shards is given.
    """
    pool = norm_shards if norm_shards is not None else shards
    if norm_shards is not None:
        pool_cum, pool_total = _build_sampler(pool)
    else:
        pool_cum = cum_frames
        pool_total = total_frames
    use_weighted = pool_cum is not None and pool_total is not None

    cols = []
    miss_limit = _max_misses(pool)
    consecutive_misses = 0
    while len(cols) < n_samples:
        if use_weighted:
            shard = _sample_shard(pool, pool_cum, pool_total, rng)
        else:
            shard = pool[rng.randrange(len(pool))]
        ex = sample_example(shard, rng, lags_ps, k)
        if ex is not None:
            cols.append(ex["u_target"])
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            if consecutive_misses > miss_limit:
                raise RuntimeError(
                    f"fit_update_norm: {consecutive_misses} consecutive misses "
                    "— lags_ps may be too large for the available shards"
                )
    return UpdateNorm.fit(torch.cat(cols, dim=0))


def train(shards, *, lags_ps, k=12, hidden=128, layers=4, lr=1e-3,
          max_union_nodes=2000, accum=4, steps=1000, T_diff=200,
          norm_samples=256, device="cpu", seed=0, lam=0.0, lam_warmup=500,
          log_every=100, grad_clip=1.0, norm_shards=None,
          frame_weighted=True, compile_model=False):
    """Train the union-graph propagator across proteins; return a checkpoint.

    Args:
        lam:            Max physics-penalty weight (C1 soft loss; 0=disabled).
        lam_warmup:     Gradient steps to ramp lam from 0 to lam.
        log_every:      Print loss + speed every this many gradient steps.
        norm_shards:    Shards to fit UpdateNorm on (default: all shards).
                        Pass ATLAS-only shards to avoid high-T mdCATH outliers.
        frame_weighted: If True (default), sample shards proportional to frame
                        count so each MD frame has equal selection probability.
        compile_model:  If True, call torch.compile() on the model for speedup.
    """
    if lam > 0.0 and lam_warmup >= steps:
        peak = pl.lambda_schedule(max(0, steps - 1), lam_warmup, lam)
        warnings.warn(
            f"train: lam_warmup={lam_warmup} >= steps={steps}, so the physics "
            f"penalty weight never reaches lam={lam} (peak is {peak:.4g}). "
            "Reduce lam_warmup or increase steps.",
            stacklevel=2,
        )
    device = torch.device(device)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    cum_frames, total_frames = _build_sampler(shards) if frame_weighted else (None, None)

    update_norm = fit_update_norm(shards, rng, lags_ps, k, norm_samples,
                                  norm_shards=norm_shards,
                                  cum_frames=cum_frames,
                                  total_frames=total_frames)
    scale = update_norm.scale.to(device)

    net = PropagatorNet(hidden=hidden, layers=layers).to(device)
    if compile_model:
        try:
            net = torch.compile(net)
            print("torch.compile: model compiled", flush=True)
        except Exception as exc:
            warnings.warn(f"torch.compile failed ({exc}); using eager mode")
    schedule = NoiseSchedule(T=T_diff).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    batches = iter_union_batches(shards, rng, lags_ps, k,
                                 max_union_nodes, n_batches=steps * accum,
                                 cum_frames=cum_frames, total_frames=total_frames)
    opt.zero_grad()
    lam_t = 0.0
    loss_acc = 0.0       # accumulated (scaled) loss for logging
    nodes_acc = 0        # nodes processed since last log
    t0 = time.perf_counter()
    t_log = t0

    for i, (b, group) in enumerate(batches):
        b_dev = {kk: vv.to(device) for kk, vv in b.items()}
        node_feats = b_dev["node_feats"]
        edge_index  = b_dev["edge_index"]
        edge_feats  = b_dev["edge_feats"]
        tau         = b_dev["tau"]
        batch       = b_dev["batch"]
        nodes_acc  += int(node_feats.shape[0])
        if i % accum == 0:  # recompute once per gradient step
            lam_t = pl.lambda_schedule(i // accum, lam_warmup, lam)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if lam_t == 0.0:
                loss = ddpm_loss_union(net, b_dev["u_target"] / scale, node_feats,
                                       edge_index, edge_feats, tau, batch,
                                       schedule) / accum
            else:
                phys = {kk: vv.to(device)
                        for kk, vv in pl.collate_physics(group).items()}
                loss = pl.ddpm_physics_loss(net, b_dev, phys, scale,
                                            schedule, lam=lam_t) / accum
        loss_val = loss.item()
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            loss_acc += loss_val * accum  # undo the /accum to log full-step loss
        # Non-finite loss: skip backward; gradient stays zero (or from prior valid
        # sub-batches in this accumulation window).  Optimizer step below always runs
        # so GradScaler state remains consistent.

        if (i + 1) % accum == 0:
            scaler.unscale_(opt)
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            # GradScaler only guards against Inf gradients (from AMP overflow), not
            # NaN.  Explicitly zero NaN/Inf gradients so they cannot corrupt weights.
            nan_params = 0
            for p in net.parameters():
                if p.grad is not None and not p.grad.isfinite().all():
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    nan_params += 1
            if nan_params:
                step = (i + 1) // accum
                print(f"  [step {step}] NaN/Inf gradient in {nan_params} tensors — zeroed",
                      flush=True)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            step = (i + 1) // accum
            if step % log_every == 0:
                now = time.perf_counter()
                dt = now - t_log
                steps_per_sec = log_every / dt
                nodes_per_sec = nodes_acc / dt
                avg_loss = loss_acc / (log_every * accum)
                elapsed = now - t0
                eta = (steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                print(
                    f"step {step:6d}/{steps}"
                    f"  loss={avg_loss:.4f}"
                    f"  {steps_per_sec:.2f} step/s"
                    f"  {nodes_per_sec:.0f} nodes/s"
                    f"  elapsed={elapsed/60:.1f}m"
                    f"  ETA={eta/60:.1f}m",
                    flush=True,
                )
                loss_acc = 0.0
                nodes_acc = 0
                t_log = now

    return {
        "model_state": {kk: vv.cpu() for kk, vv in net.state_dict().items()},
        "T_diff": T_diff,
        "update_norm": update_norm.state_dict(),
        "n_aa_types": 21,
        "hparams": {"hidden": hidden, "layers": layers, "k": k,
                    "lags_ps": list(lags_ps), "point_dim": 6,
                    "node_dim": 24, "edge_dim": 13,
                    "lam": lam, "lam_warmup": lam_warmup,
                    "frame_weighted": frame_weighted},
    }
