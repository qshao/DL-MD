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


_ALL_MDCATH_TEMPS = [320, 348, 379, 413, 450]


def _allowed_temps_set(max_temp_K):
    """Integer set of mdCATH temperatures (K) at or below max_temp_K."""
    return frozenset(t for t in _ALL_MDCATH_TEMPS if t <= max_temp_K)


def _current_max_temp(step, temp_schedule):
    """Return the max temperature (K) at gradient step `step`.

    temp_schedule: list of (start_step, max_temp_K) sorted by start_step.
    """
    current = temp_schedule[0][1]
    for thresh, temp in temp_schedule:
        if step >= thresh:
            current = temp
    return current


def _shard_temp_at_frame(shard, frame_idx):
    """Return the simulation temperature (K) for a given frame index.

    mdCATH shards carry traj_breaks and traj_temps; ATLAS shards have neither
    and are assumed to be at physiological temperature (300 K).
    """
    traj_breaks = shard.get("traj_breaks")
    traj_temps = shard.get("traj_temps")
    if traj_breaks is None or traj_temps is None:
        return 300.0
    seg_idx = bisect.bisect_right(traj_breaks.tolist(), frame_idx)
    seg_idx = min(seg_idx, len(traj_temps) - 1)
    return float(traj_temps[seg_idx])


def sample_example(shard, rng, lags_ps, k, allowed_temps=None, reverse_prob=0.0):
    """Sample one state-conditional training example from a shard, or None.

    Args:
        allowed_temps:  frozenset of integer temperatures (K) to include.
                        Only applies to mdCATH shards that carry ``traj_temps``.
                        None means all segments are eligible (ATLAS or no curriculum).
        reverse_prob:   probability of swapping source/dest frames (time-reversal
                        augmentation).  0.0 = disabled.
    """
    # Cache lag pairs per (lags_ps, allowed_temps) combination
    cache_key = "_pairs_" + "_".join(str(int(lag)) for lag in sorted(lags_ps))
    if allowed_temps is not None:
        cache_key += "_T" + "_".join(str(t) for t in sorted(allowed_temps))
    if cache_key not in shard:
        shard[cache_key] = data.physical_lag_pairs(
            shard["t"].shape[0], shard["dt"], lags_ps,
            traj_breaks=shard.get("traj_breaks"),
            traj_temps=shard.get("traj_temps"),
            allowed_temps=allowed_temps)
    pairs = shard[cache_key]
    if pairs.shape[0] == 0:
        return None
    row = pairs[rng.randrange(pairs.shape[0])]
    start, _end, tau_frames = (int(row[0]), int(row[1]), int(row[2]))
    temp_K = _shard_temp_at_frame(shard, start)
    reverse = reverse_prob > 0.0 and rng.random() < reverse_prob
    ex = data.build_training_example(shard, start, tau_frames, k,
                                     temp_K=temp_K, reverse=reverse)
    if ex is None:
        return None
    # Attach per-shard Phase 3 targets (default 0.0 so non-Phase-3 calls are unaffected)
    ex["u_cut"] = float(shard.get("_u_cut", 0.0))
    sigma_map = shard.get("_sigma_md_tau_map", {})
    if sigma_map:
        actual_tau_ps = tau_frames * float(shard["dt"])
        best_lag = min(sigma_map, key=lambda k: abs(k - actual_tau_ps))
        ex["sigma_md_tau"] = sigma_map[best_lag]
    else:
        ex["sigma_md_tau"] = 0.0
    # Ensure res_type is present (union node features depend on it; build_training_example
    # should already include it, but copy from shard as a fallback)
    if "res_type" not in ex:
        ex["res_type"] = shard["res_type"]
    return ex


def iter_union_batches(shards, rng, lags_ps, k, max_union_nodes, n_batches,
                       cum_frames=None, total_frames=None,
                       get_allowed_temps=None, reverse_prob=0.0):
    """Yield n_batches (union_batch, example_group) pairs.

    union_batch is the union_collate dict; example_group is the raw list of
    examples before collation (needed by collate_physics for the C1 loss).

    Args:
        cum_frames / total_frames: pre-computed sampler weights for
            frame-proportional shard selection.
        get_allowed_temps: optional zero-arg callable that returns the current
            frozenset of allowed mdCATH temperatures (K). Called once per
            union-batch so the curriculum can change mid-training.
        reverse_prob: probability of time-reversal per example (0.0 = disabled).
    """
    use_weighted = cum_frames is not None and total_frames is not None
    miss_limit = _max_misses(shards)
    produced = 0
    while produced < n_batches:
        allowed_temps = get_allowed_temps() if get_allowed_temps is not None else None
        group, n_nodes, misses = [], 0, 0
        while True:
            if use_weighted:
                shard = _sample_shard(shards, cum_frames, total_frames, rng)
            else:
                shard = shards[rng.randrange(len(shards))]
            ex = sample_example(shard, rng, lags_ps, k, allowed_temps=allowed_temps,
                                reverse_prob=reverse_prob)
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
          frame_weighted=True, compile_model=False, temp_schedule=None,
          temp_emb_dim=8, reverse_prob=0.0, resume_from=None,
          checkpoint_every=0, checkpoint_path=None,
          energy_ckpt=None, lam_energy=0.0, lam_fdt=0.0, phys_warmup=500,
          w_hi=1.0, w_lo=0.05):
    """Train the union-graph propagator across proteins; return a checkpoint.

    Args:
        lam:            Max physics-penalty weight (C1 soft loss; 0=disabled).
        lam_warmup:     Gradient steps to ramp lam from 0 to lam.
        log_every:      Print loss + speed every this many gradient steps.
        norm_shards:    Shards to fit UpdateNorm on (default: all shards).
                        Pass ATLAS-only shards to avoid high-T mdCATH outliers.
        frame_weighted: Sample shards proportional to frame count (default True).
        compile_model:  Call torch.compile() on the model (default False).
        temp_schedule:  Temperature curriculum: list of (start_step, max_temp_K)
                        pairs sorted by start_step.  At each gradient step the
                        sampler restricts mdCATH trajectories to temperatures
                        ≤ current max.  None (default) uses all temperatures.
                        Example: [(0,320),(2000,348),(5000,379),(10000,413),(15000,450)]
        temp_emb_dim:   Size of the temperature embedding added to PropagatorNet.
                        0 disables it (matches the original architecture).
                        Default 8: conditions the model on simulation temperature
                        so it learns correctly-scaled fluctuations at each T.
        reverse_prob:   Probability of time-reversal per training example.
                        0.5 doubles effective training data via microscopic
                        reversibility; 0.0 (default) disables.
        resume_from:    Checkpoint dict (from torch.load). If provided, model
                        weights and optimizer state are loaded and training
                        continues from checkpoint['step']. The 'steps' arg then
                        means additional steps beyond the checkpoint, not total.
        checkpoint_every: Save an intermediate checkpoint every this many
                        gradient steps (0 = disabled). Requires checkpoint_path.
        checkpoint_path: Path template for periodic saves; '{step}' is replaced
                        with the absolute step number, e.g.
                        'checkpoints/v2_{step}.pt'.
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

    net = PropagatorNet(hidden=hidden, layers=layers,
                        temp_emb_dim=temp_emb_dim).to(device)
    if compile_model:
        try:
            net = torch.compile(net)
            print("torch.compile: model compiled", flush=True)
        except Exception as exc:
            warnings.warn(f"torch.compile failed ({exc}); using eager mode")
    schedule = NoiseSchedule(T=T_diff).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # --- Phase 3: Load frozen energy model and precompute per-shard targets ---
    energy = None
    if energy_ckpt is not None:
        from lsmd.learned_energy import (LearnedCGEnergy, frame_energy_cut,
                                         md_step_cov)
        energy = LearnedCGEnergy.load(energy_ckpt, map_location=device)
        for p in energy.parameters():
            p.requires_grad_(False)
        edev = next(energy.parameters()).device
        for s in shards:
            s["_u_cut"] = frame_energy_cut(
                energy,
                s["t"].float().to(edev),
                s["res_type"].long().to(edev),
                s["chain_id"].long().to(edev),
                pct=95.0)
            s["_sigma_md_tau_map"] = {
                float(lag): md_step_cov(s["t"].float(), float(s["dt"]), float(lag))
                for lag in lags_ps
            }
        print(f"  Energy model loaded from {energy_ckpt}; "
              f"precomputed targets for {len(shards)} shards", flush=True)

    # Resume from checkpoint: load weights and optimizer state, offset step counter
    init_step = 0
    if resume_from is not None:
        net.load_state_dict(resume_from["model_state"], strict=False)
        if "optimizer_state" in resume_from:
            opt.load_state_dict(resume_from["optimizer_state"])
        init_step = resume_from.get("step", resume_from.get("hparams", {}).get("steps", 0))
        print(f"  Resumed from checkpoint at step {init_step}; "
              f"training {steps} more steps (target: {init_step + steps})", flush=True)

    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- Temperature curriculum ---
    _allowed_state = [None]  # mutable cell updated in the training loop
    if temp_schedule:
        initial_max = _current_max_temp(init_step, temp_schedule)
        _allowed_state[0] = _allowed_temps_set(initial_max)
        if init_step == 0:
            print(f"  Temperature curriculum starts at {initial_max}K "
                  f"(schedule: {temp_schedule})", flush=True)
        else:
            print(f"  Temperature curriculum resumed at step {init_step}: "
                  f"max_temp={initial_max}K", flush=True)

    def _get_allowed_temps():
        return _allowed_state[0]

    batches = iter_union_batches(shards, rng, lags_ps, k,
                                 max_union_nodes, n_batches=steps * accum,
                                 cum_frames=cum_frames, total_frames=total_frames,
                                 get_allowed_temps=_get_allowed_temps,
                                 reverse_prob=reverse_prob)
    opt.zero_grad()
    lam_t = 0.0
    lam_e = 0.0
    lam_f = 0.0
    loss_acc = 0.0       # accumulated (scaled) loss for logging
    nodes_acc = 0        # nodes processed since last log
    had_backward = False  # tracks whether any sub-batch in current accum window ran backward
    t0 = time.perf_counter()
    t_log = t0
    _prev_max_temp = _allowed_state[0] and max(_allowed_state[0])

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
            lam_e = pl.lambda_schedule(i // accum, phys_warmup, lam_energy)
            lam_f = pl.lambda_schedule(i // accum, phys_warmup, lam_fdt)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            temp_K = b_dev.get("temp_K")
            if lam_t == 0.0 and (energy is None or (lam_e == 0.0 and lam_f == 0.0)):
                loss = ddpm_loss_union(net, b_dev["u_target"] / scale, node_feats,
                                       edge_index, edge_feats, tau, batch,
                                       schedule, temp_K=temp_K) / accum
            else:
                phys = {kk: vv.to(device)
                        for kk, vv in pl.collate_physics(group).items()}
                loss = pl.ddpm_physics_loss(net, b_dev, phys, scale,
                                            schedule, lam=lam_t) / accum
        # Energy/FDT losses computed in float32 (outside autocast) to avoid
        # WCA overflow: at r≈0 the WCA gradient is ~1e22, which overflows
        # float16 (max 65504) → NaN in every tensor.
        if energy is not None and (lam_e > 0.0 or lam_f > 0.0):
            _, u_denorm = pl.recover_u_denorm(net, b_dev, scale, schedule)
            u_denorm = u_denorm.float()
            if lam_e > 0.0:
                loss = loss + (lam_e / accum) * pl.energy_match_loss(
                    phys["R_cur"].float(), phys["t_cur"].float(), u_denorm,
                    phys["res_type"], phys["protein_id"], phys["chain_id"],
                    energy, u_cut=float(phys["u_cut"].mean()),
                    u_denorm_target=b_dev["u_target"].float(), w_hi=w_hi, w_lo=w_lo)
            if lam_f > 0.0:
                loss = loss + (lam_f / accum) * pl.fdt_loss(
                    u_denorm, phys["protein_id"], phys["sigma_md_tau"])
        loss_val = loss.item()
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            loss_acc += loss_val * accum  # undo the /accum to log full-step loss
            had_backward = True

        if (i + 1) % accum == 0:
            if had_backward:
                # scaler.unscale_ requires at least one scaled backward in the window;
                # skip entirely when all sub-batches had non-finite loss to avoid the
                # "No inf checks were recorded" assertion in PyTorch >= 2.4.
                scaler.unscale_(opt)
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                # GradScaler guards against Inf (AMP overflow) but not NaN; zero both.
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
            # When had_backward is False, skip ALL scaler calls — PyTorch >= 2.4
            # requires inf checks before both step() and update().
            opt.zero_grad()
            had_backward = False

            local_step = (i + 1) // accum
            step = init_step + local_step

            # Advance curriculum temperature if schedule dictates
            if temp_schedule:
                new_max = _current_max_temp(step, temp_schedule)
                if new_max != _prev_max_temp:
                    _allowed_state[0] = _allowed_temps_set(new_max)
                    _prev_max_temp = new_max
                    print(f"  [step {step}] curriculum: max_temp={new_max}K "
                          f"(allowed: {sorted(_allowed_state[0])}K)", flush=True)

            if local_step % log_every == 0:
                now = time.perf_counter()
                dt = now - t_log
                steps_per_sec = log_every / dt
                nodes_per_sec = nodes_acc / dt
                avg_loss = loss_acc / (log_every * accum)
                elapsed = now - t0
                eta = (steps - local_step) / steps_per_sec if steps_per_sec > 0 else 0
                total_steps = init_step + steps
                print(
                    f"step {step:6d}/{total_steps}"
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

            if (checkpoint_every > 0 and checkpoint_path is not None
                    and local_step % checkpoint_every == 0):
                raw_net_cp = net._orig_mod if hasattr(net, "_orig_mod") else net
                cp = {
                    "model_state": {kk: vv.cpu()
                                    for kk, vv in raw_net_cp.state_dict().items()},
                    "optimizer_state": opt.state_dict(),
                    "step": step,
                    "T_diff": T_diff,
                    "update_norm": update_norm.state_dict(),
                    "n_aa_types": 21,
                    "hparams": {"hidden": hidden, "layers": layers, "k": k,
                                "lags_ps": list(lags_ps), "point_dim": 6,
                                "node_dim": 24, "edge_dim": 13,
                                "lam": lam, "lam_warmup": lam_warmup,
                                "frame_weighted": frame_weighted,
                                "temp_schedule": temp_schedule,
                                "temp_emb_dim": temp_emb_dim,
                                "reverse_prob": reverse_prob},
                }
                save_path = checkpoint_path.replace("{step}", str(step))
                import os as _os
                _os.makedirs(_os.path.dirname(_os.path.abspath(save_path)),
                             exist_ok=True)
                torch.save(cp, save_path)
                print(f"  [step {step}] checkpoint saved → {save_path}", flush=True)

    # Unwrap compiled model for serialization
    raw_net = net._orig_mod if hasattr(net, "_orig_mod") else net
    return {
        "model_state": {kk: vv.cpu() for kk, vv in raw_net.state_dict().items()},
        "optimizer_state": opt.state_dict(),
        "step": init_step + steps,
        "T_diff": T_diff,
        "update_norm": update_norm.state_dict(),
        "n_aa_types": 21,
        "hparams": {"hidden": hidden, "layers": layers, "k": k,
                    "lags_ps": list(lags_ps), "point_dim": 6,
                    "node_dim": 24, "edge_dim": 13,
                    "lam": lam, "lam_warmup": lam_warmup,
                    "frame_weighted": frame_weighted,
                    "temp_schedule": temp_schedule,
                    "temp_emb_dim": temp_emb_dim,
                    "reverse_prob": reverse_prob},
    }
