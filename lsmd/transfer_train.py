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
