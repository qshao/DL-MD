# Plan 2 — Baseline Cross-Protein Training System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one transferable propagator across many proteins and evaluate it zero-shot on held-out proteins, using the already-implemented Plan-1 core.

**Architecture:** A per-protein preprocessor turns trajectories into fixed-vocabulary shards; a homology-aware splitter partitions proteins into train/val/test; a single-GPU trainer samples proteins, builds state-conditional examples, packs them into disjoint-union minibatches under a node cap, normalizes targets, and optimizes the union-graph DDPM loss with AMP + gradient accumulation; a zero-shot evaluator rolls the trained model out from a held-out reference and scores it against reference MD (RMSF-profile correlation, Cα-distance JS, geometry validity) with lower/oracle brackets.

**Tech Stack:** Python, PyTorch (AMP via `torch.amp`), mdtraj, pytest. Reuses the Plan-1 core (`lsmd/vocab.py`, `lsmd/featurize.py`, `lsmd/batching.py`, `lsmd/transfer_model.py`, `lsmd/normalize.py`, `lsmd/data.py`) and `lsmd/validation.py` metrics.

## Global Constraints

- Fixed AA vocabulary everywhere: `lsmd.vocab.residue_indices` (20 canonical + UNK at 20; `N_AA_TYPES = 21`). Shards must store `res_type` keyed through this, never per-protein indices.
- SE(3) frame core dims: `point_dim = 6`, `edge_dim = 13`, `node_dim = 24`. Do not change the Plan-1 modules' signatures.
- τ is physical time in **picoseconds** throughout (`physical_lag_pairs`, `build_training_example`'s `tau`, `tau_embedding`).
- Cross-protein batching is **disjoint union** (`union_collate`): no cross-protein edges; per-graph scalars are `[G]`, broadcast via `batch`.
- Targets are normalized by a single corpus-level `UpdateNorm` (fit once at startup, persisted in the checkpoint); sampling de-normalizes.
- **Non-destructive:** existing single-protein pipeline (`FlowNet`, `scripts/train.py`, `scripts/infer.py`, `scripts/generate_md.py`, committed checkpoints) must keep working. Modifications to existing files are **additive only**.
- Splitting is **by protein**, homology-aware (whole CATH clusters to one split); train/val/test protein sets are disjoint. Test = the zero-shot set.
- Single GPU; modest model (`hidden = 128`, `layers = 4` defaults). All machinery must be testable on synthetic frames + the existing WT trajectory without an ATLAS download.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `lsmd/data.py` | Add `res_names` to `load_frames` return (additive) | Modify |
| `lsmd/atlas.py` | `build_shard` (fixed-vocab per-protein shard) + `download_atlas_entry` wrapper | Create |
| `lsmd/splits.py` | `by_protein_split` (homology-aware, by CATH cluster) | Create |
| `lsmd/transfer_train.py` | `sample_example`, `iter_union_batches`, `fit_update_norm`, `train` | Create |
| `lsmd/transfer_eval.py` | `rollout`, `evaluate` | Create |
| `scripts/train_transfer.py` | CLI around `transfer_train.train` | Create |
| `scripts/eval_transfer.py` | CLI around `transfer_eval` with lower/oracle brackets | Create |
| `tests/test_atlas.py` | `build_shard` shard correctness | Create |
| `tests/test_splits.py` | by-protein / homology-aware split | Create |
| `tests/test_transfer_train.py` | batching cap, update-norm fit, one train step | Create |
| `tests/test_transfer_eval.py` | rollout shape, evaluate finiteness | Create |
| `tests/test_transfer_e2e.py` | preprocessor→split→train-step→rollout→metrics on fixtures | Create |

---

### Task 1: Fixed-vocabulary per-protein shard

**Files:**
- Modify: `lsmd/data.py` (add one key to `load_frames` return; do not change anything else)
- Create: `lsmd/atlas.py`
- Test: `tests/test_atlas.py`

**Interfaces:**
- Consumes: `lsmd.data.load_frames(traj_path, top_path) -> dict` (existing; returns `R[F,N,3,3]`, `t[F,N,3]`, `res_type[N]`, `chain_id[N]`, `res_index[N]`, `n_types`, `mode`). `lsmd.vocab.residue_indices(res_names) -> LongTensor`.
- Produces:
  - `lsmd.atlas.build_shard(traj_path: str, top_path: str, dt: float) -> dict` with keys `R[F,N,3,3]`, `t[F,N,3]`, `res_type[N]` (fixed vocab 0..20), `chain_id[N]`, `res_index[N]`, `dt` (float ps/frame), `seq` (list[str] of 3-letter residue names, length N), `n_res` (int).
  - `lsmd.atlas.download_atlas_entry(pdbid: str, dest_dir: str) -> tuple[str, str, float]` returning `(traj_path, top_path, dt)` — thin network wrapper, **not unit-tested**.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas.py
import os
import torch
import pytest
from lsmd import atlas, vocab


def test_load_frames_exposes_res_names():
    trr, gro = "WT/WT-sol6.trr", "WT/WT-sol6.gro"
    if not (os.path.exists(trr) and os.path.exists(gro)):
        pytest.skip("WT trajectory not present")
    from lsmd import data
    fd = data.load_frames(trr, gro)
    assert "res_names" in fd
    assert len(fd["res_names"]) == fd["res_type"].shape[0]


def test_build_shard_uses_fixed_vocab_and_keys():
    trr, gro = "WT/WT-sol6.trr", "WT/WT-sol6.gro"
    if not (os.path.exists(trr) and os.path.exists(gro)):
        pytest.skip("WT trajectory not present")
    shard = atlas.build_shard(trr, gro, dt=200.0)
    N = shard["n_res"]
    assert shard["R"].shape[1] == N and shard["t"].shape[1] == N
    assert shard["res_type"].shape == (N,)
    # fixed vocab range
    assert int(shard["res_type"].min()) >= 0
    assert int(shard["res_type"].max()) <= vocab.N_AA_TYPES - 1
    # res_type must equal the vocab keying of the stored sequence
    assert torch.equal(shard["res_type"], vocab.residue_indices(shard["seq"]))
    assert shard["dt"] == 200.0
    assert len(shard["seq"]) == N
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_atlas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsmd.atlas'`

- [ ] **Step 3a: Make `load_frames` expose residue names (additive)**

In `lsmd/data.py`, find the `return` at the end of `load_frames` (the dict containing `"R": R, "t": t, "res_type": res_type, ...`). Add `"res_names": res_names` to that dict. `res_names` is the local list already built in the loop. The full edited return:

```python
    return {"R": R, "t": t, "res_type": res_type, "chain_id": chain_id,
            "res_index": res_index, "n_types": len(uniq), "mode": "ca",
            "res_names": res_names}
```

- [ ] **Step 3b: Write `lsmd/atlas.py`**

```python
# lsmd/atlas.py
"""Per-protein preprocessing into fixed-vocabulary shards.

`build_shard` reuses the single-protein frame extraction (`data.load_frames`)
but re-keys residue identities through the global fixed vocabulary
(`vocab.residue_indices`), so residue types are comparable across proteins.
`download_atlas_entry` is a thin network wrapper around the ATLAS dataset.
"""
import os
import torch
from lsmd import data
from lsmd import vocab


def build_shard(traj_path, top_path, dt):
    """Build one fixed-vocab shard from a trajectory + topology.

    Args:
        traj_path: trajectory file path.
        top_path:  topology file path.
        dt:        picoseconds per frame.

    Returns:
        dict with R [F,N,3,3], t [F,N,3], res_type [N] (fixed vocab 0..20),
        chain_id [N], res_index [N], dt (float), seq (list[str]), n_res (int).
    """
    fd = data.load_frames(traj_path, top_path)
    seq = list(fd["res_names"])
    res_type = vocab.residue_indices(seq)
    return {
        "R": fd["R"],
        "t": fd["t"],
        "res_type": res_type,
        "chain_id": fd["chain_id"],
        "res_index": fd["res_index"],
        "dt": float(dt),
        "seq": seq,
        "n_res": len(seq),
    }


def download_atlas_entry(pdbid, dest_dir):
    """Download one ATLAS entry (trajectory + reference) into dest_dir.

    Thin wrapper around the ATLAS analysis-trajectory download. Network I/O;
    not unit-tested. Returns (traj_path, top_path, dt_ps).

    ATLAS analysis trajectories are saved at 10 ps/frame; adjust if the chosen
    ATLAS product differs.
    """
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    base = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis"
    url = f"{base}/{pdbid}/{pdbid}_analysis.zip"
    zip_path = os.path.join(dest_dir, f"{pdbid}.zip")
    urllib.request.urlretrieve(url, zip_path)
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    traj_path = os.path.join(dest_dir, f"{pdbid}_prod_R1_fit.xtc")
    top_path = os.path.join(dest_dir, f"{pdbid}.pdb")
    return traj_path, top_path, 10.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_atlas.py -v`
Expected: PASS (2 passed, or 2 skipped if WT absent)

- [ ] **Step 5: Commit**

```bash
git add lsmd/data.py lsmd/atlas.py tests/test_atlas.py
git commit -m "feat: fixed-vocab per-protein shard preprocessor"
```

---

### Task 2: Homology-aware by-protein split

**Files:**
- Create: `lsmd/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Produces:
  - `lsmd.splits.by_protein_split(cluster_of: dict[str, str], fracs=(0.8, 0.1, 0.1), seed=0) -> dict` with keys `train`, `val`, `test`, each a sorted `list[str]` of protein ids. `cluster_of` maps each protein id → its CATH cluster label. **Whole clusters** go to exactly one split (no cluster spans two splits), so homologous proteins never leak across the train/test boundary.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits.py
from lsmd import splits


def _toy():
    # 10 proteins in 5 clusters (2 each)
    cluster_of = {}
    for c in range(5):
        cluster_of[f"p{2*c}"] = f"cath{c}"
        cluster_of[f"p{2*c+1}"] = f"cath{c}"
    return cluster_of


def test_splits_are_disjoint_and_cover_all():
    s = splits.by_protein_split(_toy(), fracs=(0.6, 0.2, 0.2), seed=0)
    all_ids = set(s["train"]) | set(s["val"]) | set(s["test"])
    assert all_ids == set(_toy().keys())
    assert not (set(s["train"]) & set(s["test"]))
    assert not (set(s["train"]) & set(s["val"]))
    assert not (set(s["val"]) & set(s["test"]))


def test_no_cluster_spans_two_splits():
    cluster_of = _toy()
    s = splits.by_protein_split(cluster_of, fracs=(0.6, 0.2, 0.2), seed=1)
    where = {}
    for name, ids in s.items():
        for pid in ids:
            where[cluster_of[pid]] = where.get(cluster_of[pid], set()) | {name}
    assert all(len(v) == 1 for v in where.values())


def test_deterministic_for_fixed_seed():
    a = splits.by_protein_split(_toy(), seed=3)
    b = splits.by_protein_split(_toy(), seed=3)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsmd.splits'`

- [ ] **Step 3: Write minimal implementation**

```python
# lsmd/splits.py
"""Homology-aware by-protein train/val/test splitting.

Whole CATH clusters are assigned to a single split so homologous proteins never
straddle the train/test boundary (which would inflate zero-shot scores).
"""
import random


def by_protein_split(cluster_of, fracs=(0.8, 0.1, 0.1), seed=0):
    """Partition proteins into train/val/test by whole clusters.

    Args:
        cluster_of: {protein_id: cluster_label}.
        fracs:      (train, val, test) target fractions of proteins.
        seed:       RNG seed for deterministic cluster shuffling.

    Returns:
        {"train": [...], "val": [...], "test": [...]} sorted id lists.
    """
    clusters = {}
    for pid, cl in cluster_of.items():
        clusters.setdefault(cl, []).append(pid)
    labels = sorted(clusters)
    random.Random(seed).shuffle(labels)

    total = len(cluster_of)
    n_train = int(round(fracs[0] * total))
    n_val = int(round(fracs[1] * total))

    out = {"train": [], "val": [], "test": []}
    count = 0
    for cl in labels:
        members = clusters[cl]
        if count < n_train:
            bucket = "train"
        elif count < n_train + n_val:
            bucket = "val"
        else:
            bucket = "test"
        out[bucket].extend(members)
        count += len(members)
    return {k: sorted(v) for k, v in out.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_splits.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/splits.py tests/test_splits.py
git commit -m "feat: homology-aware by-protein split"
```

---

### Task 3: Example sampling + node-capped union batching

**Files:**
- Create: `lsmd/transfer_train.py`
- Test: `tests/test_transfer_train.py`

**Interfaces:**
- Consumes: `lsmd.data.physical_lag_pairs`, `lsmd.data.build_training_example`, `lsmd.batching.union_collate`.
- Produces:
  - `lsmd.transfer_train.sample_example(shard: dict, rng: random.Random, lags_ps: list[float], k: int) -> dict | None` — pick a random valid `(start, tau_frames)` pair via `physical_lag_pairs(shard["t"].shape[0], shard["dt"], lags_ps)`, return `build_training_example(shard, start, tau_frames, k)`; `None` if no valid pair.
  - `lsmd.transfer_train.iter_union_batches(shards: list[dict], rng, lags_ps, k, max_union_nodes: int, n_batches: int) -> Iterator[dict]` — yields `n_batches` union-collated dicts; each packs sampled examples until adding another would exceed `max_union_nodes` (always emits at least one example, even if it alone exceeds the cap).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_train.py
import random
import torch
from lsmd import transfer_train as tt
from lsmd import geometry as g


def _synthetic_shard(F=20, N=10, dt=200.0, seed=0):
    torch.manual_seed(seed)
    return {
        "R": g.so3_exp(torch.randn(F, N, 3) * 0.1),
        "t": torch.randn(F, N, 3) * 5.0,
        "res_type": torch.randint(0, 21, (N,)),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "dt": dt, "seq": ["ALA"] * N, "n_res": N,
    }


def test_sample_example_shapes():
    sh = _synthetic_shard(N=10)
    ex = tt.sample_example(sh, random.Random(0), lags_ps=[200.0, 1000.0], k=4)
    assert ex["node_feats"].shape == (10, 24)
    assert ex["u_target"].shape == (10, 6)
    assert ex["edge_feats"].shape == (10 * 4, 13)


def test_sample_example_none_when_lag_too_large():
    sh = _synthetic_shard(F=3, N=10, dt=200.0)
    ex = tt.sample_example(sh, random.Random(0), lags_ps=[2000.0], k=4)  # 10 frames
    assert ex is None


def test_union_batches_respect_node_cap():
    shards = [_synthetic_shard(N=10, seed=i) for i in range(6)]
    batches = list(tt.iter_union_batches(shards, random.Random(0),
                                         lags_ps=[200.0], k=4,
                                         max_union_nodes=25, n_batches=5))
    assert len(batches) == 5
    for b in batches:
        n = b["node_feats"].shape[0]
        # at most 2 proteins of 10 nodes fit under a 25 cap; 3 would be 30 > 25
        assert n <= 20
        # union keys present and consistent
        assert b["batch"].max().item() + 1 == b["tau"].shape[0]


def test_union_batch_emits_single_oversized_example():
    shards = [_synthetic_shard(N=40, seed=0)]
    batches = list(tt.iter_union_batches(shards, random.Random(0),
                                         lags_ps=[200.0], k=4,
                                         max_union_nodes=25, n_batches=1))
    assert batches[0]["node_feats"].shape[0] == 40  # emitted despite exceeding cap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsmd.transfer_train'`

- [ ] **Step 3: Write minimal implementation**

```python
# lsmd/transfer_train.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_train.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_train.py tests/test_transfer_train.py
git commit -m "feat: example sampling and node-capped union batching"
```

---

### Task 4: Update-norm fit + train loop + CLI

**Files:**
- Modify: `lsmd/transfer_train.py` (append `fit_update_norm`, `train`)
- Create: `scripts/train_transfer.py`
- Test: `tests/test_transfer_train.py` (append)

**Interfaces:**
- Consumes: `lsmd.normalize.UpdateNorm`, `lsmd.transfer_model.PropagatorNet`, `lsmd.transfer_model.ddpm_loss_union`, `lsmd.model.NoiseSchedule`; `sample_example`, `iter_union_batches` (Task 3).
- Produces:
  - `fit_update_norm(shards, rng, lags_ps, k, n_samples: int) -> UpdateNorm` — fit corpus-level scale from `n_samples` sampled examples' `u_target`.
  - `train(shards, *, lags_ps, k=12, hidden=128, layers=4, lr=1e-3, max_union_nodes=2000, accum=4, steps=1000, T_diff=200, norm_samples=256, device="cpu", seed=0) -> dict` — returns a checkpoint dict `{"model_state", "T_diff", "update_norm" (state_dict), "n_aa_types", "hparams"}`. Normalizes each batch's `u_target` via the fitted `UpdateNorm` before the loss. Uses AMP autocast + `GradScaler` when `device` is CUDA, and gradient accumulation over `accum` union-batches per optimizer step.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_train.py  (append)
from lsmd import transfer_train as tt
from lsmd.normalize import UpdateNorm


def test_fit_update_norm_returns_positive_scale():
    shards = [_synthetic_shard(N=10, seed=i) for i in range(4)]
    norm = tt.fit_update_norm(shards, random.Random(0), lags_ps=[200.0],
                              k=4, n_samples=20)
    assert isinstance(norm, UpdateNorm)
    assert norm.scale.shape == (6,)
    assert (norm.scale > 0).all()


def test_train_one_step_returns_checkpoint_without_nans():
    shards = [_synthetic_shard(N=10, seed=i) for i in range(4)]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=2, steps=2, T_diff=20,
                    norm_samples=16, device="cpu", seed=0)
    assert "model_state" in ckpt and "update_norm" in ckpt
    assert ckpt["n_aa_types"] == 21
    for v in ckpt["model_state"].values():
        assert torch.isfinite(v).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_train.py::test_fit_update_norm_returns_positive_scale -v`
Expected: FAIL with `AttributeError: module 'lsmd.transfer_train' has no attribute 'fit_update_norm'`

- [ ] **Step 3: Append implementation to `lsmd/transfer_train.py`**

```python
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
```

- [ ] **Step 4: Write the CLI `scripts/train_transfer.py`**

```python
# scripts/train_transfer.py
"""Train the transferable cross-protein propagator from a directory of shards.

Usage
-----
python scripts/train_transfer.py \\
    --shards_dir data/atlas --split data/atlas/split.json \\
    --lags_ps 200 1000 --steps 20000 --out checkpoints/transfer.pt
"""
import argparse
import glob
import json
import os
import torch
from lsmd import transfer_train as tt


def main():
    ap = argparse.ArgumentParser(description="Train transferable propagator")
    ap.add_argument("--shards_dir", required=True, help="dir of *.pt shards")
    ap.add_argument("--split", default=None,
                    help="split.json with a 'train' id list (optional)")
    ap.add_argument("--lags_ps", type=float, nargs="+", default=[200.0, 1000.0])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_union_nodes", type=int, default=2000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--T_diff", type=int, default=200)
    ap.add_argument("--norm_samples", type=int, default=256)
    ap.add_argument("--out", default="checkpoints/transfer.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    paths = sorted(glob.glob(os.path.join(args.shards_dir, "*.pt")))
    if args.split:
        with open(args.split) as fh:
            train_ids = set(json.load(fh)["train"])
        paths = [p for p in paths
                 if os.path.splitext(os.path.basename(p))[0] in train_ids]
    shards = [torch.load(p, map_location="cpu") for p in paths]
    print(f"Loaded {len(shards)} shards from {args.shards_dir}")

    ckpt = tt.train(shards, lags_ps=args.lags_ps, k=args.k, hidden=args.hidden,
                    layers=args.layers, lr=args.lr,
                    max_union_nodes=args.max_union_nodes, accum=args.accum,
                    steps=args.steps, T_diff=args.T_diff,
                    norm_samples=args.norm_samples, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"Checkpoint saved → {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_transfer_train.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add lsmd/transfer_train.py scripts/train_transfer.py tests/test_transfer_train.py
git commit -m "feat: corpus update-norm fit, cross-protein train loop, CLI"
```

---

### Task 5: Zero-shot rollout

**Files:**
- Create: `lsmd/transfer_eval.py`
- Test: `tests/test_transfer_eval.py`

**Interfaces:**
- Consumes: `lsmd.featurize.frame_graph`, `lsmd.featurize.frame_node_features`, `lsmd.featurize.apply_update`, `lsmd.transfer_model.PropagatorNet`, `lsmd.transfer_model.sample_ddpm_union`, `lsmd.normalize.UpdateNorm`, `lsmd.model.NoiseSchedule`.
- Produces:
  - `lsmd.transfer_eval.load_checkpoint(ckpt: dict, device) -> tuple[PropagatorNet, NoiseSchedule, UpdateNorm]` — rebuild model/schedule/norm from a checkpoint dict produced by Task 4.
  - `lsmd.transfer_eval.rollout(net, schedule, update_norm, R0, t0, res_type, chain_id, res_index, *, steps, tau_ps, k, diff_steps=50, device="cpu") -> Tensor[steps+1, N, 3]` — autoregressive CA trajectory; the graph is rebuilt from the current frames each step (state-conditional), the sampled normalized update is de-normalized, then `apply_update` advances the frames.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_eval.py
import torch
from lsmd import transfer_eval as te
from lsmd import transfer_train as tt
from lsmd import geometry as g


def _synthetic_shard(F=20, N=10, dt=200.0, seed=0):
    torch.manual_seed(seed)
    return {
        "R": g.so3_exp(torch.randn(F, N, 3) * 0.1),
        "t": torch.randn(F, N, 3) * 5.0,
        "res_type": torch.randint(0, 21, (N,)),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "dt": dt, "seq": ["ALA"] * N, "n_res": N,
    }


def test_rollout_shape_and_finite():
    import random
    shards = [_synthetic_shard(N=10, seed=i) for i in range(3)]
    ckpt = tt.train(shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=1, steps=2, T_diff=20,
                    norm_samples=16, device="cpu", seed=0)
    net, sched, norm = te.load_checkpoint(ckpt, device="cpu")
    sh = shards[0]
    traj = te.rollout(net, sched, norm, sh["R"][0], sh["t"][0],
                      sh["res_type"], sh["chain_id"], sh["res_index"],
                      steps=4, tau_ps=200.0, k=4, diff_steps=3, device="cpu")
    assert traj.shape == (5, 10, 3)
    assert torch.isfinite(traj).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsmd.transfer_eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# lsmd/transfer_eval.py
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

    Returns:
        [steps+1, N, 3] CA positions (frame 0 = reference).
    """
    device = torch.device(device)
    R = R0.to(device)
    t = t0.to(device)
    res_type = res_type.to(device)
    chain_id = chain_id.to(device)
    res_index = res_index.to(device)
    N = t.shape[0]

    node_feats = feat.frame_node_features(res_type, chain_id, res_index)
    scale = update_norm.scale.to(device)
    tau = torch.tensor([float(tau_ps)], device=device)

    traj = [t.clone()]
    for _ in range(steps):
        edge_index, edge_feats = feat.frame_graph(R, t, k)
        batch = torch.zeros(N, dtype=torch.long, device=device)
        u = sample_ddpm_union(net, node_feats, edge_index, edge_feats,
                              tau, batch, schedule, steps=diff_steps)
        u = u * scale
        R, t = feat.apply_update(R, t, u)
        traj.append(t.clone())
    return torch.stack(traj, dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_eval.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_eval.py tests/test_transfer_eval.py
git commit -m "feat: zero-shot state-conditional rollout"
```

---

### Task 6: Evaluation metrics + CLI with brackets

**Files:**
- Modify: `lsmd/transfer_eval.py` (append `evaluate`)
- Create: `scripts/eval_transfer.py`
- Test: `tests/test_transfer_eval.py` (append)

**Interfaces:**
- Consumes: `lsmd.validation.rmsf_profile`, `lsmd.validation.distance_matrix_js`, `lsmd.validation.ca_geometry`; `load_checkpoint`, `rollout` (Task 5).
- Produces:
  - `evaluate(ca_model: Tensor[K,N,3], ca_md: Tensor[M,N,3]) -> dict` with keys `rmsf_corr` (float), `dist_js` (float), `ca_bond_mean` (float), `clash_count` (float) — the last two averaged over the model ensemble frames.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_eval.py  (append)
from lsmd import transfer_eval as te


def test_evaluate_keys_and_finite():
    torch.manual_seed(0)
    ca_model = torch.randn(8, 10, 3) * 2.0
    ca_md = torch.randn(12, 10, 3) * 2.0
    m = te.evaluate(ca_model, ca_md)
    for key in ("rmsf_corr", "dist_js", "ca_bond_mean", "clash_count"):
        assert key in m
    assert -1.0 <= m["rmsf_corr"] <= 1.0
    assert 0.0 <= m["dist_js"] <= 1.0
    assert torch.isfinite(torch.tensor(m["ca_bond_mean"]))


def test_evaluate_identical_ensembles_have_high_rmsf_corr():
    torch.manual_seed(1)
    ca = torch.randn(10, 12, 3) * 2.0
    m = te.evaluate(ca, ca)
    assert m["rmsf_corr"] > 0.99
    assert m["dist_js"] < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_eval.py::test_evaluate_keys_and_finite -v`
Expected: FAIL with `AttributeError: module 'lsmd.transfer_eval' has no attribute 'evaluate'`

- [ ] **Step 3: Append implementation to `lsmd/transfer_eval.py`**

```python
def evaluate(ca_model, ca_md):
    """Score a generated CA ensemble against reference MD.

    Args:
        ca_model: [K, N, 3] generated CA frames.
        ca_md:    [M, N, 3] reference MD CA frames.

    Returns:
        dict: rmsf_corr, dist_js, ca_bond_mean, clash_count.
    """
    rmsf = val.rmsf_profile(ca_model, ca_md)
    dist_js = val.distance_matrix_js(ca_model, ca_md)
    bond_means, clashes = [], []
    for fr in ca_model:
        geo = val.ca_geometry(fr)
        bond_means.append(geo["ca_bond_mean"])
        clashes.append(geo["clash_count"])
    return {
        "rmsf_corr": rmsf["corr"],
        "dist_js": dist_js,
        "ca_bond_mean": float(sum(bond_means) / len(bond_means)),
        "clash_count": float(sum(clashes) / len(clashes)),
    }
```

- [ ] **Step 4: Write the CLI `scripts/eval_transfer.py`**

```python
# scripts/eval_transfer.py
"""Zero-shot evaluation of a transferable checkpoint on a held-out shard.

Rolls the model out from the held-out reference and scores it against the
shard's own MD frames. If --oracle / --lower checkpoints are given, the same
held-out protein is scored under those models too, bracketing the result.

Usage
-----
python scripts/eval_transfer.py \\
    --checkpoint checkpoints/transfer.pt --shard data/atlas/1abc.pt \\
    --steps 200 --tau_ps 1000 --out eval_1abc.json
"""
import argparse
import json
import torch
from lsmd import transfer_eval as te


def _run(ckpt, shard, steps, tau_ps, k, diff_steps, device):
    net, sched, norm = te.load_checkpoint(ckpt, device=device)
    k_eff = ckpt["hparams"].get("k", k)
    traj = te.rollout(net, sched, norm, shard["R"][0], shard["t"][0],
                      shard["res_type"], shard["chain_id"], shard["res_index"],
                      steps=steps, tau_ps=tau_ps, k=k_eff,
                      diff_steps=diff_steps, device=device)
    return te.evaluate(traj, shard["t"])


def main():
    ap = argparse.ArgumentParser(description="Zero-shot eval of transferable model")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", required=True, help="held-out shard .pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tau_ps", type=float, default=1000.0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--diff_steps", type=int, default=50)
    ap.add_argument("--oracle", default=None, help="per-protein checkpoint (upper bracket)")
    ap.add_argument("--lower", default=None, help="marginal-prior checkpoint (lower bracket)")
    ap.add_argument("--out", default="eval.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    shard = torch.load(args.shard, map_location="cpu")

    report = {"model": _run(torch.load(args.checkpoint, map_location="cpu"),
                            shard, args.steps, args.tau_ps, args.k,
                            args.diff_steps, device)}
    if args.oracle:
        report["oracle"] = _run(torch.load(args.oracle, map_location="cpu"),
                                shard, args.steps, args.tau_ps, args.k,
                                args.diff_steps, device)
    if args.lower:
        report["lower"] = _run(torch.load(args.lower, map_location="cpu"),
                               shard, args.steps, args.tau_ps, args.k,
                               args.diff_steps, device)

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_transfer_eval.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add lsmd/transfer_eval.py scripts/eval_transfer.py tests/test_transfer_eval.py
git commit -m "feat: zero-shot evaluation metrics and CLI with brackets"
```

---

### Task 7: End-to-end baseline integration

**Files:**
- Create: `tests/test_transfer_e2e.py`

**Interfaces:**
- Consumes: everything above — `atlas.build_shard` (or synthetic shards), `splits.by_protein_split`, `transfer_train.train`, `transfer_eval.load_checkpoint`/`rollout`/`evaluate`.

This task adds no library code; it proves the baseline system composes on synthetic shards (always runs) and, if WT is present, that a real shard feeds the trainer. It also confirms the full suite stays green (non-destructive).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_e2e.py
import os
import random
import torch
import pytest
from lsmd import transfer_train as tt
from lsmd import transfer_eval as te
from lsmd import splits, geometry as g


def _synthetic_shard(F=20, N=10, dt=200.0, seed=0):
    torch.manual_seed(seed)
    return {
        "R": g.so3_exp(torch.randn(F, N, 3) * 0.1),
        "t": torch.randn(F, N, 3) * 5.0,
        "res_type": torch.randint(0, 21, (N,)),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "dt": dt, "seq": ["ALA"] * N, "n_res": N,
    }


def test_baseline_pipeline_composes_on_synthetic():
    cluster_of = {f"p{i}": f"c{i // 2}" for i in range(6)}
    sp = splits.by_protein_split(cluster_of, fracs=(0.66, 0.17, 0.17), seed=0)
    assert sp["train"] and sp["test"]

    shards = {f"p{i}": _synthetic_shard(N=10, seed=i) for i in range(6)}
    train_shards = [shards[i] for i in sp["train"]]
    ckpt = tt.train(train_shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=2, steps=3, T_diff=20,
                    norm_samples=16, device="cpu", seed=0)

    held = shards[sp["test"][0]]
    net, sched, norm = te.load_checkpoint(ckpt, device="cpu")
    traj = te.rollout(net, sched, norm, held["R"][0], held["t"][0],
                      held["res_type"], held["chain_id"], held["res_index"],
                      steps=4, tau_ps=200.0, k=4, diff_steps=3, device="cpu")
    metrics = te.evaluate(traj, held["t"])
    assert -1.0 <= metrics["rmsf_corr"] <= 1.0
    assert torch.isfinite(torch.tensor(metrics["dist_js"]))


def test_real_wt_shard_feeds_trainer_if_available():
    trr, gro = "WT/WT-sol6.trr", "WT/WT-sol6.gro"
    if not (os.path.exists(trr) and os.path.exists(gro)):
        pytest.skip("WT trajectory not present")
    from lsmd import atlas
    shard = atlas.build_shard(trr, gro, dt=200.0)
    ckpt = tt.train([shard], lags_ps=[200.0, 1000.0], k=8, hidden=16, layers=2,
                    max_union_nodes=10_000, accum=1, steps=2, T_diff=20,
                    norm_samples=8, device="cpu", seed=0)
    assert ckpt["n_aa_types"] == 21
```

- [ ] **Step 2: Run test to verify it fails (if upstream incomplete) or passes**

Run: `pytest tests/test_transfer_e2e.py -v`
Expected: PASS once Tasks 1–6 are committed (1 passed + 1 passed/ skipped depending on WT). If it errors with an import/attribute error, the named upstream module is incomplete.

- [ ] **Step 3: (no new implementation)**

All library code exists from Tasks 1–6. If the test fails, fix the specific upstream module it points to.

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -q`
Expected: all pass (Plan-1 and single-protein tests unaffected — Global Constraint: non-destructive).

- [ ] **Step 5: Commit**

```bash
git add tests/test_transfer_e2e.py
git commit -m "test: end-to-end baseline cross-protein training integration"
```

---

## Self-Review

**Spec coverage:**
- `atlas.py` preprocessor with fixed-vocab `res_type` via `vocab.residue_indices` → Task 1. ✓ (closes the cross-cutting load_frames keying gap)
- `splits.py` homology-aware by-protein split (whole CATH clusters, disjoint) → Task 2. ✓
- Trainer: protein sampler, `union_collate` minibatches, **ΣN cap** → Tasks 3–4. ✓
- Trainer: `UpdateNorm.fit` at startup + target normalization, AMP, gradient accumulation, checkpoint (model + schedule cfg + update_norm + vocab size + hparams) → Task 4. ✓
- Eval: rollout from reference (dynamic graph each step, `sample_ddpm_union` G=1, `apply_update`), no re-anchoring → Task 5. ✓
- Eval metrics: RMSF-profile correlation (headline), Cα-distance JS, geometry validity → Task 6. ✓
- Eval brackets: lower (marginal prior) + oracle (per-protein) wired in the CLI → Task 6. ✓
- Physical-τ in ps throughout → Tasks 3–6 (`lags_ps`, `tau_ps`). ✓
- Non-destructive (single-protein pipeline untouched; only additive `res_names` key) → verified by full suite in Task 7. ✓
- Out of scope (Plans 3/4): encoder/denoiser split, physics losses/guidance — not in this plan. ✓

**Type consistency:** Shard dict keys (`R`, `t`, `res_type`, `chain_id`, `res_index`, `dt`, `seq`, `n_res`) are identical across `atlas.build_shard`, `transfer_train.sample_example`, and `transfer_eval.rollout` inputs. Checkpoint keys (`model_state`, `T_diff`, `update_norm`, `n_aa_types`, `hparams` with `node_dim`/`edge_dim`/`hidden`/`layers`/`point_dim`/`k`) match between `transfer_train.train` (producer) and `transfer_eval.load_checkpoint` (consumer). `point_dim=6`, `edge_dim=13`, `node_dim=24`, `n_aa_types=21` consistent throughout. `evaluate` returns `rmsf_corr`/`dist_js`/`ca_bond_mean`/`clash_count` used identically in tests and CLI.

**Placeholder scan:** No TBD/TODO; every code step is complete and runnable. `download_atlas_entry` is intentionally not unit-tested (network I/O) and is the only piece exercised solely in production — its body is complete, not a stub.

---

## Follow-up

Once this baseline is green and a zero-shot RMSF-correlation number exists, proceed to **Plan 3** (efficient rollout: `StructuralEncoder`/`Denoiser` split, equivalence-tested, default 1 denoiser message layer; reduced-step DDIM), then **Plan 4** (physics-aware, staged C1→C2→C3), per the design spec `2026-06-20-transferable-training-system-design.md`.
