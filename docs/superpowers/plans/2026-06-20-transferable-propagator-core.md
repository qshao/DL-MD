# Transferable Propagator — Model Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the transferable, state-conditional protein-dynamics propagator core — fixed AA vocabulary, SE(3) frame featurization from the *current* conformation, disjoint-union cross-protein batching, a union-graph DDPM network, physical-time τ training pairs, and corpus-level update normalization — all unit-testable on synthetic data and the existing WT protein, with no external corpus.

**Architecture:** Each residue is an SE(3) backbone frame (R, t). At every training pair and rollout step the graph is rebuilt from the *current* frames (invariant 13-dim edge features, 24-dim structure+AA node features). The network predicts a per-residue local rigid update `[trans(3), axis-angle(3)]` via DDPM ε-prediction, conditioned on the current local environment + physical lag τ. Multiple proteins train together via a flat disjoint-union graph (`[ΣN, H]` nodes + a `batch` vector), so variable-size proteins mix in one forward pass.

**Tech Stack:** Python, PyTorch, mdtraj, pytest. Reuses `lsmd/geometry.py` (`build_frames`, `so3_exp/log`, `kabsch`), `lsmd/featurize.py` (`relative_update`, `apply_update`, `edge_features`, `knn_graph`), and the `NoiseSchedule`/`tau_embedding` from `lsmd/model.py`.

## Global Constraints

- Representation: SE(3) backbone frames; per-residue update `point_dim = 6` = `[local_trans(3), axis_angle(3)]`.
- Edge features: `edge_dim = 13` = `[rel_pos(3), dist(1), rel_R(9)]`, built from the **current** frames (state-conditional), E(3)-invariant.
- Node features: `node_dim = 24` = `[AA one-hot incl. UNK (21), chain_id (1), residue-index PE sin/cos (2)]`, in exactly that concatenation order.
- Fixed AA vocabulary: canonical 20 + UNK at index 20; `N_AA_TYPES = 21`; identical indexing for every protein.
- τ is physical time in **picoseconds**, fed to `tau_embedding` (which already takes `log(τ)`).
- Cross-protein batching is **disjoint union**: flat `[ΣN, …]` tensors + a `[ΣN]` long `batch` vector; no cross-protein edges; per-graph scalars are `[G]` and broadcast via `batch`.
- **Non-destructive:** do not modify `FlowNet`/`MessageLayer` or the existing single-protein pipeline (`infer.py`, `generate_md.py`, committed checkpoints keep working). New code lives in new modules.
- Conditioning is structure + AA only — no ESM, no MSA.
- Single GPU; keep the model size modest (`hidden = 128`, `layers = 4` defaults).

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `lsmd/vocab.py` | Fixed AA vocabulary: name→index, `residue_indices`, `N_AA_TYPES` | Create |
| `lsmd/featurize.py` | Add `frame_graph` (dynamic graph from current frames) and `frame_node_features` | Modify |
| `lsmd/batching.py` | Disjoint-union collation of per-protein graphs | Create |
| `lsmd/transfer_model.py` | `UnionMessageLayer`, `PropagatorNet`, `ddpm_loss_union`, `sample_ddpm_union` | Create |
| `lsmd/normalize.py` | `UpdateNorm` corpus-level update scaling | Create |
| `lsmd/data.py` | Add `physical_lag_pairs` and `build_training_example` (state-conditional, fixed-vocab) | Modify |
| `tests/test_vocab.py` | Vocab unit tests | Create |
| `tests/test_frame_features.py` | `frame_graph` / `frame_node_features` tests | Create |
| `tests/test_batching.py` | Union-collation tests | Create |
| `tests/test_transfer_model.py` | Network shape, batch-independence, loss, sampler tests | Create |
| `tests/test_normalize.py` | `UpdateNorm` tests | Create |
| `tests/test_propagator_pairs.py` | Physical-τ pairs + state-conditional example builder | Create |
| `tests/test_propagator_integration.py` | End-to-end single step on WT protein (synthetic frames fallback) | Create |

---

### Task 1: Fixed AA vocabulary

**Files:**
- Create: `lsmd/vocab.py`
- Test: `tests/test_vocab.py`

**Interfaces:**
- Produces:
  - `N_AA_TYPES: int = 21`
  - `UNK_INDEX: int = 20`
  - `residue_indices(res_names: list[str]) -> torch.LongTensor` — maps 3-letter residue names (case-insensitive, common protonation aliases handled) to indices `0..20`; unknown → `UNK_INDEX`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vocab.py
import torch
from lsmd import vocab


def test_canonical_twenty_are_distinct():
    canon = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
             "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
             "TYR", "VAL"]
    idx = vocab.residue_indices(canon)
    assert idx.tolist() == list(range(20))
    assert vocab.N_AA_TYPES == 21


def test_aliases_map_to_canonical():
    # protonation / naming variants collapse onto their parent residue
    idx = vocab.residue_indices(["HIE", "HID", "HIP", "CYX"])
    his = vocab.residue_indices(["HIS"])[0].item()
    cys = vocab.residue_indices(["CYS"])[0].item()
    assert idx.tolist() == [his, his, his, cys]


def test_unknown_maps_to_unk_and_is_case_insensitive():
    idx = vocab.residue_indices(["XYZ", "ala"])
    assert idx[0].item() == vocab.UNK_INDEX == 20
    assert idx[1].item() == vocab.residue_indices(["ALA"])[0].item()
    assert idx.dtype == torch.long
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vocab.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsmd.vocab'`

- [ ] **Step 3: Write minimal implementation**

```python
# lsmd/vocab.py
"""Fixed amino-acid vocabulary shared across all proteins.

Identical indexing for every protein so residue identity is comparable
across the training corpus. Index 20 is the UNK catch-all.
"""
import torch

CANONICAL = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
             "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
             "TYR", "VAL"]

UNK_INDEX = len(CANONICAL)          # 20
N_AA_TYPES = len(CANONICAL) + 1     # 21 (20 canonical + UNK)

# Common protonation / naming variants → canonical parent.
_ALIASES = {
    "HIE": "HIS", "HID": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
    "HSP": "HIS", "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU",
    "LYN": "LYS", "ARN": "ARG", "MSE": "MET",
}

_INDEX = {name: i for i, name in enumerate(CANONICAL)}


def residue_indices(res_names):
    """Map 3-letter residue names to fixed vocabulary indices.

    Args:
        res_names: iterable of residue name strings (any case).

    Returns:
        LongTensor [len(res_names)] with values in 0..20; unknown → UNK_INDEX.
    """
    out = []
    for nm in res_names:
        key = str(nm).strip().upper()
        key = _ALIASES.get(key, key)
        out.append(_INDEX.get(key, UNK_INDEX))
    return torch.tensor(out, dtype=torch.long)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vocab.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/vocab.py tests/test_vocab.py
git commit -m "feat: fixed cross-protein AA vocabulary"
```

---

### Task 2: Frame-based dynamic graph + node features

**Files:**
- Modify: `lsmd/featurize.py` (append two functions; do not touch existing ones)
- Test: `tests/test_frame_features.py`

**Interfaces:**
- Consumes: `lsmd.vocab.N_AA_TYPES` (Task 1); existing `knn_graph`, `edge_features`.
- Produces:
  - `frame_graph(R: Tensor[N,3,3], t: Tensor[N,3], k: int) -> (edge_index Tensor[2,E], edge_feats Tensor[E,13])` — graph built from the **current** frames; edge features are E(3)-invariant.
  - `frame_node_features(res_type: LongTensor[N], chain_id: LongTensor[N], res_index: LongTensor[N], n_types: int = N_AA_TYPES) -> Tensor[N,24]` — concatenation order `[one_hot(21), chain(1), PE(2)]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frame_features.py
import torch
from lsmd import geometry as g
from lsmd import featurize as f
from lsmd import vocab


def _rand_frames(n):
    R = g.so3_exp(torch.randn(n, 3) * 0.5)
    t = torch.randn(n, 3) * 5.0
    return R, t


def test_frame_graph_shapes_and_edge_dim():
    R, t = _rand_frames(20)
    ei, ef = f.frame_graph(R, t, k=6)
    assert ei.shape == (2, 20 * 6)
    assert ef.shape == (20 * 6, 13)


def test_frame_graph_edges_are_invariant():
    R, t = _rand_frames(12)
    ei, ef = f.frame_graph(R, t, k=4)
    # Apply a global rigid transform; the SAME kNN topology, invariant feats.
    Rg = g.so3_exp(torch.tensor([[0.2, -0.4, 0.6]])).expand(12, 3, 3)
    tg = torch.tensor([3.0, -2.0, 1.0])
    R2 = Rg @ R
    t2 = (Rg @ t.unsqueeze(-1)).squeeze(-1) + tg
    ei2, ef2 = f.frame_graph(R2, t2, k=4)
    assert torch.equal(ei, ei2)            # topology unchanged by rigid motion
    assert torch.allclose(ef, ef2, atol=1e-4)


def test_frame_node_features_shape_and_layout():
    res_type = torch.tensor([0, 20, 7])      # ALA, UNK, GLY
    chain_id = torch.tensor([0, 0, 1])
    res_index = torch.tensor([0, 1, 2])
    nf = f.frame_node_features(res_type, chain_id, res_index)
    assert nf.shape == (3, 24)
    # first 21 columns are the one-hot block
    assert torch.allclose(nf[0, :vocab.N_AA_TYPES],
                          torch.nn.functional.one_hot(torch.tensor(0), 21).float())
    assert nf[1, vocab.UNK_INDEX].item() == 1.0
    # chain id sits at column 21
    assert nf[2, vocab.N_AA_TYPES].item() == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_features.py -v`
Expected: FAIL with `AttributeError: module 'lsmd.featurize' has no attribute 'frame_graph'`

- [ ] **Step 3: Write minimal implementation**

Append to `lsmd/featurize.py`:

```python
from lsmd.vocab import N_AA_TYPES


def frame_graph(R, t, k):
    """kNN graph + invariant edge features from the CURRENT frames.

    State-conditional: the graph is rebuilt from (R, t) at the current step,
    so the network sees local geometry as it actually is.

    Args:
        R: per-residue rotations [N, 3, 3]
        t: per-residue translations (CA positions) [N, 3]
        k: neighbours per node.

    Returns:
        edge_index [2, E], edge_feats [E, 13] = [rel_pos(3), dist(1), rel_R(9)].
    """
    edge_index = knn_graph(t, k)
    edge_feats = edge_features(R, t, edge_index)
    return edge_index, edge_feats


def frame_node_features(res_type, chain_id, res_index, n_types=N_AA_TYPES):
    """Structure+AA node features: [one_hot(n_types), chain(1), PE(2)].

    Args:
        res_type:  [N] long, fixed-vocab indices 0..n_types-1.
        chain_id:  [N] long.
        res_index: [N] long, sequential residue index.
        n_types:   vocabulary size (default N_AA_TYPES = 21).

    Returns:
        [N, n_types + 3] float tensor.
    """
    rt = F_nn.one_hot(res_type, num_classes=n_types).float()
    ch = chain_id.float().unsqueeze(-1)
    pos = res_index.float().unsqueeze(-1)
    pe = torch.cat([torch.sin(pos / 100.0), torch.cos(pos / 100.0)], dim=-1)
    return torch.cat([rt, ch, pe], dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_features.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/featurize.py tests/test_frame_features.py
git commit -m "feat: state-conditional frame graph and structure+AA node features"
```

---

### Task 3: Disjoint-union batching

**Files:**
- Create: `lsmd/batching.py`
- Test: `tests/test_batching.py`

**Interfaces:**
- Produces:
  - `union_collate(graphs: list[dict]) -> dict` where each input graph has keys
    `node_feats[N,F]`, `edge_index[2,E]`, `edge_feats[E,De]`, `u_target[N,P]`, `tau` (float).
    Output keys: `node_feats[ΣN,F]`, `edge_index[2,ΣE]` (offset into ΣN),
    `edge_feats[ΣE,De]`, `u_target[ΣN,P]`, `batch[ΣN]` long, `tau[G]` float.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batching.py
import torch
from lsmd import batching


def _toy_graph(n, e, f=24, p=6, de=13, tau=100.0):
    return {
        "node_feats": torch.randn(n, f),
        "edge_index": torch.randint(0, n, (2, e)),
        "edge_feats": torch.randn(e, de),
        "u_target": torch.randn(n, p),
        "tau": tau,
    }


def test_union_concatenates_and_offsets():
    g0 = _toy_graph(5, 8, tau=50.0)
    g1 = _toy_graph(7, 10, tau=200.0)
    u = batching.union_collate([g0, g1])
    assert u["node_feats"].shape == (12, 24)
    assert u["u_target"].shape == (12, 6)
    assert u["edge_feats"].shape == (18, 13)
    assert u["edge_index"].shape == (2, 18)
    # batch vector tags nodes by graph
    assert u["batch"].tolist() == [0] * 5 + [1] * 7
    assert torch.equal(u["tau"], torch.tensor([50.0, 200.0]))


def test_second_graph_edges_are_offset_into_union():
    g0 = _toy_graph(5, 8)
    g1 = _toy_graph(7, 10)
    u = batching.union_collate([g0, g1])
    # graph-1 edges must index nodes 5..11, never 0..4
    second = u["edge_index"][:, 8:]
    assert second.min().item() >= 5
    assert second.max().item() <= 11


def test_no_cross_graph_edges():
    g0 = _toy_graph(5, 8)
    g1 = _toy_graph(7, 10)
    u = batching.union_collate([g0, g1])
    src, dst = u["edge_index"]
    # every edge stays within one graph
    assert torch.equal(u["batch"][src], u["batch"][dst])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batching.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsmd.batching'`

- [ ] **Step 3: Write minimal implementation**

```python
# lsmd/batching.py
"""Disjoint-union batching of per-protein graphs.

Concatenates several proteins into one flat graph so variable-size proteins
train together in a single forward pass. Edges are offset into the union so
no edge ever crosses a protein boundary; a `batch` vector records each node's
protein index for broadcasting per-graph scalars (flow-time, tau).
"""
import torch


def union_collate(graphs):
    """Collate a list of per-protein graph dicts into one union graph.

    Args:
        graphs: list of dicts with keys node_feats [N,F], edge_index [2,E],
                edge_feats [E,De], u_target [N,P], tau (float).

    Returns:
        dict with node_feats [ΣN,F], edge_index [2,ΣE], edge_feats [ΣE,De],
        u_target [ΣN,P], batch [ΣN] long, tau [G] float.
    """
    node_feats, edge_feats, u_target = [], [], []
    edge_index, batch, taus = [], [], []
    offset = 0
    for i, gr in enumerate(graphs):
        n = gr["node_feats"].shape[0]
        node_feats.append(gr["node_feats"])
        edge_feats.append(gr["edge_feats"])
        u_target.append(gr["u_target"])
        edge_index.append(gr["edge_index"] + offset)
        batch.append(torch.full((n,), i, dtype=torch.long))
        taus.append(float(gr["tau"]))
        offset += n
    return {
        "node_feats": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_index, dim=1),
        "edge_feats": torch.cat(edge_feats, dim=0),
        "u_target": torch.cat(u_target, dim=0),
        "batch": torch.cat(batch, dim=0),
        "tau": torch.tensor(taus, dtype=torch.float32),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_batching.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/batching.py tests/test_batching.py
git commit -m "feat: disjoint-union batching of per-protein graphs"
```

---

### Task 4: Union-graph propagator network + DDPM loss + sampler

**Files:**
- Create: `lsmd/transfer_model.py`
- Test: `tests/test_transfer_model.py`

**Interfaces:**
- Consumes: `lsmd.model.tau_embedding`, `lsmd.model.NoiseSchedule`; union dict from Task 3.
- Produces:
  - `UnionMessageLayer(hidden, edge_dim)` — flat `[ΣN, H]` message passing over a union `edge_index`.
  - `PropagatorNet(node_dim=24, edge_dim=13, hidden=128, layers=4, tau_emb_dim=16, point_dim=6)` with
    `forward(u[ΣN,P], s[G], node_feats[ΣN,F], edge_index[2,E], edge_feats[E,De], tau[G], batch[ΣN]) -> [ΣN,P]`.
  - `ddpm_loss_union(net, u_target, node_feats, edge_index, edge_feats, tau, batch, schedule) -> scalar`.
  - `sample_ddpm_union(net, node_feats, edge_index, edge_feats, tau, batch, schedule, steps=50, eta=1.0, sigma_init=1.0) -> [ΣN,P]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_model.py
import torch
from lsmd import batching
from lsmd.model import NoiseSchedule
from lsmd import transfer_model as tm


def _toy_graph(n, k=4, tau=100.0):
    R_t = torch.randn(n, 24)
    e = n * k
    return {
        "node_feats": R_t,
        "edge_index": torch.randint(0, n, (2, e)),
        "edge_feats": torch.randn(e, 13),
        "u_target": torch.randn(n, 6),
        "tau": tau,
    }


def test_forward_shape():
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    net = tm.PropagatorNet(hidden=32, layers=2)
    s = torch.rand(2)
    out = net(u["u_target"], s, u["node_feats"], u["edge_index"],
              u["edge_feats"], u["tau"], u["batch"])
    assert out.shape == (13, 6)


def test_graphs_do_not_interact():
    # Output on graph-0 nodes is identical whether or not graph-1 is in the batch
    torch.manual_seed(0)
    g0 = _toy_graph(5, tau=100.0)
    g1 = _toy_graph(8, tau=100.0)
    net = tm.PropagatorNet(hidden=32, layers=2).eval()

    solo = batching.union_collate([g0])
    pair = batching.union_collate([g0, g1])
    with torch.no_grad():
        out_solo = net(solo["u_target"], torch.tensor([0.3]),
                       solo["node_feats"], solo["edge_index"],
                       solo["edge_feats"], solo["tau"], solo["batch"])
        out_pair = net(pair["u_target"], torch.tensor([0.3, 0.7]),
                       pair["node_feats"], pair["edge_index"],
                       pair["edge_feats"], pair["tau"], pair["batch"])
    assert torch.allclose(out_solo, out_pair[:5], atol=1e-5)


def test_ddpm_loss_is_finite_scalar():
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    net = tm.PropagatorNet(hidden=32, layers=2)
    sched = NoiseSchedule(T=50)
    loss = tm.ddpm_loss_union(net, u["u_target"], u["node_feats"],
                              u["edge_index"], u["edge_feats"], u["tau"],
                              u["batch"], sched)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_sampler_shape_single_graph():
    g0 = _toy_graph(7)
    u = batching.union_collate([g0])
    net = tm.PropagatorNet(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)
    out = tm.sample_ddpm_union(net, u["node_feats"], u["edge_index"],
                               u["edge_feats"], u["tau"], u["batch"], sched,
                               steps=5)
    assert out.shape == (7, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsmd.transfer_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# lsmd/transfer_model.py
"""Union-graph propagator network for transferable protein dynamics.

Operates on a flat disjoint-union graph ([ΣN, ...] nodes + a batch vector),
so multiple proteins of different sizes train in one forward pass. Predicts a
per-residue SE(3) local update via DDPM epsilon-prediction, conditioned on the
current-state graph and physical lag tau.
"""
import torch
import torch.nn as nn
from lsmd.model import tau_embedding


def _scatter_mean(src, index, dim_size):
    """Mean of `src` rows grouped by `index` (0..dim_size-1)."""
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp_min(1.0).reshape(-1, *([1] * (src.dim() - 1)))


class UnionMessageLayer(nn.Module):
    """Flat message-passing layer over a union edge_index ([ΣN, H] nodes)."""

    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h, edge_index, edge_feats):
        src, dst = edge_index                       # [E]
        N, H = h.shape
        msg = self.msg(torch.cat([h[src], h[dst], edge_feats], dim=-1))  # [E,H]
        agg = torch.zeros(N, H, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(N, 1, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        return h + self.upd(torch.cat([h, agg], dim=-1))


class PropagatorNet(nn.Module):
    """DDPM epsilon-predictor over a union graph."""

    def __init__(self, node_dim=24, edge_dim=13, hidden=128, layers=4,
                 tau_emb_dim=16, point_dim=6):
        super().__init__()
        self.tau_emb_dim = tau_emb_dim
        self.point_dim = point_dim
        self.embed = nn.Linear(node_dim + point_dim + 1 + tau_emb_dim, hidden)
        self.layers = nn.ModuleList(
            [UnionMessageLayer(hidden, edge_dim) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, point_dim),
        )

    def forward(self, u, s, node_feats, edge_index, edge_feats, tau, batch):
        """Predict per-node epsilon.

        Args:
            u:          [ΣN, point_dim] noisy update.
            s:          [G] flow-time per graph.
            node_feats: [ΣN, node_dim]
            edge_index: [2, ΣE] (union indices)
            edge_feats: [ΣE, edge_dim]
            tau:        [G] physical lag (ps) per graph.
            batch:      [ΣN] long, node→graph.

        Returns:
            [ΣN, point_dim]
        """
        s = torch.as_tensor(s, dtype=u.dtype, device=u.device)
        s_nodes = s[batch].unsqueeze(-1)                       # [ΣN,1]
        tau_emb = tau_embedding(tau, dim=self.tau_emb_dim,
                                device=u.device, dtype=u.dtype)  # [G, tau_dim]
        tau_nodes = tau_emb[batch]                             # [ΣN, tau_dim]
        h = self.embed(torch.cat([node_feats, u, s_nodes, tau_nodes], dim=-1))
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)
        return self.head(h)


def ddpm_loss_union(net, u_target, node_feats, edge_index, edge_feats, tau,
                    batch, schedule, graph_weights=None):
    """DDPM epsilon-prediction loss over a union batch.

    Each graph gets its own noise level; per-graph node-mean losses are then
    averaged (optionally weighted) so large proteins don't dominate.
    """
    G = tau.shape[0]
    T = schedule.T
    t_min = max(1, T // 20)
    t_idx = torch.randint(t_min, T + 1, (G,), device=u_target.device)   # [G]
    t_nodes = t_idx[batch]                                              # [ΣN]
    eps = torch.randn_like(u_target)

    sqrt_ab = schedule.sqrt_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    sqrt_1mab = schedule.sqrt_one_minus_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    noisy = sqrt_ab * u_target + sqrt_1mab * eps

    s = (t_idx.float() / T).to(u_target.dtype)                         # [G]
    pred = net(noisy, s, node_feats, edge_index, edge_feats, tau, batch)

    node_se = ((pred - eps) ** 2).mean(dim=-1)                         # [ΣN]
    per_graph = _scatter_mean(node_se, batch, G)                       # [G]
    if graph_weights is not None:
        w = graph_weights.to(per_graph)
        return (w * per_graph).mean()
    return per_graph.mean()


@torch.no_grad()
def sample_ddpm_union(net, node_feats, edge_index, edge_feats, tau, batch,
                      schedule, steps=50, eta=1.0, sigma_init=1.0):
    """Reverse-diffusion sampler over a union graph (one update per node)."""
    T = schedule.T
    N = node_feats.shape[0]
    device, dtype = node_feats.device, node_feats.dtype
    u = torch.randn(N, net.point_dim, device=device, dtype=dtype) * sigma_init

    t_full = torch.round(torch.linspace(T - 1, 0, steps + 1, device=device)).long().clamp(0, T - 1)
    for i in range(steps):
        t = t_full[i].item()
        t_prev = t_full[i + 1].item()
        s = torch.full((tau.shape[0],), t / T, dtype=dtype, device=device)   # [G]
        eps_pred = net(u, s, node_feats, edge_index, edge_feats, tau, batch)

        sqrt_ab_t = schedule.sqrt_alphas_bar[t].to(dtype)
        sqrt_1mab_t = schedule.sqrt_one_minus_alphas_bar[t].to(dtype)
        ab_prev = schedule.alphas_bar[t_prev].to(dtype)
        ab_t = schedule.alphas_bar[t].to(dtype)

        u0_hat = (u - sqrt_1mab_t * eps_pred) / sqrt_ab_t.clamp_min(1e-8)
        pv = (1 - ab_prev) / (1 - ab_t).clamp_min(1e-8) * (1 - ab_t / ab_prev.clamp_min(1e-8))
        sigma_t = eta * pv.clamp_min(0.0).sqrt()
        dir_coeff = (1 - ab_prev - sigma_t ** 2).clamp_min(0.0).sqrt()
        z = torch.randn_like(u) if t_prev > 0 else torch.zeros_like(u)
        u = ab_prev.sqrt() * u0_hat + dir_coeff * eps_pred + sigma_t * z
    return u
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer_model.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_model.py tests/test_transfer_model.py
git commit -m "feat: union-graph propagator net, DDPM loss, and sampler"
```

---

### Task 5: Corpus-level update normalization

**Files:**
- Create: `lsmd/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Produces:
  - `UpdateNorm(scale: Tensor[point_dim])` with `.normalize(u)`, `.denormalize(u)`, `.state_dict()`.
  - `UpdateNorm.fit(updates: Tensor[M, point_dim]) -> UpdateNorm` — per-component std, floored at 1e-6.
  - `UpdateNorm.from_state_dict(d) -> UpdateNorm`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalize.py
import torch
from lsmd.normalize import UpdateNorm


def test_fit_and_roundtrip():
    torch.manual_seed(0)
    u = torch.randn(1000, 6) * torch.tensor([2.0, 2.0, 2.0, 0.3, 0.3, 0.3])
    norm = UpdateNorm.fit(u)
    # normalized columns have ~unit std
    z = norm.normalize(u)
    assert torch.allclose(z.std(0), torch.ones(6), atol=0.1)
    # round-trip is exact
    assert torch.allclose(norm.denormalize(norm.normalize(u)), u, atol=1e-5)


def test_state_dict_roundtrip():
    u = torch.randn(50, 6) + 1.0
    norm = UpdateNorm.fit(u)
    norm2 = UpdateNorm.from_state_dict(norm.state_dict())
    assert torch.allclose(norm.scale, norm2.scale)


def test_scale_is_floored():
    u = torch.zeros(10, 6)            # zero variance
    norm = UpdateNorm.fit(u)
    assert (norm.scale >= 1e-6).all()
    assert torch.isfinite(norm.normalize(u)).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsmd.normalize'`

- [ ] **Step 3: Write minimal implementation**

```python
# lsmd/normalize.py
"""Corpus-level normalization of per-residue SE(3) updates.

Update units (Angstrom for translation, radians for rotation) are already
consistent across proteins, so a single global per-component scale suffices to
put the DDPM target near unit variance.
"""
import torch


class UpdateNorm:
    def __init__(self, scale):
        self.scale = scale                       # [point_dim]

    @classmethod
    def fit(cls, updates):
        """Fit per-component scale from a sample of updates [M, point_dim]."""
        scale = updates.reshape(-1, updates.shape[-1]).std(dim=0).clamp_min(1e-6)
        return cls(scale)

    def normalize(self, u):
        return u / self.scale.to(u)

    def denormalize(self, u):
        return u * self.scale.to(u)

    def state_dict(self):
        return {"scale": self.scale.clone()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["scale"].clone())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalize.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/normalize.py tests/test_normalize.py
git commit -m "feat: corpus-level SE(3) update normalization"
```

---

### Task 6: Physical-τ pairs + state-conditional training-example builder

**Files:**
- Modify: `lsmd/data.py` (append two functions)
- Test: `tests/test_propagator_pairs.py`

**Interfaces:**
- Consumes: `lsmd.featurize.frame_graph`, `lsmd.featurize.frame_node_features`, `lsmd.featurize.relative_update`.
- Produces:
  - `physical_lag_pairs(num_frames: int, dt: float, lags_ps: list[float]) -> LongTensor[P, 3]` — columns `(start_frame, end_frame, tau_frames)`; lags whose `tau_frames >= num_frames` are skipped.
  - `build_training_example(frames: dict, i: int, tau_frames: int, k: int) -> dict` — uses frame `i` (current) to build the graph and `relative_update(R_i,t_i,R_j,t_j)` as the target; returns the dict shape consumed by `union_collate` (keys `node_feats`, `edge_index`, `edge_feats`, `u_target`, `tau`). `frames` must provide `R[F,N,3,3]`, `t[F,N,3]`, `res_type`, `chain_id`, `res_index`, and `dt` (ps/frame); `tau` in the output is `tau_frames * dt`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_propagator_pairs.py
import torch
from lsmd import data
from lsmd import geometry as g


def _synthetic_frames(F=20, N=10, dt=200.0):
    R = g.so3_exp(torch.randn(F, N, 3) * 0.1)
    t = torch.randn(F, N, 3) * 5.0
    return {
        "R": R, "t": t,
        "res_type": torch.randint(0, 21, (N,)),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "dt": dt,
    }


def test_physical_lag_pairs_converts_ps_to_frames():
    # dt=200 ps/frame; lags 200 and 1000 ps -> 1 and 5 frames
    pairs = data.physical_lag_pairs(num_frames=20, dt=200.0, lags_ps=[200.0, 1000.0])
    taus = sorted(set(pairs[:, 2].tolist()))
    assert taus == [1, 5]
    # end = start + tau_frames, all within range
    assert (pairs[:, 1] == pairs[:, 0] + pairs[:, 2]).all()
    assert pairs[:, 1].max().item() <= 19


def test_physical_lag_pairs_skips_too_large():
    pairs = data.physical_lag_pairs(num_frames=4, dt=200.0, lags_ps=[2000.0])  # 10 frames
    assert pairs.shape[0] == 0


def test_build_training_example_shapes_and_tau():
    fr = _synthetic_frames(F=20, N=10, dt=200.0)
    ex = data.build_training_example(fr, i=0, tau_frames=5, k=4)
    assert ex["node_feats"].shape == (10, 24)
    assert ex["u_target"].shape == (10, 6)
    assert ex["edge_feats"].shape == (10 * 4, 13)
    assert ex["edge_index"].shape == (2, 10 * 4)
    assert ex["tau"] == 5 * 200.0            # physical ps


def test_build_training_example_zero_update_for_identical_frames():
    fr = _synthetic_frames(F=20, N=10)
    fr["t"][7] = fr["t"][0]                    # frame 7 identical to frame 0
    fr["R"][7] = fr["R"][0]
    ex = data.build_training_example(fr, i=0, tau_frames=7, k=4)
    assert ex["u_target"].abs().max().item() < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagator_pairs.py -v`
Expected: FAIL with `AttributeError: module 'lsmd.data' has no attribute 'physical_lag_pairs'`

- [ ] **Step 3: Write minimal implementation**

Append to `lsmd/data.py`:

```python
from lsmd import featurize as _feat


def physical_lag_pairs(num_frames, dt, lags_ps):
    """Frame pairs at physical lag times (picoseconds).

    Args:
        num_frames: frames in the trajectory.
        dt:         ps per frame.
        lags_ps:    iterable of physical lags in ps.

    Returns:
        LongTensor [P, 3] — columns (start_frame, end_frame, tau_frames).
        Lags requiring >= num_frames frames are skipped.
    """
    segs = []
    for lag in lags_ps:
        tau_frames = max(1, int(round(float(lag) / dt)))
        if tau_frames >= num_frames:
            continue
        starts = torch.arange(0, num_frames - tau_frames, dtype=torch.long)
        tau_col = torch.full((len(starts),), tau_frames, dtype=torch.long)
        segs.append(torch.stack([starts, starts + tau_frames, tau_col], dim=1))
    if not segs:
        return torch.zeros((0, 3), dtype=torch.long)
    return torch.cat(segs, dim=0)


def build_training_example(frames, i, tau_frames, k):
    """State-conditional training example from frame i to frame i+tau_frames.

    The graph is built from the CURRENT frame i; the target is the per-residue
    SE(3) update from frame i to frame i+tau_frames.

    Args:
        frames: dict with R [F,N,3,3], t [F,N,3], res_type [N], chain_id [N],
                res_index [N], dt (ps/frame).
        i:          source frame index.
        tau_frames: lag in frames.
        k:          kNN neighbours.

    Returns:
        dict with node_feats [N,24], edge_index [2,E], edge_feats [E,13],
        u_target [N,6], tau (float, ps) — consumable by union_collate.
    """
    j = i + tau_frames
    R_i, t_i = frames["R"][i], frames["t"][i]
    R_j, t_j = frames["R"][j], frames["t"][j]
    edge_index, edge_feats = _feat.frame_graph(R_i, t_i, k)
    node_feats = _feat.frame_node_features(
        frames["res_type"], frames["chain_id"], frames["res_index"])
    u_target = _feat.relative_update(R_i, t_i, R_j, t_j)
    return {
        "node_feats": node_feats,
        "edge_index": edge_index,
        "edge_feats": edge_feats,
        "u_target": u_target,
        "tau": float(tau_frames) * float(frames["dt"]),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propagator_pairs.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/data.py tests/test_propagator_pairs.py
git commit -m "feat: physical-tau pairs and state-conditional example builder"
```

---

### Task 7: End-to-end single-step propagator integration

**Files:**
- Create: `tests/test_propagator_integration.py`

**Interfaces:**
- Consumes: everything above — `build_training_example`, `union_collate`, `PropagatorNet`, `sample_ddpm_union`, `apply_update`, `UpdateNorm`.

This task adds no new library code; it proves the pieces compose into one valid propagator step (build graph from current frames → sample update → `apply_update` → finite, correctly-shaped next frames). It uses synthetic frames so it runs without the WT trajectory; if `WT/WT-sol6.trr` is present it also runs one real step.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_propagator_integration.py
import os
import torch
from lsmd import data, batching, geometry as g
from lsmd import featurize as f
from lsmd.transfer_model import PropagatorNet, sample_ddpm_union
from lsmd.model import NoiseSchedule
from lsmd.normalize import UpdateNorm


def _synthetic_frames(F=10, N=12, dt=200.0):
    R = g.so3_exp(torch.randn(F, N, 3) * 0.1)
    t = torch.randn(F, N, 3) * 5.0
    return {"R": R, "t": t,
            "res_type": torch.randint(0, 21, (N,)),
            "chain_id": torch.zeros(N, dtype=torch.long),
            "res_index": torch.arange(N), "dt": dt}


def test_single_propagator_step_produces_valid_frames():
    torch.manual_seed(0)
    fr = _synthetic_frames(N=12)
    ex = data.build_training_example(fr, i=0, tau_frames=2, k=4)

    # fit normalization from a few example updates, normalize is identity-safe here
    norm = UpdateNorm.fit(ex["u_target"])
    u = batching.union_collate([ex])
    net = PropagatorNet(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)

    u_sample = sample_ddpm_union(net, u["node_feats"], u["edge_index"],
                                 u["edge_feats"], u["tau"], u["batch"],
                                 sched, steps=5)
    u_sample = norm.denormalize(u_sample)          # [12, 6]
    assert u_sample.shape == (12, 6)
    assert torch.isfinite(u_sample).all()

    # apply update to current frames → next frames are valid rotations + finite
    R_next, t_next = f.apply_update(fr["R"][0], fr["t"][0], u_sample)
    assert R_next.shape == (12, 3, 3) and t_next.shape == (12, 3)
    assert torch.isfinite(R_next).all() and torch.isfinite(t_next).all()
    # rotations stay orthonormal (R R^T = I)
    eye = torch.eye(3).expand(12, 3, 3)
    assert torch.allclose(R_next @ R_next.transpose(-1, -2), eye, atol=1e-3)


def test_real_wt_step_if_available():
    trr, gro = "WT/WT-sol6.trr", "WT/WT-sol6.gro"
    if not (os.path.exists(trr) and os.path.exists(gro)):
        import pytest
        pytest.skip("WT trajectory not present")
    fd = data.load_frames(trr, gro)              # provides R [F,N,3,3], t [F,N,3]
    from lsmd import vocab
    # re-key residue types onto the fixed vocab via residue names is done in the
    # ATLAS pipeline (Plan 2); here we just confirm frames feed the propagator.
    fr = {"R": fd["R"], "t": fd["t"],
          "res_type": fd["res_type"].clamp(max=vocab.N_AA_TYPES - 1),
          "chain_id": fd["chain_id"], "res_index": fd["res_index"], "dt": 200.0}
    ex = data.build_training_example(fr, i=0, tau_frames=2, k=8)
    u = batching.union_collate([ex])
    net = PropagatorNet(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)
    u_sample = sample_ddpm_union(net, u["node_feats"], u["edge_index"],
                                 u["edge_feats"], u["tau"], u["batch"],
                                 sched, steps=5)
    R_next, t_next = f.apply_update(fr["R"][0], fr["t"][0], u_sample)
    assert torch.isfinite(R_next).all() and torch.isfinite(t_next).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagator_integration.py -v`
Expected: FAIL initially only if any upstream task is incomplete; once Tasks 1–6 are committed this should pass. Run it first to confirm the wiring; if it errors with an import/attribute error, the relevant upstream task is missing.

- [ ] **Step 3: (no new implementation)**

All library code exists from Tasks 1–6. If the test fails, fix the specific upstream module it points to rather than adding code here.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propagator_integration.py -v`
Expected: PASS (1 passed, 1 skipped if WT absent; 2 passed if present)

- [ ] **Step 5: Run the full suite and commit**

Run: `pytest tests/ -q`
Expected: all pass (existing single-protein tests unaffected — Global Constraint: non-destructive).

```bash
git add tests/test_propagator_integration.py
git commit -m "test: end-to-end single-step transferable propagator integration"
```

---

## Self-Review

**Spec coverage:**
- Fixed AA vocabulary → Task 1. ✓
- SE(3) frame representation / `point_dim=6` / invariant `edge_dim=13` → Tasks 2, 6. ✓
- State-conditional dynamic graph (graph from current frame) → Tasks 2, 6. ✓
- 24-dim structure+AA node features, no ESM → Task 2. ✓
- Disjoint-union cross-protein batching → Tasks 3, 4. ✓
- Union-graph network + DDPM loss + sampler → Task 4. ✓
- Physical-time τ in ps → Task 6. ✓
- Corpus-level update normalization → Task 5. ✓
- Non-destructive (existing pipeline untouched) → new modules only; verified by full suite in Task 7. ✓
- Out of scope here (Plan 2): ATLAS pipeline, by-protein split, cross-protein trainer, zero-shot eval. Tracked as the follow-up plan.

**Type consistency:** `point_dim=6`, `edge_dim=13`, `node_dim=24`, `N_AA_TYPES=21`, `UNK_INDEX=20` are used identically across Tasks 1–7. `union_collate` output keys (`node_feats`, `edge_index`, `edge_feats`, `u_target`, `batch`, `tau`) match `build_training_example` output keys and `PropagatorNet.forward` / `ddpm_loss_union` / `sample_ddpm_union` signatures. `tau` is float ps everywhere.

**Placeholder scan:** No TBD/TODO; every code step is complete and runnable.

---

## Follow-up: Plan 2 (data + training + eval)

Once this core is green, the second plan covers spec build-steps 4–6:
1. **ATLAS pipeline** — downloader/preprocessor producing per-protein `.pt` shards with fixed-vocab `res_type` (via `lsmd.vocab.residue_indices` on residue names), per-residue frames, sequence, and `dt`; homology-aware by-protein train/val/test split.
2. **Cross-protein trainer** — protein sampler, `union_collate` minibatches, gradient accumulation, multi-τ schedule (physical ps), `UpdateNorm.fit` over a corpus sample, checkpointing (model + noise schedule + `UpdateNorm` + vocab size).
3. **Zero-shot inference + evaluation harness** — rollout from a held-out protein's reference structure (dynamic graph each step, `sample_ddpm_union` with `G=1`, `apply_update`); RMSF-profile correlation + Cα distance JS + SS retention vs reference MD; oracle / lower-baseline bracketing.
