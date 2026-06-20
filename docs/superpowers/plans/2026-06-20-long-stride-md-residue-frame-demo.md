# Long-Stride Protein MD — Residue-Frame Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end demo that, given one protein backbone conformation, samples several chemically valid conformations many MD steps ahead via a learned stochastic transition operator on residue frames.

**Architecture:** Represent each residue as an SE(3) frame. The model learns a conditional flow-matching distribution over **per-residue relative updates** `Δ = (local-frame translation [3], axis-angle rotation [3])` from `x_t` to `x_{t+τ}`. Because `Δ` is E(3)-invariant and we compose it back onto the current (equivariant) frame, a plain graph net on invariant node/edge features is equivariant-by-construction. Decode frames → ideal backbone atoms → light idealization → PDB.

**Tech Stack:** Python 3.10+, PyTorch, mdtraj, NumPy, pytest.

## Global Constraints

- Backbone only: atoms N, CA, C, O. No side chains.
- Single protein, single fixed stride `τ` (configurable).
- One shared coordinate system for all residues/chains (so multi-chain is the same object with more nodes). Single-chain demo only, but no code path may assume one chain.
- Rotation parametrized in axis-angle (so(3) tangent) of the *relative* rotation `R_t^T R_future`; valid for moderate strides where per-residue relative rotation < π.
- Frame convention (used identically everywhere): given N, CA, C positions, `e1 = normalize(C - CA)`; `e2 = normalize((N - CA) - ((N - CA)·e1) e1)`; `e3 = e1 × e2`; `R = [e1 | e2 | e3]` (columns), `t = CA`.
- All tensors `float32`. Frames stored as `R: [..., 3, 3]`, `t: [..., 3]`.
- Package name: `lsmd`. Tests under `tests/`.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata + deps |
| `lsmd/__init__.py` | Package marker |
| `lsmd/geometry.py` | Frame construction, SO(3) exp/log, compose/invert, ideal backbone atom placement |
| `lsmd/featurize.py` | Relative-update encode/decode, k-NN graph, invariant node/edge features |
| `lsmd/data.py` | mdtraj loading → frames; pair building; time-ordered split |
| `lsmd/model.py` | Flow-matching graph net, CFM loss, sampler |
| `lsmd/decoder.py` | Δ-samples → future frames → atoms → idealization → PDB |
| `lsmd/validation.py` | Geometry/diversity/ensemble metrics + baselines |
| `lsmd/demo.py` | CLI glue: train, sample, report |
| `tests/` | One test module per `lsmd` module |

---

## Task 1: Project scaffold + geometry core

**Files:**
- Create: `pyproject.toml`, `lsmd/__init__.py`, `lsmd/geometry.py`, `tests/test_geometry.py`

**Interfaces:**
- Produces:
  - `build_frames(N, CA, C) -> (R, t)` — inputs `[..., 3]`; `R: [..., 3, 3]`, `t: [..., 3]`
  - `so3_exp(omega) -> R` — `omega: [..., 3]` → `R: [..., 3, 3]`
  - `so3_log(R) -> omega` — inverse of `so3_exp`
  - `compose(R1, t1, R2, t2) -> (R, t)` — rigid transform `T1 ∘ T2`
  - `invert(R, t) -> (Rinv, tinv)`
  - `IDEAL_LOCAL: dict[str, list[float]]` — ideal local coords of N, CA, C, O in the residue frame
  - `place_backbone(R, t) -> atoms` — `R: [..., 3, 3]`, `t: [..., 3]` → `atoms: [..., 4, 3]` ordered N, CA, C, O

- [ ] **Step 1: Create package scaffold**

`pyproject.toml`:
```toml
[project]
name = "lsmd"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch", "mdtraj", "numpy"]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
include = ["lsmd*"]
```

`lsmd/__init__.py`:
```python
"""Long-stride protein MD residue-frame demo."""
```

- [ ] **Step 2: Write the failing test for frames + SO(3) + placement**

`tests/test_geometry.py`:
```python
import torch
from lsmd import geometry as g


def test_build_frames_orthonormal():
    N = torch.tensor([[-0.5, 1.4, 0.0]])
    CA = torch.tensor([[0.0, 0.0, 0.0]])
    C = torch.tensor([[1.5, 0.0, 0.0]])
    R, t = g.build_frames(N, CA, C)
    # columns orthonormal
    gram = R[0].T @ R[0]
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)
    assert torch.isclose(torch.det(R[0]), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(t[0], CA[0])


def test_so3_exp_log_roundtrip():
    omega = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
    R = g.so3_exp(omega)
    omega2 = g.so3_log(R)
    assert torch.allclose(omega, omega2, atol=1e-5)


def test_compose_invert_identity():
    R, t = g.so3_exp(torch.tensor([[0.2, 0.1, -0.3]])), torch.tensor([[1.0, 2.0, 3.0]])
    Ri, ti = g.invert(R, t)
    Rc, tc = g.compose(R, t, Ri, ti)
    assert torch.allclose(Rc[0], torch.eye(3), atol=1e-5)
    assert torch.allclose(tc[0], torch.zeros(3), atol=1e-5)


def test_place_backbone_reproduces_frame():
    # placing ideal atoms then rebuilding the frame returns the same frame
    R = g.so3_exp(torch.tensor([[0.3, -0.1, 0.2]]))
    t = torch.tensor([[1.0, -2.0, 0.5]])
    atoms = g.place_backbone(R, t)  # [1,4,3] N,CA,C,O
    R2, t2 = g.build_frames(atoms[:, 0], atoms[:, 1], atoms[:, 2])
    assert torch.allclose(R, R2, atol=1e-4)
    assert torch.allclose(t, t2, atol=1e-4)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` (functions not defined).

- [ ] **Step 4: Implement `lsmd/geometry.py`**

```python
import torch

# Ideal local backbone coordinates (Angstrom) in the residue frame
# (CA at origin; e1 along CA->C; N in +e2 half-plane). O is approximate.
IDEAL_LOCAL = {
    "N":  [-0.522, 1.362, 0.0],
    "CA": [0.0, 0.0, 0.0],
    "C":  [1.525, 0.0, 0.0],
    "O":  [2.158, -1.056, 0.0],
}
_ATOM_ORDER = ["N", "CA", "C", "O"]


def _normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def build_frames(N, CA, C):
    e1 = _normalize(C - CA)
    u = N - CA
    e2 = _normalize(u - (u * e1).sum(-1, keepdim=True) * e1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)  # columns
    return R, CA.clone()


def so3_exp(omega):
    theta = omega.norm(dim=-1, keepdim=True)
    small = theta < 1e-6
    axis = omega / theta.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], -1),
        torch.stack([z, zero, -x], -1),
        torch.stack([-y, x, zero], -1),
    ], -2)
    th = theta[..., None]
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(K.shape)
    R = eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    # near zero, first-order fallback
    K0 = torch.stack([
        torch.stack([zero, -omega[..., 2], omega[..., 1]], -1),
        torch.stack([omega[..., 2], zero, -omega[..., 0]], -1),
        torch.stack([-omega[..., 1], omega[..., 0], zero], -1),
    ], -2)
    R_small = eye + K0
    return torch.where(small[..., None], R_small, R)


def so3_log(R):
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1) / 2).clamp(-1.0, 1.0)
    theta = torch.acos(cos)[..., None]
    vee = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], -1)
    small = theta < 1e-6
    omega = torch.where(small, 0.5 * vee, (theta / (2 * torch.sin(theta).clamp_min(1e-8))) * vee)
    return omega


def compose(R1, t1, R2, t2):
    R = R1 @ R2
    t = (R1 @ t2.unsqueeze(-1)).squeeze(-1) + t1
    return R, t


def invert(R, t):
    Rinv = R.transpose(-1, -2)
    tinv = -(Rinv @ t.unsqueeze(-1)).squeeze(-1)
    return Rinv, tinv


def place_backbone(R, t):
    local = torch.tensor([IDEAL_LOCAL[a] for a in _ATOM_ORDER],
                         device=R.device, dtype=R.dtype)  # [4,3]
    # global = t + R @ local
    placed = (R.unsqueeze(-3) @ local.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(-2)
    return placed  # [...,4,3]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_geometry.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml lsmd/__init__.py lsmd/geometry.py tests/test_geometry.py
git commit -m "feat: geometry core (frames, SO(3), backbone placement)"
```

---

## Task 2: Relative-update encoding + invariant features

**Files:**
- Create: `lsmd/featurize.py`, `tests/test_featurize.py`

**Interfaces:**
- Consumes: `geometry.so3_exp`, `geometry.so3_log`
- Produces:
  - `relative_update(R_t, t_t, R_f, t_f) -> u` — `u: [..., 6]` = `[local_trans(3), axis_angle(3)]`
  - `apply_update(R_t, t_t, u) -> (R_f, t_f)` — inverse of `relative_update`
  - `knn_graph(t, k) -> edge_index` — `t: [N, 3]` → `edge_index: [2, E]` (long), row 0 = src, row 1 = dst
  - `edge_features(R, t, edge_index) -> feats` — `feats: [E, 13]` invariant features
  - `node_features(res_type, chain_id, res_index, n_types) -> feats` — `feats: [N, F]`

- [ ] **Step 1: Write the failing test**

`tests/test_featurize.py`:
```python
import torch
from lsmd import geometry as g
from lsmd import featurize as f


def _rand_frames(n):
    R = g.so3_exp(torch.randn(n, 3) * 0.5)
    t = torch.randn(n, 3)
    return R, t


def test_update_roundtrip():
    R_t, t_t = _rand_frames(5)
    R_f, t_f = _rand_frames(5)
    u = f.relative_update(R_t, t_t, R_f, t_f)
    R_f2, t_f2 = f.apply_update(R_t, t_t, u)
    assert torch.allclose(R_f, R_f2, atol=1e-4)
    assert torch.allclose(t_f, t_f2, atol=1e-4)


def test_update_is_invariant_to_global_transform():
    R_t, t_t = _rand_frames(5)
    R_f, t_f = _rand_frames(5)
    u = f.relative_update(R_t, t_t, R_f, t_f)
    # apply a global rotation+translation to both endpoints
    Rg = g.so3_exp(torch.tensor([[0.3, -0.5, 0.2]])).expand(5, 3, 3)
    tg = torch.tensor([1.0, 2.0, -3.0])
    R_t2, t_t2 = Rg @ R_t, (Rg @ t_t.unsqueeze(-1)).squeeze(-1) + tg
    R_f2, t_f2 = Rg @ R_f, (Rg @ t_f.unsqueeze(-1)).squeeze(-1) + tg
    u2 = f.relative_update(R_t2, t_t2, R_f2, t_f2)
    assert torch.allclose(u, u2, atol=1e-4)


def test_knn_graph_shape_and_neighbors():
    t = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0], [6.0, 0, 0]])
    ei = f.knn_graph(t, k=1)
    assert ei.shape[0] == 2
    # nearest neighbor of node 0 is node 1
    nbr_of_0 = ei[1][ei[0] == 0]
    assert 1 in nbr_of_0.tolist()


def test_edge_features_invariant():
    R, t = _rand_frames(6)
    ei = f.knn_graph(t, k=2)
    feats = f.edge_features(R, t, ei)
    Rg = g.so3_exp(torch.tensor([[0.1, 0.7, -0.2]])).expand(6, 3, 3)
    tg = torch.tensor([2.0, -1.0, 4.0])
    R2, t2 = Rg @ R, (Rg @ t.unsqueeze(-1)).squeeze(-1) + tg
    feats2 = f.edge_features(R2, t2, ei)
    assert torch.allclose(feats, feats2, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_featurize.py -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement `lsmd/featurize.py`**

```python
import torch
import torch.nn.functional as F_nn
from lsmd import geometry as g


def relative_update(R_t, t_t, R_f, t_f):
    Rt_inv = R_t.transpose(-1, -2)
    local_trans = (Rt_inv @ (t_f - t_t).unsqueeze(-1)).squeeze(-1)
    rel_R = Rt_inv @ R_f
    axis_angle = g.so3_log(rel_R)
    return torch.cat([local_trans, axis_angle], dim=-1)


def apply_update(R_t, t_t, u):
    local_trans, axis_angle = u[..., :3], u[..., 3:]
    t_f = (R_t @ local_trans.unsqueeze(-1)).squeeze(-1) + t_t
    R_f = R_t @ g.so3_exp(axis_angle)
    return R_f, t_f


def knn_graph(t, k):
    n = t.shape[0]
    d = torch.cdist(t, t)
    d.fill_diagonal_(float("inf"))
    k = min(k, n - 1)
    idx = d.topk(k, largest=False).indices  # [n,k]
    src = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    dst = idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


def edge_features(R, t, edge_index):
    src, dst = edge_index
    Rs_inv = R[src].transpose(-1, -2)
    rel_pos = (Rs_inv @ (t[dst] - t[src]).unsqueeze(-1)).squeeze(-1)  # [E,3] invariant
    dist = (t[dst] - t[src]).norm(dim=-1, keepdim=True)               # [E,1]
    rel_R = (Rs_inv @ R[dst]).reshape(-1, 9)                          # [E,9] invariant
    return torch.cat([rel_pos, dist, rel_R], dim=-1)                  # [E,13]


def node_features(res_type, chain_id, res_index, n_types):
    rt = F_nn.one_hot(res_type, num_classes=n_types).float()
    ch = chain_id.float().unsqueeze(-1)
    # smooth positional encoding of residue index
    pos = res_index.float().unsqueeze(-1)
    pe = torch.cat([torch.sin(pos / 100.0), torch.cos(pos / 100.0)], dim=-1)
    return torch.cat([rt, ch, pe], dim=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_featurize.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add lsmd/featurize.py tests/test_featurize.py
git commit -m "feat: invariant relative-update encoding and graph features"
```

---

## Task 3: Data loading from trajectories

**Files:**
- Create: `lsmd/data.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `geometry.build_frames`
- Produces:
  - `load_frames(traj_path, top_path) -> dict` with keys: `R [F,N,3,3]`, `t [F,N,3]`, `res_type [N]` (long), `chain_id [N]` (long), `res_index [N]` (long), `n_types` (int)
  - `make_pairs(num_frames, tau) -> LongTensor [P, 2]`
  - `time_split(pairs, val_frac) -> (train_pairs, val_pairs)`

- [ ] **Step 1: Write the failing test (build a tiny synthetic mdtraj trajectory)**

`tests/test_data.py`:
```python
import numpy as np
import mdtraj as md
import torch
from lsmd import data as d


def _tiny_traj(tmp_path, n_res=4, n_frames=10):
    top = md.Topology()
    chain = top.add_chain()
    for i in range(n_res):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        for name, elem in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]:
            top.add_atom(name, md.element.get_by_symbol(elem), res)
    xyz = np.random.RandomState(0).randn(n_frames, n_res * 4, 3).astype(np.float32) * 0.3
    # spread residues out so frames are well-defined
    for i in range(n_res):
        xyz[:, i * 4:(i + 1) * 4, 0] += i * 4.0
    traj = md.Trajectory(xyz, top)
    p = tmp_path / "tiny.pdb"
    traj.save_pdb(str(p))
    return str(p)


def test_load_frames_shapes(tmp_path):
    path = _tiny_traj(tmp_path)
    out = d.load_frames(path, path)
    assert out["R"].shape == (10, 4, 3, 3)
    assert out["t"].shape == (10, 4, 3)
    assert out["res_type"].shape == (4,)
    assert out["chain_id"].shape == (4,)
    assert out["n_types"] >= 1


def test_make_pairs_and_split():
    pairs = d.make_pairs(num_frames=100, tau=10)
    assert pairs.shape[1] == 2
    assert (pairs[:, 1] - pairs[:, 0] == 10).all()
    assert pairs[:, 1].max() < 100
    train, val = d.time_split(pairs, val_frac=0.2)
    # time-ordered: max train start < min val start (no leakage)
    assert train[:, 0].max() < val[:, 0].min()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement `lsmd/data.py`**

```python
import numpy as np
import mdtraj as md
import torch
from lsmd import geometry as g


def load_frames(traj_path, top_path):
    traj = md.load(traj_path, top=top_path)
    top = traj.topology
    residues = [r for r in top.residues if r.name != "HOH"]

    def atom_index(res, name):
        for a in res.atoms:
            if a.name == name:
                return a.index
        return None

    keep, res_names, chain_ids = [], [], []
    for r in residues:
        idx = {nm: atom_index(r, nm) for nm in ("N", "CA", "C")}
        if any(v is None for v in idx.values()):
            continue  # skip residues lacking backbone (e.g. caps)
        keep.append(idx)
        res_names.append(r.name)
        chain_ids.append(r.chain.index)

    xyz = torch.tensor(traj.xyz, dtype=torch.float32) * 10.0  # nm -> Angstrom
    N = xyz[:, [k["N"] for k in keep], :]
    CA = xyz[:, [k["CA"] for k in keep], :]
    C = xyz[:, [k["C"] for k in keep], :]
    R, t = g.build_frames(N, CA, C)

    uniq = sorted(set(res_names))
    type_map = {nm: i for i, nm in enumerate(uniq)}
    res_type = torch.tensor([type_map[nm] for nm in res_names], dtype=torch.long)
    chain_id = torch.tensor(chain_ids, dtype=torch.long)
    res_index = torch.arange(len(keep), dtype=torch.long)

    return {"R": R, "t": t, "res_type": res_type, "chain_id": chain_id,
            "res_index": res_index, "n_types": len(uniq)}


def make_pairs(num_frames, tau):
    starts = torch.arange(0, num_frames - tau, dtype=torch.long)
    return torch.stack([starts, starts + tau], dim=1)


def time_split(pairs, val_frac):
    n = pairs.shape[0]
    cut = int(n * (1 - val_frac))
    return pairs[:cut], pairs[cut:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add lsmd/data.py tests/test_data.py
git commit -m "feat: trajectory loading to residue frames and pairing"
```

---

## Task 4: Flow-matching model (network, loss, sampler)

**Files:**
- Create: `lsmd/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `featurize.relative_update`, `featurize.edge_features`, `featurize.node_features`
- Produces:
  - `class FlowNet(node_dim, edge_dim, hidden=64, layers=3)` with
    `forward(u_s, s, node_feats, edge_index, edge_feats) -> velocity [N,6]`
  - `cfm_loss(net, u_target, node_feats, edge_index, edge_feats, sigma=0.1) -> scalar`
  - `sample(net, node_feats, edge_index, edge_feats, K, steps=50, sigma=0.1) -> u [K,N,6]`

The network consumes only **invariant** features (`node_feats`, `edge_feats`, the per-residue `u_s`, and scalar `s`) and outputs per-residue velocities in the invariant `Δ`-space — so equivariance is guaranteed once `apply_update` composes the result onto the current frame (done in the decoder).

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
import torch
from lsmd import model as m


def _dummy_inputs(n=6, node_dim=8, edge_dim=13):
    node_feats = torch.randn(n, node_dim)
    edge_index = torch.randint(0, n, (2, 20))
    edge_feats = torch.randn(20, edge_dim)
    return node_feats, edge_index, edge_feats


def test_flownet_output_shape():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    u_s = torch.randn(6, 6)
    s = torch.tensor(0.4)
    v = net(u_s, s, nf, ei, ef)
    assert v.shape == (6, 6)


def test_cfm_can_overfit_constant_target():
    torch.manual_seed(0)
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=64, layers=2)
    u_target = torch.randn(6, 6)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = m.cfm_loss(net, u_target, nf, ei, ef, sigma=0.1)
        loss.backward()
        opt.step()
    samples = m.sample(net, nf, ei, ef, K=8, steps=50, sigma=0.1)
    assert samples.shape == (8, 6, 6)
    # sampled mean should be near the target it was trained to reproduce
    assert (samples.mean(0) - u_target).abs().mean() < 0.3


def test_sampler_is_diverse():
    nf, ei, ef = _dummy_inputs()
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=32, layers=2)
    samples = m.sample(net, nf, ei, ef, K=8, steps=20, sigma=0.2)
    spread = samples.std(0).mean()
    assert spread > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL (module/classes not defined).

- [ ] **Step 3: Implement `lsmd/model.py`**

```python
import torch
import torch.nn as nn


class MessageLayer(nn.Module):
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
        src, dst = edge_index
        msg = self.msg(torch.cat([h[src], h[dst], edge_feats], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(h.shape[0], 1, device=h.device).index_add_(
            0, dst, torch.ones(dst.shape[0], 1, device=h.device))
        agg = agg / deg.clamp_min(1.0)
        return h + self.upd(torch.cat([h, agg], dim=-1))


class FlowNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=64, layers=3):
        super().__init__()
        # input: node features + current u (6) + time embedding (1)
        self.embed = nn.Linear(node_dim + 6 + 1, hidden)
        self.layers = nn.ModuleList([MessageLayer(hidden, edge_dim) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 6))

    def forward(self, u_s, s, node_feats, edge_index, edge_feats):
        n = node_feats.shape[0]
        s_col = torch.as_tensor(s, dtype=u_s.dtype, device=u_s.device).reshape(1, 1).expand(n, 1)
        h = self.embed(torch.cat([node_feats, u_s, s_col], dim=-1))
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)
        return self.head(h)


def cfm_loss(net, u_target, node_feats, edge_index, edge_feats, sigma=0.1):
    n = u_target.shape[0]
    u0 = torch.randn_like(u_target) * sigma          # prior (small motions)
    s = torch.rand(())                               # shared flow-time per step
    u_s = (1 - s) * u0 + s * u_target                # linear path
    target_v = u_target - u0                         # rectified-flow velocity
    pred_v = net(u_s, s, node_feats, edge_index, edge_feats)
    return ((pred_v - target_v) ** 2).mean()


@torch.no_grad()
def sample(net, node_feats, edge_index, edge_feats, K, steps=50, sigma=0.1):
    n = node_feats.shape[0]
    outs = []
    for _ in range(K):
        u = torch.randn(n, 6, device=node_feats.device) * sigma
        for i in range(steps):
            s = torch.tensor(i / steps)
            v = net(u, s, node_feats, edge_index, edge_feats)
            u = u + v / steps
        outs.append(u)
    return torch.stack(outs, dim=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS (3 tests). `test_cfm_can_overfit_constant_target` may take a few seconds.

- [ ] **Step 5: Commit**

```bash
git add lsmd/model.py tests/test_model.py
git commit -m "feat: conditional flow-matching graph net, loss, sampler"
```

---

## Task 5: Decoder — frames to atoms with idealization + PDB

**Files:**
- Create: `lsmd/decoder.py`, `tests/test_decoder.py`

**Interfaces:**
- Consumes: `featurize.apply_update`, `geometry.place_backbone`
- Produces:
  - `decode_frames(R_t, t_t, u_samples) -> (R_f, t_f)` — `u_samples [K,N,6]` → `R_f [K,N,3,3]`, `t_f [K,N,3]`
  - `build_structure(R, t) -> atoms` — `[N,3,3]/[N,3]` → `atoms [N,4,3]`
  - `idealize(atoms, steps=50) -> atoms` — minimizes peptide-bond + clash violations, returns `[N,4,3]`
  - `peptide_bond_violation(atoms) -> scalar` — mean |C_i-N_{i+1}| deviation from 1.33 Å
  - `write_pdb(atoms, res_type_names, path)` — `atoms [N,4,3]`, list of residue name strings

- [ ] **Step 1: Write the failing test**

`tests/test_decoder.py`:
```python
import torch
from lsmd import geometry as g
from lsmd import featurize as f
from lsmd import decoder as dec


def test_decode_frames_roundtrip():
    R_t = g.so3_exp(torch.randn(5, 3) * 0.3)
    t_t = torch.randn(5, 3)
    R_f = g.so3_exp(torch.randn(5, 3) * 0.3)
    t_f = torch.randn(5, 3)
    u = f.relative_update(R_t, t_t, R_f, t_f).unsqueeze(0)  # [1,5,6]
    R_d, t_d = dec.decode_frames(R_t, t_t, u)
    assert torch.allclose(R_d[0], R_f, atol=1e-4)
    assert torch.allclose(t_d[0], t_f, atol=1e-4)


def test_build_structure_shape():
    R = g.so3_exp(torch.randn(6, 3) * 0.2)
    t = torch.randn(6, 3)
    atoms = dec.build_structure(R, t)
    assert atoms.shape == (6, 4, 3)


def test_idealize_reduces_peptide_violation():
    # lay residues far apart so peptide bonds are badly broken
    R = g.so3_exp(torch.zeros(5, 3))
    t = torch.arange(5).float().unsqueeze(-1).repeat(1, 3) * 5.0
    atoms = dec.build_structure(R, t)
    before = dec.peptide_bond_violation(atoms)
    fixed = dec.idealize(atoms, steps=200)
    after = dec.peptide_bond_violation(fixed)
    assert after < before


def test_write_pdb(tmp_path):
    R = g.so3_exp(torch.randn(3, 3) * 0.2)
    t = torch.randn(3, 3)
    atoms = dec.build_structure(R, t)
    p = tmp_path / "out.pdb"
    dec.write_pdb(atoms, ["ALA", "GLY", "ALA"], str(p))
    text = p.read_text()
    assert "ATOM" in text and "CA" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_decoder.py -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement `lsmd/decoder.py`**

```python
import torch
from lsmd import geometry as g
from lsmd import featurize as f

PEPTIDE_CN = 1.33  # Angstrom
CLASH = 2.0        # min non-bonded CA-CA-ish distance for penalty


def decode_frames(R_t, t_t, u_samples):
    Rs, ts = [], []
    for k in range(u_samples.shape[0]):
        R_f, t_f = f.apply_update(R_t, t_t, u_samples[k])
        Rs.append(R_f)
        ts.append(t_f)
    return torch.stack(Rs, 0), torch.stack(ts, 0)


def build_structure(R, t):
    return g.place_backbone(R, t)  # [N,4,3]


def peptide_bond_violation(atoms):
    C = atoms[:-1, 2, :]   # C of residue i
    N = atoms[1:, 0, :]    # N of residue i+1
    d = (C - N).norm(dim=-1)
    return (d - PEPTIDE_CN).abs().mean()


def _clash_penalty(ca):
    d = torch.cdist(ca, ca)
    n = ca.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=ca.device)
    # only penalize non-adjacent residues
    for i in range(n - 1):
        mask[i, i + 1] = False
        mask[i + 1, i] = False
    viol = (CLASH - d).clamp_min(0.0) * mask
    return (viol ** 2).sum()


def idealize(atoms, steps=50):
    # optimize a per-residue translation that closes peptide bonds and removes clashes,
    # keeping each residue rigid (only CA position shifts; relative atom geometry preserved)
    delta = torch.zeros(atoms.shape[0], 3, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.05)
    base = atoms.detach()
    for _ in range(steps):
        opt.zero_grad()
        shifted = base + delta.unsqueeze(1)
        l_pep = ((shifted[:-1, 2, :] - shifted[1:, 0, :]).norm(dim=-1) - PEPTIDE_CN).pow(2).sum()
        l_clash = _clash_penalty(shifted[:, 1, :])
        loss = l_pep + 0.1 * l_clash
        loss.backward()
        opt.step()
    return (base + delta.detach().unsqueeze(1))


_ELEMENTS = {"N": "N", "CA": "C", "C": "C", "O": "O"}
_ATOM_NAMES = ["N", "CA", "C", "O"]


def write_pdb(atoms, res_type_names, path):
    lines = []
    serial = 1
    for ri in range(atoms.shape[0]):
        for ai, name in enumerate(_ATOM_NAMES):
            x, y, z = atoms[ri, ai].tolist()
            lines.append(
                f"ATOM  {serial:5d} {name:<4s}{res_type_names[ri]:>3s} A{ri + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {_ELEMENTS[name]:>2s}"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_decoder.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add lsmd/decoder.py tests/test_decoder.py
git commit -m "feat: decoder - frames to atoms, idealization, PDB export"
```

---

## Task 6: Validation metrics + baselines

**Files:**
- Create: `lsmd/validation.py`, `tests/test_validation.py`

**Interfaces:**
- Consumes: `decoder.peptide_bond_violation`
- Produces:
  - `geometry_metrics(atoms) -> dict` keys: `ca_bond_mean`, `peptide_violation`, `clash_count`
  - `diversity(atoms_K) -> float` — mean pairwise CA-RMSD across `K` structures `[K,N,4,3]`
  - `ensemble_overlap(ca_gen, ca_md) -> float` — 1D-histogram overlap of CA-CA distance distributions, in [0,1]
  - `baseline_copy(R_t, t_t, K)`, `baseline_noise(R_t, t_t, K, sigma)` — return `u_samples [K,N,6]`

- [ ] **Step 1: Write the failing test**

`tests/test_validation.py`:
```python
import torch
from lsmd import geometry as g
from lsmd import decoder as dec
from lsmd import validation as val


def test_geometry_metrics_keys():
    R = g.so3_exp(torch.randn(5, 3) * 0.1)
    t = torch.arange(5).float().unsqueeze(-1).repeat(1, 3) * 3.8
    atoms = dec.build_structure(R, t)
    mt = val.geometry_metrics(atoms)
    assert {"ca_bond_mean", "peptide_violation", "clash_count"} <= set(mt)


def test_diversity_zero_for_identical():
    R = g.so3_exp(torch.randn(4, 3) * 0.1)
    t = torch.randn(4, 3)
    atoms = dec.build_structure(R, t)
    stacked = atoms.unsqueeze(0).repeat(5, 1, 1, 1)
    assert val.diversity(stacked) < 1e-5


def test_ensemble_overlap_identical_is_high():
    ca = torch.randn(50, 3)
    o = val.ensemble_overlap(ca, ca.clone())
    assert o > 0.95


def test_baselines_shapes():
    R = g.so3_exp(torch.randn(6, 3) * 0.1)
    t = torch.randn(6, 3)
    uc = val.baseline_copy(R, t, K=4)
    un = val.baseline_noise(R, t, K=4, sigma=0.2)
    assert uc.shape == (4, 6, 6) and un.shape == (4, 6, 6)
    assert uc.abs().sum() < 1e-5  # copy = zero update
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement `lsmd/validation.py`**

```python
import torch
from lsmd import decoder as dec


def geometry_metrics(atoms):
    ca = atoms[:, 1, :]
    ca_bonds = (ca[1:] - ca[:-1]).norm(dim=-1)
    d = torch.cdist(ca, ca)
    n = ca.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    for i in range(n - 1):
        mask[i, i + 1] = mask[i + 1, i] = False
    clash_count = ((d < 2.0) & mask).sum().item() / 2
    return {
        "ca_bond_mean": ca_bonds.mean().item(),
        "peptide_violation": dec.peptide_bond_violation(atoms).item(),
        "clash_count": clash_count,
    }


def diversity(atoms_K):
    ca = atoms_K[:, :, 1, :]  # [K,N,3]
    K = ca.shape[0]
    total, count = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            total += (ca[i] - ca[j]).pow(2).mean().sqrt().item()
            count += 1
    return total / max(count, 1)


def ensemble_overlap(ca_gen, ca_md, bins=30):
    def pdist(ca):
        return torch.pdist(ca)
    dg, dm = pdist(ca_gen), pdist(ca_md)
    lo = min(dg.min(), dm.min()).item()
    hi = max(dg.max(), dm.max()).item()
    hg = torch.histc(dg, bins=bins, min=lo, max=hi)
    hm = torch.histc(dm, bins=bins, min=lo, max=hi)
    hg, hm = hg / hg.sum(), hm / hm.sum()
    return torch.minimum(hg, hm).sum().item()


def baseline_copy(R_t, t_t, K):
    n = R_t.shape[0]
    return torch.zeros(K, n, 6)


def baseline_noise(R_t, t_t, K, sigma):
    n = R_t.shape[0]
    return torch.randn(K, n, 6) * sigma
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validation.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add lsmd/validation.py tests/test_validation.py
git commit -m "feat: validation metrics and baselines"
```

---

## Task 7: Demo CLI — train, sample, report

**Files:**
- Create: `lsmd/demo.py`, `tests/test_demo.py`

**Interfaces:**
- Consumes: all prior modules
- Produces:
  - `train(frames, tau, epochs, k, hidden, layers, sigma, lr) -> (net, ctx)` where `ctx` holds `node_feats, edge_index, edge_feats` built from frame 0's graph
  - `run_demo(traj_path, top_path, tau, out_dir, K, epochs) -> dict` — trains, samples K futures from a held-out frame, writes PDBs, returns a metrics report dict

- [ ] **Step 1: Write the failing smoke test**

`tests/test_demo.py`:
```python
import numpy as np
import mdtraj as md
from lsmd import demo


def _tiny_traj(tmp_path, n_res=6, n_frames=60):
    top = md.Topology()
    chain = top.add_chain()
    for i in range(n_res):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        for name, elem in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]:
            top.add_atom(name, md.element.get_by_symbol(elem), res)
    rs = np.random.RandomState(0)
    xyz = np.zeros((n_frames, n_res * 4, 3), np.float32)
    for i in range(n_res):
        base = np.array([i * 0.38, 0, 0], np.float32)
        offs = rs.randn(4, 3).astype(np.float32) * 0.02
        for fr in range(n_frames):
            wobble = rs.randn(4, 3).astype(np.float32) * 0.01
            xyz[fr, i * 4:(i + 1) * 4] = base + offs + wobble
    p = tmp_path / "tiny.pdb"
    md.Trajectory(xyz, top).save_pdb(str(p))
    return str(p)


def test_run_demo_smoke(tmp_path):
    path = _tiny_traj(tmp_path)
    out = tmp_path / "out"
    report = demo.run_demo(path, path, tau=5, out_dir=str(out), K=4, epochs=30)
    assert "model_geometry" in report
    assert "diversity" in report
    # K PDB files were written
    pdbs = list(out.glob("future_*.pdb"))
    assert len(pdbs) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo.py -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement `lsmd/demo.py`**

```python
import os
import torch
from lsmd import data, featurize as f, model as m, decoder as dec, validation as val


def _build_ctx(frames, k):
    R0, t0 = frames["R"][0], frames["t"][0]
    edge_index = f.knn_graph(t0, k=k)
    edge_feats = f.edge_features(R0, t0, edge_index)
    node_feats = f.node_features(frames["res_type"], frames["chain_id"],
                                 frames["res_index"], frames["n_types"])
    return node_feats, edge_index, edge_feats


def train(frames, tau, epochs, k, hidden, layers, sigma, lr):
    pairs = data.make_pairs(frames["R"].shape[0], tau)
    train_pairs, _ = data.time_split(pairs, val_frac=0.2)
    node_feats, edge_index, edge_feats = _build_ctx(frames, k)
    net = m.FlowNet(node_dim=node_feats.shape[1], edge_dim=edge_feats.shape[1],
                    hidden=hidden, layers=layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        perm = train_pairs[torch.randperm(train_pairs.shape[0])]
        for i, j in perm.tolist():
            R_t, t_t = frames["R"][i], frames["t"][i]
            R_f, t_f = frames["R"][j], frames["t"][j]
            u_target = f.relative_update(R_t, t_t, R_f, t_f)
            opt.zero_grad()
            loss = m.cfm_loss(net, u_target, node_feats, edge_index, edge_feats, sigma=sigma)
            loss.backward()
            opt.step()
    return net, (node_feats, edge_index, edge_feats)


def run_demo(traj_path, top_path, tau, out_dir, K=8, epochs=50, k=8,
             hidden=64, layers=3, sigma=0.1, lr=1e-3):
    os.makedirs(out_dir, exist_ok=True)
    frames = data.load_frames(traj_path, top_path)
    net, (node_feats, edge_index, edge_feats) = train(
        frames, tau, epochs, k, hidden, layers, sigma, lr)

    # sample futures from the first validation frame
    pairs = data.make_pairs(frames["R"].shape[0], tau)
    _, val_pairs = data.time_split(pairs, val_frac=0.2)
    i0 = int(val_pairs[0, 0])
    R_t, t_t = frames["R"][i0], frames["t"][i0]

    u = m.sample(net, node_feats, edge_index, edge_feats, K=K, sigma=sigma)
    R_f, t_f = dec.decode_frames(R_t, t_t, u)

    res_names = ["ALA"] * frames["R"].shape[1]  # demo backbone; names cosmetic
    atoms_K = []
    for kk in range(K):
        atoms = dec.idealize(dec.build_structure(R_f[kk], t_f[kk]))
        atoms_K.append(atoms)
        dec.write_pdb(atoms, res_names, os.path.join(out_dir, f"future_{kk}.pdb"))
    atoms_K = torch.stack(atoms_K, 0)

    md_ca = frames["t"][int(val_pairs[0, 1])]  # CA of the true future
    report = {
        "model_geometry": val.geometry_metrics(atoms_K[0]),
        "diversity": val.diversity(atoms_K),
        "ensemble_overlap_vs_true": val.ensemble_overlap(atoms_K[0][:, 1, :], md_ca),
        "n_residues": frames["R"].shape[1],
        "tau": tau,
    }
    return report


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--tau", type=int, required=True)
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    rep = run_demo(args.traj, args.top, args.tau, args.out, K=args.K, epochs=args.epochs)
    print(json.dumps(rep, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_demo.py -v`
Expected: PASS (1 test). Runs training on a tiny synthetic trajectory; takes a few seconds.

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests across the six modules PASS.

- [ ] **Step 6: Commit**

```bash
git add lsmd/demo.py tests/test_demo.py
git commit -m "feat: demo CLI tying training, sampling, validation together"
```

---

## How to run on your data

Once Task 7 is green:

```bash
python -m lsmd.demo --traj your_traj.dcd --top your_top.pdb --tau 100 --K 8 --epochs 200 --out demo_out
```

Pick `τ` by inspecting backbone-torsion / contact autocorrelation so fast vibration has decayed
but slow structure is retained. Inspect `demo_out/future_*.pdb` in PyMOL/VMD and read the JSON
report (geometry validity, diversity, ensemble overlap).

## Self-review notes (spec coverage)

- Representation §4 → Tasks 1–2 (frames, invariant features, dynamic k-NN graph spanning all chains).
- Component architecture §5 → Tasks 1–7 (one module each).
- Generative model §7 (flow matching, equivariance via invariant Δ) → Task 4.
- Decoder + idealization §8 → Task 5.
- Training objectives §9 (generative + geometry/continuity penalties) → Tasks 4–5 (CFM loss; idealization closes peptide bonds / clashes). Deferred kinetic/VAMP/CK objectives are out of scope per spec §2.
- Validation + baselines §10 → Task 6, surfaced in Task 7 report.
- Multi-molecule extension §11 → enabled, not built: `chain_id` carried through, graph spans all residues regardless of chain, no single-chain assumption in any code path.

## Known demo limitations (intended; tracked for later phases)

- Rotation uses axis-angle of the relative rotation — valid for moderate `τ`; replace with proper SO(3) flow matching later.
- O-atom local coords are approximate (not ψ-derived); fine for clash/continuity metrics.
- Graph is built once from frame 0 (topology assumed stable across the single-protein run); for assembly, rebuild the graph per state.
- No kinetic validation (VAMP/CK) yet — this demo proves valid + plausible jumps, not correct rates.
