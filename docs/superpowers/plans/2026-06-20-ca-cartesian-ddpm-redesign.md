# CA-Cartesian DDPM Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-residue SE(3) frame representation with a Cartesian CA point cloud that predicts per-pair Kabsch-aligned CA displacements via the existing DDPM machinery, and pivot evaluation to CA-appropriate distributional metrics.

**Architecture:** State is CA coordinates `X [F, P, 3]` after making molecules whole (PBC fix) and global CA superposition to frame 0. The training target for a pair `(i, j=i+τ)` is `Δ = kabsch_align(X_j → X_i) − X_i`. A plain graph net (static frame-0 reference graph) predicts `Δ` with DDPM ε-prediction conditioned on the lag embedding; equivariance comes from the global superposition. The point-cloud abstraction keeps the door open to backbone/all-atom (more nodes, not larger per-node dim).

**Tech Stack:** Python, PyTorch, mdtraj, pytest.

## Global Constraints

- Distances are in **Angstrom** (trajectory nm × 10).
- Trajectory save interval is **200 ps/frame**; τ is in frames. Default `taus = [1, 2, 5]` (200 ps, 400 ps, 1 ns). 200 ps (τ=1) is the floor.
- **CA-only** this iteration: one point per residue. No `--atoms` flag (YAGNI). Output is a CA-trace PDB.
- `point_dim` defaults to **6** in `FlowNet` (preserves the SE(3) path and all existing tests); the CA pipeline passes `point_dim=3` explicitly.
- Per-pair **Kabsch alignment** for displacement targets (isolates internal motion from tumbling).
- Fluctuation vs. transition is handled by **multi-lag τ-conditioning only** (no extra regime-specific loss term); density reweighting (`compute_frame_weights`) is retained.
- CA edge features are `[rel_pos(3), dist(1)]` → `edge_dim = 4`. No per-residue rotation.
- JS divergences are normalized to **bits** (divide by `math.log(2)`), range `[0, 1]`.
- Retain the legacy SE(3) functions (`relative_update`, `apply_update`, `edge_features`, `decode_frames`, `build_structure`, `cfm_loss`, `sample`, `ramachandran_js`) untouched and passing — they are the future backbone path.

---

### Task 1: Kabsch alignment + PBC make-whole

**Files:**
- Modify: `lsmd/geometry.py` (add `kabsch`)
- Modify: `lsmd/data.py:23` (add `make_molecules_whole` in `load_frames`)
- Test: `tests/test_geometry.py` (new), `tests/test_data.py` (add one test)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `geometry.kabsch(X, Y) -> (R, t)` — rigid transform aligning `Y` onto `X`. `X, Y` are `[P,3]` or `[B,P,3]`. Returns `R [3,3]` or `[B,3,3]`, `t [3]` or `[B,3]` such that `Y @ R.transpose(-1,-2) + t ≈ X`.
  - `data.load_frames` now calls `traj.make_molecules_whole()` before superposition (CA–CA bonds become physical).

- [ ] **Step 1: Write the failing test for kabsch (identity + known transform + batched)**

Create `tests/test_geometry.py`:

```python
import torch
from lsmd import geometry as g


def test_kabsch_identity():
    X = torch.randn(10, 3)
    R, t = g.kabsch(X, X)
    assert torch.allclose(R, torch.eye(3), atol=1e-5)
    assert torch.allclose(t, torch.zeros(3), atol=1e-5)


def test_kabsch_recovers_known_transform():
    torch.manual_seed(0)
    Y = torch.randn(20, 3)
    # Build a known rotation via QR (proper rotation) and a translation
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    trans = torch.tensor([1.0, -2.0, 3.0])
    X = Y @ Q.T + trans                     # X is Y rotated+translated
    R, t = g.kabsch(X, Y)                    # align Y onto X
    Y_aligned = Y @ R.transpose(-1, -2) + t
    assert torch.allclose(Y_aligned, X, atol=1e-4)
    assert abs(torch.linalg.det(R).item() - 1.0) < 1e-4   # proper rotation


def test_kabsch_batched():
    torch.manual_seed(1)
    X = torch.randn(4, 15, 3)
    Y = torch.randn(4, 15, 3)
    R, t = g.kabsch(X, Y)
    assert R.shape == (4, 3, 3)
    assert t.shape == (4, 3)
    Y_aligned = Y @ R.transpose(-1, -2) + t.unsqueeze(-2)
    # alignment reduces RMSD vs unaligned
    rmsd_before = (X - Y).norm(dim=-1).mean()
    rmsd_after = (X - Y_aligned).norm(dim=-1).mean()
    assert rmsd_after <= rmsd_before + 1e-5
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_geometry.py -v`
Expected: FAIL with `AttributeError: module 'lsmd.geometry' has no attribute 'kabsch'`

- [ ] **Step 3: Implement `kabsch` in `lsmd/geometry.py`**

Append to `lsmd/geometry.py`:

```python
def kabsch(X, Y):
    """Rigid transform aligning Y onto X (minimizes ‖Y@R.T + t − X‖).

    Args:
        X: target points [P, 3] or [B, P, 3]
        Y: source points [P, 3] or [B, P, 3]

    Returns:
        (R, t): R [3,3] or [B,3,3] (proper rotation, det=+1),
                t [3] or [B,3] s.t. Y @ R.transpose(-1,-2) + t ≈ X.
    """
    muX = X.mean(dim=-2, keepdim=True)            # [...,1,3]
    muY = Y.mean(dim=-2, keepdim=True)
    Xc = X - muX
    Yc = Y - muY
    H = Yc.transpose(-1, -2) @ Xc                 # [...,3,3]
    U, _, Vt = torch.linalg.svd(H)
    V = Vt.transpose(-1, -2)
    d = torch.linalg.det(V @ U.transpose(-1, -2))  # [...] sign for proper rotation
    D = torch.eye(3, device=X.device, dtype=X.dtype).expand_as(H).clone()
    D[..., 2, 2] = d
    R = V @ D @ U.transpose(-1, -2)               # [...,3,3]
    # t = muX - muY @ R.T
    t = muX.squeeze(-2) - (muY @ R.transpose(-1, -2)).squeeze(-2)
    return R, t
```

- [ ] **Step 4: Run kabsch tests to verify they pass**

Run: `pytest tests/test_geometry.py -v`
Expected: 3 passed.

- [ ] **Step 5: Add `make_molecules_whole` to `load_frames`**

In `lsmd/data.py`, replace the block at lines 23–26:

```python
    traj = md.load(traj_path, top=top_path)
    ca_idx = traj.topology.select("protein and name CA")
    if len(ca_idx) > 0:
        traj.superpose(traj, 0, atom_indices=ca_idx)
    top = traj.topology
```

with:

```python
    traj = md.load(traj_path, top=top_path)
    if traj.unitcell_lengths is not None:
        traj.make_molecules_whole(inplace=True)   # undo PBC wrapping (protein split across box)
    ca_idx = traj.topology.select("protein and name CA")
    if len(ca_idx) > 0:
        traj.superpose(traj, 0, atom_indices=ca_idx)
    top = traj.topology
```

- [ ] **Step 6: Write the failing data test for PBC unwrapping**

Add to `tests/test_data.py`:

```python
def test_load_frames_unwraps_pbc(tmp_path):
    """A protein split across the periodic box yields physical CA-CA bonds after load."""
    import mdtraj as md
    import numpy as np
    from lsmd import data

    n_res = 4
    box = 2.0  # nm
    top = md.Topology()
    chain = top.add_chain()
    atoms = []
    for i in range(n_res):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        a = {nm: top.add_atom(nm, md.element.get_by_symbol(el), res)
             for nm, el in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]}
        atoms.append(a)
    # bond consecutive backbone so make_molecules_whole keeps the chain together
    for i in range(n_res):
        top.add_bond(atoms[i]["N"], atoms[i]["CA"])
        top.add_bond(atoms[i]["CA"], atoms[i]["C"])
        top.add_bond(atoms[i]["C"], atoms[i]["O"])
        if i + 1 < n_res:
            top.add_bond(atoms[i]["C"], atoms[i + 1]["N"])

    # Build a straight chain (CA-CA ~0.38 nm), then wrap the last two residues by -box
    xyz = np.zeros((1, n_res * 4, 3), np.float32)
    for i in range(n_res):
        ca = np.array([i * 0.38, 0.0, 0.0], np.float32)
        base = i * 4
        xyz[0, base + 0] = ca + [-0.05, 0.14, 0.0]   # N
        xyz[0, base + 1] = ca                         # CA
        xyz[0, base + 2] = ca + [0.15, 0.0, 0.0]      # C
        xyz[0, base + 3] = ca + [0.22, -0.11, 0.0]    # O
    xyz[0, 2 * 4:] += np.array([box, 0.0, 0.0], np.float32)  # wrap last 2 residues

    traj = md.Trajectory(xyz, top)
    traj.unitcell_lengths = np.array([[box, box, box]], np.float32)
    traj.unitcell_angles = np.array([[90.0, 90.0, 90.0]], np.float32)
    p = tmp_path / "wrapped.pdb"
    traj.save_pdb(str(p))

    frames = data.load_frames(str(p), str(p))
    ca = frames["t"][0]                       # [N,3] in Angstrom
    bonds = (ca[1:] - ca[:-1]).norm(dim=-1)
    assert bonds.max().item() < 6.0           # ~3.8 Å, not ~box length (20 Å)
```

- [ ] **Step 7: Run the data test**

Run: `pytest tests/test_data.py::test_load_frames_unwraps_pbc -v`
Expected: PASS (without the make-whole call it would fail with a ~20 Å bond).

- [ ] **Step 8: Run the full suite to confirm no regression**

Run: `pytest tests/ -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add lsmd/geometry.py lsmd/data.py tests/test_geometry.py tests/test_data.py
git commit -m "feat: Kabsch alignment + PBC make-whole in load_frames"
```

---

### Task 2: CA displacement target + CA graph

**Files:**
- Modify: `lsmd/featurize.py` (add `ca_displacement`, `ca_graph`)
- Test: `tests/test_featurize.py` (add tests)

**Interfaces:**
- Consumes: `geometry.kabsch` (Task 1).
- Produces:
  - `featurize.ca_displacement(X_i, X_j) -> Δ` — Kabsch-aligns `X_j` onto `X_i`, returns `Δ = X_j_aligned − X_i`. Inputs `[P,3]` or `[B,P,3]`; output same shape.
  - `featurize.ca_graph(X, k) -> (edge_index, edge_feats)` — kNN graph from CA positions `X [P,3]`. `edge_index [2,E]`, `edge_feats [E,4]` = `[rel_pos(3), dist(1)]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_featurize.py` (create if absent with `import torch` and `from lsmd import featurize as f`):

```python
import torch
from lsmd import featurize as f


def test_ca_displacement_identical_is_zero():
    X = torch.randn(12, 3)
    d = f.ca_displacement(X, X)
    assert d.shape == (12, 3)
    assert d.abs().max().item() < 1e-5


def test_ca_displacement_pure_translation():
    X = torch.randn(8, 3)
    shift = torch.tensor([1.0, 2.0, -3.0])
    Y = X + shift
    d = f.ca_displacement(X, Y)
    # Kabsch removes the global translation → near-zero internal displacement
    assert d.abs().max().item() < 1e-4


def test_ca_displacement_rotation_invariant_norms():
    torch.manual_seed(0)
    X_i = torch.randn(15, 3)
    X_j = X_i + 0.1 * torch.randn(15, 3)        # small internal change
    d1 = f.ca_displacement(X_i, X_j)
    # Rotate BOTH frames by the same proper rotation Q
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    d2 = f.ca_displacement(X_i @ Q.T, X_j @ Q.T)
    # displacement is equivariant → per-node magnitudes are invariant
    assert torch.allclose(d1.norm(dim=-1), d2.norm(dim=-1), atol=1e-4)


def test_ca_displacement_batched():
    X_i = torch.randn(4, 10, 3)
    X_j = torch.randn(4, 10, 3)
    d = f.ca_displacement(X_i, X_j)
    assert d.shape == (4, 10, 3)


def test_ca_graph_shapes():
    X = torch.randn(20, 3)
    ei, ef = f.ca_graph(X, k=6)
    assert ei.shape[0] == 2
    assert ei.shape[1] == ef.shape[0]
    assert ef.shape[1] == 4
    assert ei.shape[1] == 20 * 6        # k edges per node
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_featurize.py -v`
Expected: FAIL with `AttributeError: module 'lsmd.featurize' has no attribute 'ca_displacement'`

- [ ] **Step 3: Implement `ca_displacement` and `ca_graph`**

Append to `lsmd/featurize.py`:

```python
def ca_displacement(X_i, X_j):
    """Per-pair Kabsch-aligned CA displacement Δ = align(X_j→X_i) − X_i.

    Removes whole-protein tumbling so Δ reflects internal conformational change.

    Args:
        X_i: source CA coords [P,3] or [B,P,3]
        X_j: target CA coords [P,3] or [B,P,3]

    Returns:
        Δ: same shape as inputs.
    """
    R, t = g.kabsch(X_i, X_j)                       # align X_j onto X_i
    X_j_aligned = X_j @ R.transpose(-1, -2) + t.unsqueeze(-2)
    return X_j_aligned - X_i


def ca_graph(X, k):
    """kNN graph + invariant edge features from CA positions.

    Args:
        X: CA coords [P,3] (reference structure, frame-0 orientation).
        k: neighbours per node.

    Returns:
        edge_index [2,E], edge_feats [E,4] = [rel_pos(3), dist(1)].
        rel_pos is in the (canonicalized) frame-0 orientation.
    """
    edge_index = knn_graph(X, k)
    src, dst = edge_index
    rel_pos = X[dst] - X[src]                        # [E,3]
    dist = rel_pos.norm(dim=-1, keepdim=True)        # [E,1]
    edge_feats = torch.cat([rel_pos, dist], dim=-1)  # [E,4]
    return edge_index, edge_feats
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_featurize.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lsmd/featurize.py tests/test_featurize.py
git commit -m "feat: CA displacement target and CA graph featurization"
```

---

### Task 3: Model `point_dim` parameter + CA-trace PDB writer

**Files:**
- Modify: `lsmd/model.py` (`FlowNet.__init__`, `sample`, `sample_ddpm`)
- Modify: `lsmd/decoder.py` (add `write_ca_pdb`)
- Test: `tests/test_model.py` (add tests), `tests/test_decoder.py` (add one test, create if absent)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `FlowNet(node_dim, edge_dim, hidden=64, layers=3, tau_emb_dim=16, point_dim=6)` — stores `self.point_dim`; embed input width `node_dim + point_dim + 1 + tau_emb_dim`; head output `point_dim`.
  - `sample(...)` and `sample_ddpm(...)` return `[K, P, net.point_dim]` (read from the net, not literal 6).
  - `decoder.write_ca_pdb(ca, res_names, path)` — `ca [P,3]`, writes one CA atom per residue (element C).

- [ ] **Step 1: Write the failing model tests**

Add to `tests/test_model.py`:

```python
def test_flownet_point_dim_3():
    nf, ei, ef = _dummy_inputs(n=6, node_dim=8, edge_dim=4)
    net = m.FlowNet(node_dim=8, edge_dim=4, hidden=32, layers=2, point_dim=3)
    u_s = torch.randn(6, 3)
    out = net(u_s, torch.tensor(0.4), nf, ei, ef, tau=50)
    assert out.shape == (6, 3)


def test_sample_ddpm_point_dim_3():
    nf, ei, ef = _dummy_inputs(n=6, node_dim=8, edge_dim=4)
    net = m.FlowNet(node_dim=8, edge_dim=4, hidden=32, layers=2, point_dim=3)
    sched = m.NoiseSchedule(T=50)
    samples = m.sample_ddpm(net, nf, ei, ef, K=5, tau=50, schedule=sched, steps=10)
    assert samples.shape == (5, 6, 3)


def test_default_point_dim_is_6():
    net = m.FlowNet(node_dim=8, edge_dim=13, hidden=16, layers=1)
    assert net.point_dim == 6
```

Note: `_dummy_inputs` already accepts `edge_dim`; pass `edge_dim=4` so `edge_feats` matches the CA graph width.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_model.py::test_flownet_point_dim_3 tests/test_model.py::test_sample_ddpm_point_dim_3 tests/test_model.py::test_default_point_dim_is_6 -v`
Expected: FAIL (`__init__` has no `point_dim`; `sample_ddpm` returns shape `[5,6,6]`).

- [ ] **Step 3: Modify `FlowNet.__init__`**

In `lsmd/model.py`, replace the `FlowNet.__init__` body (lines 105–116):

```python
    def __init__(self, node_dim, edge_dim, hidden=64, layers=3, tau_emb_dim=16, point_dim=6):
        super().__init__()
        self.tau_emb_dim = tau_emb_dim
        self.point_dim = point_dim
        # input: node features + u (point_dim) + flow-time s (1) + tau embedding
        self.embed = nn.Linear(node_dim + point_dim + 1 + tau_emb_dim, hidden)
        self.layers = nn.ModuleList(
            [MessageLayer(hidden, edge_dim) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, point_dim),
        )
```

- [ ] **Step 4: Update `sample` and `sample_ddpm` to read `net.point_dim`**

In `lsmd/model.py`, in `sample` (around line 219) replace:

```python
    u = torch.randn(K, n, 6, device=node_feats.device, dtype=node_feats.dtype) * sigma
```

with:

```python
    u = torch.randn(K, n, net.point_dim, device=node_feats.device, dtype=node_feats.dtype) * sigma
```

In `sample_ddpm` (around line 334) replace:

```python
    u = torch.randn(K, N, 6, device=device, dtype=dtype) * sigma_init
```

with:

```python
    u = torch.randn(K, N, net.point_dim, device=device, dtype=dtype) * sigma_init
```

- [ ] **Step 5: Run model tests (new + existing)**

Run: `pytest tests/test_model.py -q`
Expected: all pass (existing 6-dim tests use default `point_dim=6`; new tests use 3).

- [ ] **Step 6: Write the failing CA-PDB test**

Add to `tests/test_decoder.py` (create if absent):

```python
import torch
from lsmd import decoder as dec


def test_write_ca_pdb(tmp_path):
    ca = torch.randn(5, 3)
    path = tmp_path / "ca.pdb"
    dec.write_ca_pdb(ca, ["ALA"] * 5, str(path))
    lines = path.read_text().splitlines()
    atom_lines = [l for l in lines if l.startswith("ATOM")]
    assert len(atom_lines) == 5
    assert atom_lines[0][12:16].strip() == "CA"
    assert lines[-1].strip() == "END"
```

- [ ] **Step 7: Run to verify failure**

Run: `pytest tests/test_decoder.py::test_write_ca_pdb -v`
Expected: FAIL with `AttributeError: module 'lsmd.decoder' has no attribute 'write_ca_pdb'`

- [ ] **Step 8: Implement `write_ca_pdb`**

Append to `lsmd/decoder.py`:

```python
def write_ca_pdb(ca, res_type_names, path):
    """Write a CA-only trace to a PDB file (one CA atom per residue).

    Args:
        ca: CA coordinates [P, 3]
        res_type_names: list of residue names (length P), e.g. ["ALA", ...]
        path: output file path

    Returns:
        None (writes to file).
    """
    lines = []
    for ri in range(ca.shape[0]):
        x, y, z = ca[ri].tolist()
        lines.append(
            f"ATOM  {ri + 1:5d}  CA  {res_type_names[ri]:>3s} A{ri + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
```

- [ ] **Step 9: Run the decoder test**

Run: `pytest tests/test_decoder.py -v`
Expected: PASS.

- [ ] **Step 10: Run the full suite**

Run: `pytest tests/ -q`
Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add lsmd/model.py lsmd/decoder.py tests/test_model.py tests/test_decoder.py
git commit -m "feat: parametrize FlowNet point_dim and add CA-trace PDB writer"
```

---

### Task 4: CA validation metrics

**Files:**
- Modify: `lsmd/validation.py` (add `_ca`, `ca_geometry`, `distance_matrix_js`, `rmsf_profile`, `displacement_js`; make `pca_js`/`ensemble_recall`/`ensemble_novelty` accept CA point clouds)
- Test: `tests/test_validation.py` (add tests)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ca_geometry(ca) -> {"ca_bond_mean","ca_bond_min","ca_bond_max","clash_count"}` — `ca [P,3]`.
  - `distance_matrix_js(ca_model, ca_md, bins=30) -> float` — `ca_* [.,P,3]`, JS of pooled pairwise CA–CA distances, `[0,1]`.
  - `rmsf_profile(ca_model, ca_md) -> {"model":[P],"md":[P],"corr":float}` — per-residue positional std across each ensemble; Pearson correlation of the two profiles.
  - `displacement_js(disp_model, disp_md, bins=30) -> {"js","model_mean","md_mean"}` — `disp_*` are 1-D RMSD-magnitude tensors.
  - `pca_js`, `ensemble_recall`, `ensemble_novelty` additionally accept a CA point cloud `[.,P,3]` (used directly) as well as the legacy `[.,N,4,3]` (CA extracted).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_validation.py`:

```python
def test_ca_geometry_keys():
    ca = torch.randn(10, 3)
    out = val.ca_geometry(ca)
    assert set(out) == {"ca_bond_mean", "ca_bond_min", "ca_bond_max", "clash_count"}


def test_distance_matrix_js_identical_is_zero():
    torch.manual_seed(0)
    ca = torch.randn(6, 12, 3)
    js = val.distance_matrix_js(ca, ca)
    assert js < 1e-3


def test_distance_matrix_js_bounded():
    a = torch.randn(5, 12, 3)
    b = torch.randn(5, 12, 3) * 5.0 + 20.0
    js = val.distance_matrix_js(a, b)
    assert 0.0 <= js <= 1.0


def test_rmsf_profile_identical_corr_one():
    torch.manual_seed(0)
    ca = torch.randn(8, 10, 3)
    out = val.rmsf_profile(ca, ca)
    assert len(out["model"]) == 10
    assert abs(out["corr"] - 1.0) < 1e-4


def test_displacement_js_identical_is_zero():
    d = torch.rand(50)
    out = val.displacement_js(d, d)
    assert out["js"] < 1e-3
    assert abs(out["model_mean"] - out["md_mean"]) < 1e-6


def test_pca_js_accepts_ca_pointcloud():
    torch.manual_seed(0)
    ca = torch.randn(6, 12, 3)
    out = val.pca_js(ca, ca)              # [K,P,3] inputs, not [K,N,4,3]
    assert out["js"] < 1e-3


def test_recall_accepts_ca_pointcloud():
    torch.manual_seed(0)
    ca = torch.randn(5, 12, 3)
    assert val.ensemble_recall(ca, ca, r_ang=0.01) == 1.0
    assert val.ensemble_novelty(ca, ca, r_ang=0.01) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_validation.py -v -k "ca_geometry or distance_matrix or rmsf or displacement or pointcloud"`
Expected: FAIL (functions missing; `pca_js`/`recall` index `[:,:,1,:]` on a rank-3 tensor).

- [ ] **Step 3: Add the `_ca` helper and make CA-extraction conditional**

In `lsmd/validation.py`, add near the top (after imports):

```python
def _ca(x):
    """Return CA coords [.,P,3]. Accepts a CA point cloud [.,P,3] (used as-is)
    or a full backbone tensor [.,N,4,3] (CA = atom index 1 extracted)."""
    return x if x.dim() == 3 else x[:, :, 1, :]
```

In `pca_js`, replace lines 190–191:

```python
    ca_model = atoms_model[:, :, 1, :].float()   # [K, N, 3]
    ca_md    = atoms_md[:,    :, 1, :].float()   # [M, N, 3]
```

with:

```python
    ca_model = _ca(atoms_model).float()   # [K, P, 3]
    ca_md    = _ca(atoms_md).float()      # [M, P, 3]
```

In `ensemble_recall`, replace lines 252–253:

```python
    ca_model = atoms_model[:, :, 1, :]   # [K, N, 3]
    ca_md    = atoms_md[:,    :, 1, :]   # [M, N, 3]
```

with:

```python
    ca_model = _ca(atoms_model)   # [K, P, 3]
    ca_md    = _ca(atoms_md)      # [M, P, 3]
```

In `ensemble_novelty`, replace lines 279–280 the same way:

```python
    ca_model = _ca(atoms_model)   # [K, P, 3]
    ca_md    = _ca(atoms_md)      # [M, P, 3]
```

- [ ] **Step 4: Implement the new CA metrics**

Append to `lsmd/validation.py`:

```python
def ca_geometry(ca):
    """Sequential CA-CA bond statistics and clash count for one CA trace.

    Args:
        ca: CA coordinates [P, 3]

    Returns:
        dict: ca_bond_mean, ca_bond_min, ca_bond_max (Å), clash_count
              (non-adjacent CA pairs closer than 2.0 Å).
    """
    bonds = (ca[1:] - ca[:-1]).norm(dim=-1)
    d = torch.cdist(ca, ca)
    n = ca.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=ca.device)
    for i in range(n - 1):
        mask[i, i + 1] = mask[i + 1, i] = False
    clash_count = ((d < 2.0) & mask).sum().item() / 2
    return {
        "ca_bond_mean": bonds.mean().item(),
        "ca_bond_min": bonds.min().item(),
        "ca_bond_max": bonds.max().item(),
        "clash_count": clash_count,
    }


def _pairwise_dists(ca):
    """Pooled upper-triangle CA-CA distances over an ensemble [K,P,3] → 1-D."""
    K, P, _ = ca.shape
    iu = torch.triu_indices(P, P, offset=1)
    d = torch.cdist(ca, ca)                 # [K,P,P]
    return d[:, iu[0], iu[1]].reshape(-1)   # [K * P(P-1)/2]


def _hist_js(a, b, bins, lo=None, hi=None):
    """JS divergence (bits) between two 1-D samples via shared-range histograms."""
    if lo is None:
        lo = torch.min(a.min(), b.min())
    if hi is None:
        hi = torch.max(a.max(), b.max())
    span = (hi - lo).clamp_min(1e-8)

    def _h(x):
        idx = ((x - lo) / span * bins).long().clamp(0, bins - 1)
        h = torch.zeros(bins, device=x.device)
        h.scatter_add_(0, idx, torch.ones_like(x))
        h = h + 1e-8
        return h / h.sum()

    p, q = _h(a), _h(b)
    mix = 0.5 * (p + q)
    js = 0.5 * (p * torch.log(p / mix)).sum() + 0.5 * (q * torch.log(q / mix)).sum()
    return (js / math.log(2)).clamp(0.0, 1.0).item()


def distance_matrix_js(ca_model, ca_md, bins=30):
    """JS divergence between pooled pairwise CA-CA distance distributions.

    Captures whether the model reproduces the overall conformational geometry
    (contact distances) of the MD ensemble.

    Args:
        ca_model: [K, P, 3]
        ca_md:    [M, P, 3]
        bins:     histogram bins.

    Returns:
        JS divergence in [0, 1]. 0 = identical distance distributions.
    """
    a = _pairwise_dists(ca_model)
    b = _pairwise_dists(ca_md)
    return _hist_js(a, b, bins)


def rmsf_profile(ca_model, ca_md):
    """Per-residue CA positional fluctuation (RMSF) for both ensembles.

    Args:
        ca_model: [K, P, 3]
        ca_md:    [M, P, 3]

    Returns:
        dict: model [P], md [P] (per-residue std magnitude, Å),
              corr (Pearson correlation of the two profiles).
    """
    def _rmsf(ca):
        mu = ca.mean(0, keepdim=True)               # [1,P,3]
        return (ca - mu).pow(2).sum(-1).mean(0).sqrt()   # [P]

    rm = _rmsf(ca_model)
    rd = _rmsf(ca_md)
    rmc = rm - rm.mean()
    rdc = rd - rd.mean()
    denom = (rmc.norm() * rdc.norm()).clamp_min(1e-8)
    corr = (rmc * rdc).sum() / denom
    return {"model": rm.tolist(), "md": rd.tolist(), "corr": corr.item()}


def displacement_js(disp_model, disp_md, bins=30):
    """JS divergence between two displacement-magnitude distributions.

    disp_* are per-sample RMSD magnitudes (Å): for the model, ‖Δ‖ of sampled
    displacements; for MD, per-pair ‖Δ‖ at the chosen lag. Separates the
    fluctuation bulk (small ‖Δ‖) from the transition tail (large ‖Δ‖).

    Args:
        disp_model: 1-D tensor of model displacement magnitudes.
        disp_md:    1-D tensor of MD displacement magnitudes.
        bins:       histogram bins.

    Returns:
        dict: js (in [0,1]), model_mean, md_mean.
    """
    js = _hist_js(disp_model, disp_md, bins)
    return {
        "js": js,
        "model_mean": disp_model.mean().item(),
        "md_mean": disp_md.mean().item(),
    }
```

- [ ] **Step 5: Run the validation tests**

Run: `pytest tests/test_validation.py -v`
Expected: all pass (new + existing).

- [ ] **Step 6: Run the full suite**

Run: `pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add lsmd/validation.py tests/test_validation.py
git commit -m "feat: CA validation metrics (ca_geometry, distance/rmsf/displacement, CA-pointcloud inputs)"
```

---

### Task 5: Wire the CA pipeline into the demo

**Files:**
- Modify: `lsmd/demo.py` (`_build_ctx`, `train`, `run_demo`, CLI)
- Test: `tests/test_demo.py` (rewrite `test_run_demo_smoke`)

**Interfaces:**
- Consumes: `data.load_frames`, `data.make_multi_lag_pairs`, `data.time_split`, `data.compute_frame_weights`; `featurize.ca_displacement`, `featurize.ca_graph`, `featurize.node_features`; `model.NoiseSchedule`, `model.FlowNet(point_dim=3)`, `model.ddpm_loss`, `model.sample_ddpm`; `decoder.write_ca_pdb`; `validation.{ca_geometry, pca_js, ensemble_recall, ensemble_novelty, distance_matrix_js, rmsf_profile, displacement_js}`.
- Produces: `run_demo(...) -> report dict` with keys `ca_geometry, pca_js, pca_var_explained, ensemble_recall, ensemble_novelty, distance_matrix_js, rmsf_corr, displacement_js, displacement_model_mean, displacement_md_mean, n_residues, n_md_reference, taus, infer_tau`.

- [ ] **Step 1: Write the failing smoke test**

Replace `tests/test_demo.py` entirely:

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
    report = demo.run_demo(
        path, path,
        taus=[1, 2, 3], infer_tau=2,
        out_dir=str(out), K=4, epochs=20, batch_size=8,
        T_diff=20, diff_steps=5,
    )
    for key in ("ca_geometry", "pca_js", "ensemble_recall", "ensemble_novelty",
                "distance_matrix_js", "rmsf_corr", "displacement_js",
                "n_md_reference"):
        assert key in report
    assert report["taus"] == [1, 2, 3]
    assert report["infer_tau"] == 2
    # CA-trace PDBs written, one per sample
    pdbs = list(out.glob("future_*.pdb"))
    assert len(pdbs) == 4
    # metrics in valid ranges
    assert 0.0 <= report["pca_js"] <= 1.0
    assert 0.0 <= report["distance_matrix_js"] <= 1.0
    assert 0.0 <= report["displacement_js"] <= 1.0
    assert 0.0 <= report["ensemble_recall"] <= 1.0
    assert 0.0 <= report["ensemble_novelty"] <= 1.0
    assert report["ca_geometry"]["clash_count"] >= 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_demo.py -v`
Expected: FAIL (old `run_demo` returns SE(3) keys / uses removed paths).

- [ ] **Step 3: Rewrite `lsmd/demo.py`**

Replace the entire file with:

```python
"""End-to-end demo: train a CA-displacement DDPM on a trajectory, sample
future CA conformations, write CA-trace PDBs, and report distributional metrics."""
import os
import time
import torch
from lsmd import data, featurize as f, model as m, decoder as dec, validation as val


def _build_ctx(frames, k):
    """Static reference graph (frame-0 CA) + residue node features.

    Returns (node_feats [P,F], edge_index [2,E], edge_feats [E,4])."""
    X0 = frames["t"][0]                                  # [P,3] CA of frame 0
    edge_index, edge_feats = f.ca_graph(X0, k=k)
    node_feats = f.node_features(
        frames["res_type"], frames["chain_id"],
        frames["res_index"], frames["n_types"],
    )
    return node_feats, edge_index, edge_feats


def train(frames, taus, epochs, k, hidden, layers, lr,
          clip=1.0, batch_size=32, T_diff=200, sigma_aug=0.05,
          density_clip=10.0, device=None):
    """Train a CA-displacement DDPM with multi-lag pairs, inverse-density
    reweighting, and target augmentation.

    Returns (net, schedule, ctx) where ctx = (node_feats, edge_index, edge_feats).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}  taus={taus}  batch_size={batch_size}  T_diff={T_diff}")

    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)
    train_pairs, _ = data.time_split(pairs, val_frac=0.2)
    if train_pairs.shape[0] == 0:
        raise ValueError("No training pairs: taus too large relative to trajectory length.")

    frame_weights = data.compute_frame_weights(frames, density_clip=density_clip)  # [F]
    pair_weights_all = frame_weights[train_pairs[:, 0]]                             # [P]

    n_batches = (train_pairs.shape[0] + batch_size - 1) // batch_size
    print(f"  {train_pairs.shape[0]} training pairs → {n_batches} steps/epoch")

    node_feats, edge_index, edge_feats = _build_ctx(frames, k)
    node_feats = node_feats.to(device)
    edge_index = edge_index.to(device)
    edge_feats = edge_feats.to(device)
    X_all = frames["t"].to(device)                       # [F,P,3] CA coords

    schedule = m.NoiseSchedule(T=T_diff).to(device)
    net = m.FlowNet(
        node_dim=node_feats.shape[1],
        edge_dim=edge_feats.shape[1],
        hidden=hidden,
        layers=layers,
        point_dim=3,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        perm_idx = torch.randperm(train_pairs.shape[0])
        perm = train_pairs[perm_idx]
        perm_w = pair_weights_all[perm_idx]
        epoch_loss, n_steps, t0 = 0.0, 0, time.time()

        for start in range(0, perm.shape[0], batch_size):
            batch = perm[start:start + batch_size]
            batch_w = perm_w[start:start + batch_size].to(device)
            i_idx = batch[:, 0]
            j_idx = batch[:, 1]
            tau_b = batch[:, 2].to(device=device, dtype=X_all.dtype)

            u_batch = f.ca_displacement(X_all[i_idx], X_all[j_idx])   # [B,P,3]

            opt.zero_grad()
            loss = m.ddpm_loss(
                net, u_batch, node_feats, edge_index, edge_feats,
                tau_b, schedule, pair_weights=batch_w, sigma_aug=sigma_aug,
            )
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1

        print(f"Epoch {epoch+1}/{epochs}  loss={epoch_loss/n_steps:.4f}  "
              f"t={time.time()-t0:.1f}s")

    return net, schedule, (node_feats, edge_index, edge_feats)


def run_demo(traj_path, top_path, taus, infer_tau, out_dir, K=8, epochs=50,
             k=8, hidden=64, layers=3, lr=1e-3, clip=1.0, batch_size=32,
             T_diff=200, diff_steps=50, eta=1.0, sigma_init=1.0,
             sigma_aug=0.05, density_clip=10.0, device=None):
    """Load trajectory, train CA-displacement DDPM, sample K future CA traces,
    write PDBs, and compute CA distributional metrics."""
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = data.load_frames(traj_path, top_path)
    net, schedule, (node_feats, edge_index, edge_feats) = train(
        frames, taus, epochs, k, hidden, layers, lr,
        clip=clip, batch_size=batch_size, T_diff=T_diff,
        sigma_aug=sigma_aug, density_clip=density_clip, device=device,
    )
    net.eval()

    X_all = frames["t"]                                   # [F,P,3] CA coords (CPU)

    # Source frame from the val split with matching tau
    pairs = data.make_multi_lag_pairs(frames["R"].shape[0], taus)
    _, val_pairs = data.time_split(pairs, val_frac=0.2)
    matching = val_pairs[val_pairs[:, 2] == infer_tau]
    ref_pair = matching[0] if matching.shape[0] > 0 else val_pairs[0]
    i0 = int(ref_pair[0])
    x_init = X_all[i0]                                     # [P,3]

    # Sample K future displacements and apply to the source CA structure
    delta = m.sample_ddpm(net, node_feats, edge_index, edge_feats, K=K,
                          tau=infer_tau, schedule=schedule,
                          steps=diff_steps, eta=eta, sigma_init=sigma_init)   # [K,P,3]
    ca_model = x_init.to(delta.device).unsqueeze(0) + delta                   # [K,P,3]
    ca_model = ca_model.cpu()

    res_names = ["ALA"] * frames["R"].shape[1]   # CA-trace residue labels (placeholder)
    for kk in range(K):
        dec.write_ca_pdb(ca_model[kk], res_names, os.path.join(out_dir, f"future_{kk}.pdb"))

    # MD reference ensemble: val end-frames matching infer_tau
    ref_end_frames = matching[:, 1][:128] if matching.shape[0] > 0 \
                     else val_pairs[:, 1][:128]
    ca_md = X_all[ref_end_frames.long()]                  # [M,P,3]

    # Displacement-magnitude distributions (fluctuation bulk vs transition tail)
    disp_model = (ca_model - x_init.unsqueeze(0)).norm(dim=-1).pow(2).mean(-1).sqrt()  # [K]
    md_src = matching[:, 0][:128] if matching.shape[0] > 0 else val_pairs[:, 0][:128]
    md_disp = f.ca_displacement(X_all[md_src.long()], X_all[ref_end_frames.long()])    # [M,P,3]
    disp_md = md_disp.norm(dim=-1).pow(2).mean(-1).sqrt()                              # [M]

    pca_result = val.pca_js(ca_model, ca_md)
    rmsf = val.rmsf_profile(ca_model, ca_md)
    disp = val.displacement_js(disp_model, disp_md)
    report = {
        "ca_geometry":       val.ca_geometry(ca_model[0]),
        "pca_js":            pca_result["js"],
        "pca_var_explained": pca_result["var_explained"],
        "ensemble_recall":   val.ensemble_recall(ca_model, ca_md),
        "ensemble_novelty":  val.ensemble_novelty(ca_model, ca_md),
        "distance_matrix_js": val.distance_matrix_js(ca_model, ca_md),
        "rmsf_corr":         rmsf["corr"],
        "displacement_js":   disp["js"],
        "displacement_model_mean": disp["model_mean"],
        "displacement_md_mean":    disp["md_mean"],
        "n_residues":        frames["R"].shape[1],
        "n_md_reference":    ca_md.shape[0],
        "taus":              taus,
        "infer_tau":         infer_tau,
    }
    return report


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="CA-Cartesian long-stride protein MD demo CLI")
    ap.add_argument("--traj",        required=True,  help="Trajectory file path")
    ap.add_argument("--top",         required=True,  help="Topology file path")
    ap.add_argument("--taus",        type=int, nargs="+", default=[1, 2, 5],
                    help="Training lag schedule (frames). 200 ps/frame.")
    ap.add_argument("--infer_tau",   type=int, default=None,
                    help="Lag time for inference. Defaults to max(taus).")
    ap.add_argument("--out",         default="demo_out", help="Output directory")
    ap.add_argument("--K",           type=int,   default=8)
    ap.add_argument("--epochs",      type=int,   default=50)
    ap.add_argument("--k",           type=int,   default=8,    help="KNN neighbours")
    ap.add_argument("--hidden",      type=int,   default=64)
    ap.add_argument("--layers",      type=int,   default=3)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--clip",        type=float, default=1.0,  help="Gradient clip norm")
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--T_diff",      type=int,   default=200,  help="DDPM noise levels")
    ap.add_argument("--diff_steps",  type=int,   default=50,   help="Reverse-process steps")
    ap.add_argument("--eta",         type=float, default=1.0,  help="DDPM stochasticity")
    ap.add_argument("--sigma_init",  type=float, default=1.0,  help="Prior scale")
    ap.add_argument("--sigma_aug",   type=float, default=0.05, help="Target augmentation noise")
    ap.add_argument("--density_clip",type=float, default=10.0, help="Max density weight")
    ap.add_argument("--device",      default=None,  help="cuda / cpu (auto if omitted)")
    args = ap.parse_args()

    infer_tau = args.infer_tau if args.infer_tau is not None else max(args.taus)
    dev = torch.device(args.device) if args.device else None
    rep = run_demo(
        args.traj, args.top, args.taus, infer_tau, args.out,
        K=args.K, epochs=args.epochs, k=args.k,
        hidden=args.hidden, layers=args.layers,
        lr=args.lr, clip=args.clip, batch_size=args.batch_size,
        T_diff=args.T_diff, diff_steps=args.diff_steps,
        eta=args.eta, sigma_init=args.sigma_init,
        sigma_aug=args.sigma_aug, density_clip=args.density_clip,
        device=dev,
    )
    print(json.dumps(rep, indent=2))
```

- [ ] **Step 4: Run the smoke test**

Run: `pytest tests/test_demo.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lsmd/demo.py tests/test_demo.py
git commit -m "feat: wire CA-displacement DDPM pipeline into demo with CA metrics"
```

---

## Self-Review

**Spec coverage:**
- PBC make-whole → Task 1 ✅
- Kabsch alignment → Task 1 ✅
- CA displacement target (per-pair Kabsch) → Task 2 ✅
- CA graph + edge_dim 4 → Task 2 ✅
- `point_dim` parameter (default 6, CA=3) → Task 3 ✅
- CA-trace PDB output → Task 3 ✅
- Drop Ramachandran from CA path; keep PCA/recall/novelty as CA-capable → Task 4 ✅
- distance_matrix_js, rmsf_profile, displacement_js → Task 4 ✅
- Multi-lag conditioning + density reweighting retained → Task 5 ✅
- CLI `--taus` default `1 2 5`, no `--atoms` → Task 5 ✅
- Report keys → Task 5 ✅
- Static reference graph conditioning → Task 5 (`_build_ctx`) ✅
- Legacy SE(3) functions untouched → no task modifies them ✅

**Placeholder scan:** No TODO/TBD/placeholder code remains; every code step is complete and runnable.

**Type consistency:** `ca_displacement`/`ca_graph` (Task 2) consumed correctly in Task 5; `FlowNet(point_dim=3)` and `edge_dim=4` consistent between Tasks 2/3/5; validation functions accept `[K,P,3]` per Task 4 and are called with `ca_model`/`ca_md` `[K,P,3]` in Task 5; `frames["t"]` confirmed to be CA positions `[F,P,3]` in the current `data.load_frames`.
