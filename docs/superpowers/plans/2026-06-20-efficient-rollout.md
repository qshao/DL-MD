# Plan 3 — Efficient Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut rollout cost by splitting the propagator into a structural **encoder** (run once per propagation step over the static graph) and a lightweight **denoiser** (run per reverse-diffusion step), with a cached sampler — a pure speedup, no behavioral change.

**Architecture:** Within one propagation step the reverse diffusion runs ~20–50 iterations over a fixed graph; only the noisy update `u` and flow-time `s` change. The current `PropagatorNet` injects `u` at layer 0, so all `L` message layers recompute every reverse step. We split it: a `StructuralEncoder` does the `L` message-passing layers over `node_feats`/`edges`/`tau` once → a per-node `context`; a `Denoiser` (default 1 message layer) consumes `context` + `(u, s)` per reverse step. `CachedPropagator.forward` composes both (drop-in for `ddpm_loss_union`), and `sample_ddpm_union_cached` encodes once then loops the denoiser.

**Tech Stack:** Python, PyTorch, pytest. Reuses Plan-1 `lsmd/transfer_model.py` (`UnionMessageLayer`, `ddpm_loss_union`, the `sample_ddpm_union` reverse-step math) and `lsmd/model.py` (`tau_embedding`, `NoiseSchedule`).

## Global Constraints

- Core dims unchanged: `point_dim = 6`, `edge_dim = 13`, `node_dim = 24`, `tau_emb_dim = 16`, default `hidden = 128`, encoder `layers = 4`.
- Denoiser depth `n_denoise_layers` defaults to **1**; `0` = pure per-node MLP (opt-in).
- Conditioning split: **`tau` (fixed lag) enters the encoder**; **`s` (per-step diffusion flow-time) enters the denoiser**. This is deliberate — the per-step quantity must be the cheap one.
- `CachedPropagator.forward` keeps the **exact signature** of `PropagatorNet.forward` — `(u, s, node_feats, edge_index, edge_feats, tau, batch) -> [ΣN, point_dim]` — so `ddpm_loss_union` works on it unchanged.
- Correctness gate: caching must be a pure speedup — `forward(...)` must equal `denoise(u, s, encode(...), ...)` to numerical tolerance (the cached sampler recomputes nothing the uncached path would compute differently).
- Disjoint-union batching preserved: no cross-protein edges; `batch` broadcasts per-graph scalars. `batch` is moved to the feature device inside each module (as in the Plan-1 device fix).
- **Non-destructive:** append new classes/functions to `lsmd/transfer_model.py`; do not modify `PropagatorNet`, `ddpm_loss_union`, or `sample_ddpm_union`. The full suite stays green.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `lsmd/transfer_model.py` | Append `StructuralEncoder`, `Denoiser`, `CachedPropagator`, `sample_ddpm_union_cached` | Modify |
| `tests/test_cached_propagator.py` | Encoder/denoiser shapes, isolation, forward==encode∘denoise, loss compat, sampler | Create |

---

### Task 1: Structural encoder

**Files:**
- Modify: `lsmd/transfer_model.py` (append `StructuralEncoder`)
- Test: `tests/test_cached_propagator.py`

**Interfaces:**
- Consumes: `UnionMessageLayer` (Plan-1, same file), `lsmd.model.tau_embedding`.
- Produces:
  - `StructuralEncoder(node_dim=24, edge_dim=13, hidden=128, layers=4, tau_emb_dim=16)` with
    `forward(node_feats[ΣN,node_dim], edge_index[2,E], edge_feats[E,edge_dim], tau[G], batch[ΣN]) -> context[ΣN, hidden]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cached_propagator.py
import torch
from lsmd import batching
from lsmd import transfer_model as tm


def _toy_graph(n, k=4, tau=100.0):
    e = n * k
    return {
        "node_feats": torch.randn(n, 24),
        "edge_index": torch.randint(0, n, (2, e)),
        "edge_feats": torch.randn(e, 13),
        "u_target": torch.randn(n, 6),
        "tau": tau,
    }


def test_encoder_context_shape():
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    enc = tm.StructuralEncoder(hidden=32, layers=2)
    ctx = enc(u["node_feats"], u["edge_index"], u["edge_feats"], u["tau"], u["batch"])
    assert ctx.shape == (13, 32)


def test_encoder_graphs_do_not_interact():
    torch.manual_seed(0)
    g0, g1 = _toy_graph(5, tau=100.0), _toy_graph(8, tau=100.0)
    enc = tm.StructuralEncoder(hidden=32, layers=2).eval()
    solo = batching.union_collate([g0])
    pair = batching.union_collate([g0, g1])
    with torch.no_grad():
        c_solo = enc(solo["node_feats"], solo["edge_index"], solo["edge_feats"],
                     solo["tau"], solo["batch"])
        c_pair = enc(pair["node_feats"], pair["edge_index"], pair["edge_feats"],
                     pair["tau"], pair["batch"])
    assert torch.allclose(c_solo, c_pair[:5], atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cached_propagator.py -v`
Expected: FAIL with `AttributeError: module 'lsmd.transfer_model' has no attribute 'StructuralEncoder'`

- [ ] **Step 3: Append implementation to `lsmd/transfer_model.py`**

```python
class StructuralEncoder(nn.Module):
    """Per-node context from the static graph + lag tau. Run once per step.

    Carries the expensive L message-passing layers. Independent of the noisy
    update u and the diffusion flow-time s, so its output can be cached across
    all reverse-diffusion steps of one propagation step.
    """

    def __init__(self, node_dim=24, edge_dim=13, hidden=128, layers=4,
                 tau_emb_dim=16):
        super().__init__()
        self.tau_emb_dim = tau_emb_dim
        self.embed = nn.Linear(node_dim + tau_emb_dim, hidden)
        self.layers = nn.ModuleList(
            [UnionMessageLayer(hidden, edge_dim) for _ in range(layers)]
        )

    def forward(self, node_feats, edge_index, edge_feats, tau, batch):
        batch = batch.to(node_feats.device)
        tau_emb = tau_embedding(tau, dim=self.tau_emb_dim,
                                device=node_feats.device, dtype=node_feats.dtype)
        tau_nodes = tau_emb[batch]                              # [ΣN, tau_dim]
        h = self.embed(torch.cat([node_feats, tau_nodes], dim=-1))
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)
        return h                                               # context [ΣN, H]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cached_propagator.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_model.py tests/test_cached_propagator.py
git commit -m "feat: structural encoder for cached propagator"
```

---

### Task 2: Denoiser

**Files:**
- Modify: `lsmd/transfer_model.py` (append `Denoiser`)
- Test: `tests/test_cached_propagator.py` (append)

**Interfaces:**
- Consumes: `UnionMessageLayer`, `StructuralEncoder` output (Task 1).
- Produces:
  - `Denoiser(hidden=128, edge_dim=13, point_dim=6, n_denoise_layers=1)` with
    `forward(u[ΣN,point_dim], s[G], context[ΣN,hidden], edge_index[2,E], edge_feats[E,edge_dim], batch[ΣN]) -> [ΣN, point_dim]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cached_propagator.py  (append)
def test_denoiser_output_shape_default_one_layer():
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    enc = tm.StructuralEncoder(hidden=32, layers=2)
    ctx = enc(u["node_feats"], u["edge_index"], u["edge_feats"], u["tau"], u["batch"])
    den = tm.Denoiser(hidden=32, n_denoise_layers=1)
    s = torch.rand(2)
    out = den(u["u_target"], s, ctx, u["edge_index"], u["edge_feats"], u["batch"])
    assert out.shape == (13, 6)


def test_denoiser_zero_layers_is_pure_mlp():
    u = batching.union_collate([_toy_graph(6)])
    enc = tm.StructuralEncoder(hidden=32, layers=2)
    ctx = enc(u["node_feats"], u["edge_index"], u["edge_feats"], u["tau"], u["batch"])
    den = tm.Denoiser(hidden=32, n_denoise_layers=0)
    out = den(u["u_target"], torch.rand(1), ctx, u["edge_index"],
              u["edge_feats"], u["batch"])
    assert out.shape == (6, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cached_propagator.py::test_denoiser_output_shape_default_one_layer -v`
Expected: FAIL with `AttributeError: module 'lsmd.transfer_model' has no attribute 'Denoiser'`

- [ ] **Step 3: Append implementation to `lsmd/transfer_model.py`**

```python
class Denoiser(nn.Module):
    """Lightweight per-step epsilon predictor over cached context.

    Injects the noisy update u and diffusion flow-time s into the cached
    structural context, runs n_denoise_layers message layers (default 1; 0 = a
    pure per-node MLP), and predicts epsilon.
    """

    def __init__(self, hidden=128, edge_dim=13, point_dim=6, n_denoise_layers=1):
        super().__init__()
        self.point_dim = point_dim
        self.inject = nn.Linear(hidden + point_dim + 1, hidden)   # context + u + s
        self.layers = nn.ModuleList(
            [UnionMessageLayer(hidden, edge_dim) for _ in range(n_denoise_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, point_dim),
        )

    def forward(self, u, s, context, edge_index, edge_feats, batch):
        batch = batch.to(context.device)
        s = torch.as_tensor(s, dtype=u.dtype, device=u.device)
        s_nodes = s[batch].unsqueeze(-1)                         # [ΣN,1]
        h = self.inject(torch.cat([context, u, s_nodes], dim=-1))
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)
        return self.head(h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cached_propagator.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_model.py tests/test_cached_propagator.py
git commit -m "feat: lightweight denoiser over cached context"
```

---

### Task 3: CachedPropagator (drop-in for the union loss)

**Files:**
- Modify: `lsmd/transfer_model.py` (append `CachedPropagator`)
- Test: `tests/test_cached_propagator.py` (append)

**Interfaces:**
- Consumes: `StructuralEncoder` (Task 1), `Denoiser` (Task 2), `ddpm_loss_union` (Plan-1).
- Produces:
  - `CachedPropagator(node_dim=24, edge_dim=13, hidden=128, layers=4, tau_emb_dim=16, point_dim=6, n_denoise_layers=1)` with:
    - `encode(node_feats, edge_index, edge_feats, tau, batch) -> context[ΣN,hidden]`
    - `denoise(u, s, context, edge_index, edge_feats, batch) -> [ΣN,point_dim]`
    - `forward(u, s, node_feats, edge_index, edge_feats, tau, batch) -> [ΣN,point_dim]` (= `denoise(encode(...))`; same signature as `PropagatorNet.forward`).
    - attribute `point_dim`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cached_propagator.py  (append)
from lsmd.model import NoiseSchedule


def test_forward_equals_encode_then_denoise():
    torch.manual_seed(0)
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    net = tm.CachedPropagator(hidden=32, layers=2, n_denoise_layers=1).eval()
    s = torch.rand(2)
    with torch.no_grad():
        direct = net(u["u_target"], s, u["node_feats"], u["edge_index"],
                     u["edge_feats"], u["tau"], u["batch"])
        ctx = net.encode(u["node_feats"], u["edge_index"], u["edge_feats"],
                         u["tau"], u["batch"])
        split = net.denoise(u["u_target"], s, ctx, u["edge_index"],
                            u["edge_feats"], u["batch"])
    assert torch.allclose(direct, split, atol=1e-6)


def test_cached_propagator_is_drop_in_for_union_loss():
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    net = tm.CachedPropagator(hidden=32, layers=2)
    sched = NoiseSchedule(T=50)
    loss = tm.ddpm_loss_union(net, u["u_target"], u["node_feats"],
                              u["edge_index"], u["edge_feats"], u["tau"],
                              u["batch"], sched)
    assert loss.ndim == 0 and torch.isfinite(loss)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cached_propagator.py::test_forward_equals_encode_then_denoise -v`
Expected: FAIL with `AttributeError: module 'lsmd.transfer_model' has no attribute 'CachedPropagator'`

- [ ] **Step 3: Append implementation to `lsmd/transfer_model.py`**

```python
class CachedPropagator(nn.Module):
    """Encoder + denoiser propagator with a cacheable structural pass.

    `forward` has the same signature as PropagatorNet.forward (drop-in for
    ddpm_loss_union). `encode`/`denoise` expose the split so a sampler can run
    the expensive encoder once and the cheap denoiser per reverse step.
    """

    def __init__(self, node_dim=24, edge_dim=13, hidden=128, layers=4,
                 tau_emb_dim=16, point_dim=6, n_denoise_layers=1):
        super().__init__()
        self.point_dim = point_dim
        self.encoder = StructuralEncoder(node_dim, edge_dim, hidden, layers,
                                         tau_emb_dim)
        self.denoiser = Denoiser(hidden, edge_dim, point_dim, n_denoise_layers)

    def encode(self, node_feats, edge_index, edge_feats, tau, batch):
        return self.encoder(node_feats, edge_index, edge_feats, tau, batch)

    def denoise(self, u, s, context, edge_index, edge_feats, batch):
        return self.denoiser(u, s, context, edge_index, edge_feats, batch)

    def forward(self, u, s, node_feats, edge_index, edge_feats, tau, batch):
        context = self.encode(node_feats, edge_index, edge_feats, tau, batch)
        return self.denoise(u, s, context, edge_index, edge_feats, batch)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cached_propagator.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_model.py tests/test_cached_propagator.py
git commit -m "feat: CachedPropagator (encoder+denoiser, drop-in for union loss)"
```

---

### Task 4: Cached sampler + reduced-step profile

**Files:**
- Modify: `lsmd/transfer_model.py` (append `sample_ddpm_union_cached`)
- Test: `tests/test_cached_propagator.py` (append)

**Interfaces:**
- Consumes: `CachedPropagator` (Task 3); mirrors the reverse-step math of `sample_ddpm_union` (Plan-1) but encodes once.
- Produces:
  - `sample_ddpm_union_cached(net, node_feats, edge_index, edge_feats, tau, batch, schedule, steps=50, eta=1.0, sigma_init=1.0) -> [ΣN, point_dim]` — `net` must be a `CachedPropagator`; `eta=0` gives the deterministic reduced-step DDIM profile.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cached_propagator.py  (append)
def test_cached_sampler_shape_and_finite():
    u = batching.union_collate([_toy_graph(7)])
    net = tm.CachedPropagator(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)
    out = tm.sample_ddpm_union_cached(net, u["node_feats"], u["edge_index"],
                                      u["edge_feats"], u["tau"], u["batch"],
                                      sched, steps=5)
    assert out.shape == (7, 6)
    assert torch.isfinite(out).all()


def test_cached_sampler_matches_uncached_reference():
    # Caching the context must not change results vs recomputing it each step.
    u = batching.union_collate([_toy_graph(6)])
    net = tm.CachedPropagator(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)

    torch.manual_seed(123)
    cached = tm.sample_ddpm_union_cached(net, u["node_feats"], u["edge_index"],
                                         u["edge_feats"], u["tau"], u["batch"],
                                         sched, steps=5, eta=0.0)

    # Uncached reference: identical loop but net.forward (recomputes encode each step)
    torch.manual_seed(123)
    import torch as _t
    T = sched.T
    N = u["node_feats"].shape[0]
    uu = _t.randn(N, net.point_dim)
    t_full = _t.round(_t.linspace(T - 1, 0, 6, )).long().clamp(0, T - 1)
    with _t.no_grad():
        for i in range(5):
            t = t_full[i].item(); t_prev = t_full[i + 1].item()
            s = _t.full((u["tau"].shape[0],), t / T)
            eps = net(uu, s, u["node_feats"], u["edge_index"], u["edge_feats"],
                      u["tau"], u["batch"])
            sqrt_ab_t = sched.sqrt_alphas_bar[t]
            sqrt_1mab_t = sched.sqrt_one_minus_alphas_bar[t]
            ab_prev = sched.alphas_bar[t_prev]; ab_t = sched.alphas_bar[t]
            u0 = (uu - sqrt_1mab_t * eps) / sqrt_ab_t.clamp_min(1e-8)
            dir_coeff = (1 - ab_prev).clamp_min(0.0).sqrt()
            uu = ab_prev.sqrt() * u0 + dir_coeff * eps
    assert torch.allclose(cached, uu, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cached_propagator.py::test_cached_sampler_shape_and_finite -v`
Expected: FAIL with `AttributeError: module 'lsmd.transfer_model' has no attribute 'sample_ddpm_union_cached'`

- [ ] **Step 3: Append implementation to `lsmd/transfer_model.py`**

```python
@torch.no_grad()
def sample_ddpm_union_cached(net, node_feats, edge_index, edge_feats, tau, batch,
                             schedule, steps=50, eta=1.0, sigma_init=1.0):
    """Reverse-diffusion sampler that encodes the static graph once.

    `net` must expose `encode`/`denoise` (a CachedPropagator). Identical reverse
    math to sample_ddpm_union; the only difference is the structural context is
    computed once and reused across all reverse steps. eta=0 → deterministic
    DDIM (use with a small `steps` for fast rollout).
    """
    T = schedule.T
    N = node_feats.shape[0]
    device, dtype = node_feats.device, node_feats.dtype
    context = net.encode(node_feats, edge_index, edge_feats, tau, batch)
    u = torch.randn(N, net.point_dim, device=device, dtype=dtype) * sigma_init

    t_full = torch.round(torch.linspace(T - 1, 0, steps + 1, device=device)).long().clamp(0, T - 1)
    for i in range(steps):
        t = t_full[i].item()
        t_prev = t_full[i + 1].item()
        s = torch.full((tau.shape[0],), t / T, dtype=dtype, device=device)
        eps_pred = net.denoise(u, s, context, edge_index, edge_feats, batch)

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

Run: `pytest tests/test_cached_propagator.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/transfer_model.py tests/test_cached_propagator.py
git commit -m "feat: cached union sampler with reduced-step DDIM profile"
```

---

### Task 5: Full-suite green + speedup sanity

**Files:**
- Test: `tests/test_cached_propagator.py` (append a timing-shape sanity check; no library code)

**Interfaces:**
- Consumes: everything above. Confirms the split composes and the cached sampler issues exactly one encode call per propagation step.

- [ ] **Step 1: Write the test**

```python
# tests/test_cached_propagator.py  (append)
def test_cached_sampler_encodes_once_per_propagation_step():
    u = batching.union_collate([_toy_graph(6)])
    net = tm.CachedPropagator(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)

    calls = {"encode": 0, "denoise": 0}
    real_encode, real_denoise = net.encode, net.denoise
    net.encode = lambda *a, **k: (calls.__setitem__("encode", calls["encode"] + 1) or real_encode(*a, **k))
    net.denoise = lambda *a, **k: (calls.__setitem__("denoise", calls["denoise"] + 1) or real_denoise(*a, **k))

    tm.sample_ddpm_union_cached(net, u["node_feats"], u["edge_index"],
                                u["edge_feats"], u["tau"], u["batch"],
                                sched, steps=5)
    assert calls["encode"] == 1          # encoded once
    assert calls["denoise"] == 5         # denoised per reverse step
```

- [ ] **Step 2: Run the new test**

Run: `pytest tests/test_cached_propagator.py::test_cached_sampler_encodes_once_per_propagation_step -v`
Expected: PASS

- [ ] **Step 3: Run the full suite**

Run: `pytest tests/ -q`
Expected: all pass (Plan-1 `PropagatorNet`/`sample_ddpm_union` untouched — Global Constraint: non-destructive).

- [ ] **Step 4: Commit**

```bash
git add tests/test_cached_propagator.py
git commit -m "test: cached sampler encodes once per step; full suite green"
```

---

## Self-Review

**Spec coverage** (against `2026-06-20-transferable-training-system-design.md`, Plan 3 section):
- `StructuralEncoder` (L message layers over static graph, run once) → Task 1. ✓
- `Denoiser` (default 1 message layer; 0 = pure MLP; consumes cached context + u + s) → Task 2. ✓
- `CachedPropagator` (forward = encode∘denoise, drop-in for `ddpm_loss_union`; training reuses the same modules) → Task 3. ✓
- `sample_ddpm_union_cached` (encode once, loop denoiser) + reduced-step DDIM (`eta=0`) → Task 4. ✓
- Equivalence guarantee (caching is a pure speedup) → Task 3 (`forward==encode∘denoise`) + Task 4 (cached==uncached reference). ✓
- `tau` in encoder, `s` in denoiser → Tasks 1–2. ✓
- Non-destructive (PropagatorNet/sampler/loss untouched) → verified by full suite in Task 5. ✓
- Dynamic-graph O(N²) cost left as-is (YAGNI) → no task, per spec. ✓

**Type consistency:** `hidden`/`edge_dim`/`point_dim`/`tau_emb_dim` consistent across `StructuralEncoder`, `Denoiser`, `CachedPropagator`. `encode` returns `[ΣN,hidden]`; `denoise` consumes `context[ΣN,hidden]`; `forward` signature matches `PropagatorNet.forward` exactly (so `ddpm_loss_union` is reused unchanged). `net.point_dim` exists for the sampler.

**Placeholder scan:** No TBD/TODO; every code step is complete and runnable.

---

## Follow-up

The trainer (Plan 2) can swap `PropagatorNet` → `CachedPropagator` with no loss-code change (same `forward` signature); the eval rollout (Plan 2) can swap `sample_ddpm_union` → `sample_ddpm_union_cached`. Both swaps are gated on **not regressing** zero-shot RMSF correlation vs the Plan-2 baseline. After Plan 3 is green, proceed to **Plan 4** (physics-aware, staged C1→C2→C3).
