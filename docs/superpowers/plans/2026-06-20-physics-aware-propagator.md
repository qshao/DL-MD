# Plan 4 — Physics-Aware Propagator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the transferable propagator respect protein geometry and energetics in **three staged levels** — C1 soft geometric losses (λ-annealed), C2 differentiable energy guidance during sampling, C3 learned-energy design — without destabilizing the DDPM objective.

**Architecture:** A single chain-aware **`geometric_penalty`** decodes a predicted per-residue update onto the current frames and scores Cα–Cα chain connectivity + steric clashes + an optional Ramachandran prior (φ/ψ reconstructed from frames via the idealized backbone). C1 adds `λ · geometric_penalty(x̂₀)` to the union DDPM loss with λ ramped from 0. C2 reuses the *same* penalty as a differentiable energy and nudges the clean-update estimate down its gradient at each reverse step (reconstruction guidance; `γ=0` recovers the plain sampler). C3 is documented, not built.

**Tech Stack:** Python, PyTorch, pytest. Reuses Plan-1 `lsmd/transfer_model.py` (`ddpm_loss_union`, reverse-step math), `lsmd/featurize.py` (`apply_update`), `lsmd/data.py` (`build_training_example`), `lsmd/batching.py` (`union_collate`), `lsmd/decoder.py` (`build_structure`: frames→[N,CA,C,O]), `lsmd/validation.py` (`RamachandranPotential`, and the bond/clash energy *form* from `minimize_energy`).

## Global Constraints

- Core dims unchanged: `point_dim = 6` (update = `[local_trans(3), axis_angle(3)]`), CA position = the frame translation `t`.
- The geometric energy form matches `validation.minimize_energy`: bonds `Σ (|Δ| − 3.8)²`, clashes `Σ max(0, clash_dist − d)²` over non-adjacent CA pairs (`offset≥2`). Default `bond_target = 3.8 Å`, `clash_dist = 3.0 Å`.
- **Chain-aware:** Cα–Cα bonds are penalized only between residues in the **same chain** (via a per-node global chain id); clashes apply to all non-adjacent CA pairs.
- Ramachandran is **optional** (`rama_pot=None` disables it); when present it is a `validation.RamachandranPotential` and φ/ψ come from `decoder.build_structure(R, t)` (returns `[N, 4, 3]` ordered N, CA, C, O — exactly what `RamachandranPotential.energy` consumes).
- C1: `λ = 0` must reproduce the Plan-1 `ddpm_loss_union` value exactly under identical RNG (the geometric term is purely additive). λ is ramped from 0 by `lambda_schedule`.
- C2: `γ = 0` must reproduce the Plan-1 `sample_ddpm_union` output exactly under identical RNG (guidance is a no-op nudge).
- Targets/updates are normalized by `UpdateNorm`; the physics term operates on **de-normalized** updates (multiply the normalized estimate by `scale` before decoding).
- **Non-destructive:** new modules `lsmd/physics_loss.py`, `lsmd/guidance.py`; the only edit to existing code is an **additive** extension of `data.build_training_example` (extra keys). `PropagatorNet`, `ddpm_loss_union`, `sample_ddpm_union`, `validation.py` unchanged. Full suite stays green.
- C3 is **design-only** (documentation), no code.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `lsmd/physics_loss.py` | `geometric_penalty`, `collate_physics`, `ddpm_physics_loss`, `lambda_schedule` | Create |
| `lsmd/guidance.py` | `guidance_step`, `sample_ddpm_union_guided` | Create |
| `lsmd/data.py` | Add `R_cur`, `t_cur`, `chain_id` to `build_training_example` return (additive) | Modify |
| `tests/test_physics_loss.py` | penalty semantics, λ=0 equivalence, physics raises loss | Create |
| `tests/test_guidance.py` | guidance lowers energy, γ=0 equivalence | Create |
| `tests/test_physics_e2e.py` | C1 batch + C2 rollout compose; full suite green | Create |

---

### Task 1: Chain-aware geometric penalty

**Files:**
- Create: `lsmd/physics_loss.py`
- Test: `tests/test_physics_loss.py`

**Interfaces:**
- Consumes: `lsmd.featurize.apply_update`, `lsmd.decoder.build_structure`, `lsmd.validation.RamachandranPotential` (optional).
- Produces:
  - `geometric_penalty(R_cur[ΣN,3,3], t_cur[ΣN,3], u_denorm[ΣN,6], global_chain[ΣN], rama_pot=None, w_bond=1.0, w_clash=1.0, w_rama=0.1, bond_target=3.8, clash_dist=3.0) -> scalar` — `apply_update` → next CA frames → same-chain Cα–Cα bond penalty + non-adjacent clash penalty (+ optional Rama). Differentiable w.r.t. `u_denorm`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_physics_loss.py
import torch
from lsmd import physics_loss as pl
from lsmd import geometry as g


def _frames(n, seed=0):
    torch.manual_seed(seed)
    R = g.so3_exp(torch.randn(n, 3) * 0.1)
    # place CAs on a straight 3.8 Å chain
    t = torch.zeros(n, 3)
    t[:, 0] = torch.arange(n).float() * 3.8
    return R, t


def test_penalty_zero_update_on_ideal_chain_is_small():
    R, t = _frames(6)
    u = torch.zeros(6, 6)                      # identity update → CAs stay ideal
    chain = torch.zeros(6, dtype=torch.long)
    pen = pl.geometric_penalty(R, t, u, chain, w_clash=1.0)
    assert pen.item() < 1e-4


def test_chain_break_update_penalized_more_than_preserving():
    R, t = _frames(6)
    chain = torch.zeros(6, dtype=torch.long)
    keep = torch.zeros(6, 6)
    # push residue 3 far along its local x → stretches bonds 2-3 and 3-4
    breaker = torch.zeros(6, 6); breaker[3, 0] = 10.0
    assert pl.geometric_penalty(R, t, breaker, chain) > \
           pl.geometric_penalty(R, t, keep, chain) + 1.0


def test_no_bond_penalty_across_chain_boundary():
    R, t = _frames(6)
    # two chains 0,0,0 | 1,1,1 — the 2-3 CA gap is a chain break, not a bond
    chain = torch.tensor([0, 0, 0, 1, 1, 1])
    # move the whole second chain far away: only clashes (none) + no bond across break
    u = torch.zeros(6, 6)
    pen = pl.geometric_penalty(R, t, u, chain, w_clash=0.0)
    assert pen.item() < 1e-4


def test_penalty_is_differentiable():
    R, t = _frames(5)
    u = torch.zeros(5, 6, requires_grad=True)
    chain = torch.zeros(5, dtype=torch.long)
    pl.geometric_penalty(R, t, u, chain).backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_physics_loss.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsmd.physics_loss'`

- [ ] **Step 3: Write minimal implementation**

```python
# lsmd/physics_loss.py
"""Physics-aware terms for the transferable propagator (Plan 4).

A single chain-aware geometric penalty decodes a predicted per-residue SE(3)
update onto the current frames and scores Cα–Cα chain connectivity, steric
clashes, and an optional Ramachandran prior. It is shared by C1 (soft training
loss) and C2 (sampling guidance).
"""
import torch

from lsmd import featurize as feat


def geometric_penalty(R_cur, t_cur, u_denorm, global_chain, rama_pot=None,
                      w_bond=1.0, w_clash=1.0, w_rama=0.1,
                      bond_target=3.8, clash_dist=3.0):
    """Geometric energy of the frames obtained by applying u_denorm to (R_cur, t_cur).

    Args:
        R_cur:       [ΣN,3,3] current rotations.
        t_cur:       [ΣN,3] current CA positions.
        u_denorm:    [ΣN,6] de-normalized predicted update.
        global_chain:[ΣN] long, globally-unique chain id (same value ⟺ same
                     protein and same chain; see collate_physics).
        rama_pot:    optional validation.RamachandranPotential.
        w_bond, w_clash, w_rama: term weights.
        bond_target: ideal Cα–Cα distance (Å).
        clash_dist:  minimum non-bonded Cα–Cα distance (Å).

    Returns:
        scalar energy, differentiable w.r.t. u_denorm.
    """
    R_next, t_next = feat.apply_update(R_cur, t_cur, u_denorm)
    ca = t_next                                            # [ΣN,3]

    # same-chain consecutive Cα–Cα bonds
    same = global_chain[1:] == global_chain[:-1]           # [ΣN-1]
    bonds = (ca[1:] - ca[:-1]).norm(dim=-1)                # [ΣN-1]
    if same.any():
        e_bond = ((bonds - bond_target) ** 2)[same].sum()
    else:
        e_bond = ca.new_zeros(())

    # non-adjacent Cα–Cα clashes (all pairs with index gap ≥ 2)
    P = ca.shape[0]
    ii, jj = torch.triu_indices(P, P, offset=2, device=ca.device)
    d = (ca[ii] - ca[jj]).norm(dim=-1)
    e_clash = torch.clamp(clash_dist - d, min=0.0).pow(2).sum()

    e = w_bond * e_bond + w_clash * e_clash

    if rama_pot is not None:
        from lsmd import decoder
        beads = decoder.build_structure(R_next, t_next)    # [ΣN,4,3] (N,CA,C,O)
        e = e + w_rama * rama_pot.energy(beads)
    return e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_physics_loss.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/physics_loss.py tests/test_physics_loss.py
git commit -m "feat: chain-aware geometric penalty for physics-aware propagator"
```

---

### Task 2: C1 soft loss — physics batch + λ-annealed DDPM loss

**Files:**
- Modify: `lsmd/data.py` (additive keys on `build_training_example`)
- Modify: `lsmd/physics_loss.py` (append `collate_physics`, `ddpm_physics_loss`, `lambda_schedule`)
- Test: `tests/test_physics_loss.py` (append)

**Interfaces:**
- Consumes: `lsmd.transfer_model.ddpm_loss_union`, `lsmd.model.NoiseSchedule`, `geometric_penalty` (Task 1); `lsmd.batching.union_collate`.
- Produces:
  - `data.build_training_example(...)` additionally returns `R_cur[N,3,3]` (= `frames["R"][i]`), `t_cur[N,3]` (= `frames["t"][i]`), `chain_id[N]` (= `frames["chain_id"]`).
  - `collate_physics(examples: list[dict]) -> dict` with `R_cur[ΣN,3,3]`, `t_cur[ΣN,3]`, `global_chain[ΣN]` (per-graph offset so chains never merge across proteins: `global_chain = graph_idx * 1000 + chain_id`).
  - `ddpm_physics_loss(net, union, physics, scale, schedule, *, rama_pot=None, lam=0.0, w_bond=1.0, w_clash=1.0, w_rama=0.1) -> scalar` — Plan-1 DDPM score loss **plus** `lam · geometric_penalty(x̂₀·scale)`. `union` is a `union_collate` dict; `physics` is a `collate_physics` dict; `scale` is `UpdateNorm.scale`. With `lam=0` and identical RNG it equals `ddpm_loss_union`.
  - `lambda_schedule(step: int, warmup_steps: int, lam_max: float) -> float` — linear ramp 0→`lam_max` over `warmup_steps`, then constant.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_physics_loss.py  (append)
import random
from lsmd import data, batching
from lsmd import physics_loss as pl
from lsmd import transfer_model as tm
from lsmd.model import NoiseSchedule
from lsmd import geometry as g


def _shard(F=20, N=8, dt=200.0, seed=0):
    torch.manual_seed(seed)
    return {"R": g.so3_exp(torch.randn(F, N, 3) * 0.1),
            "t": torch.randn(F, N, 3) * 5.0,
            "res_type": torch.randint(0, 21, (N,)),
            "chain_id": torch.zeros(N, dtype=torch.long),
            "res_index": torch.arange(N), "dt": dt}


def test_build_training_example_carries_current_frames():
    ex = data.build_training_example(_shard(N=8), i=0, tau_frames=2, k=4)
    assert ex["R_cur"].shape == (8, 3, 3)
    assert ex["t_cur"].shape == (8, 3)
    assert ex["chain_id"].shape == (8,)


def test_collate_physics_offsets_chains_per_graph():
    e0 = data.build_training_example(_shard(N=5, seed=0), 0, 2, 4)
    e1 = data.build_training_example(_shard(N=7, seed=1), 0, 2, 4)
    phys = pl.collate_physics([e0, e1])
    assert phys["R_cur"].shape == (12, 3, 3)
    assert phys["t_cur"].shape == (12, 3)
    # graph 0 chains and graph 1 chains share no value
    g0 = set(phys["global_chain"][:5].tolist())
    g1 = set(phys["global_chain"][5:].tolist())
    assert not (g0 & g1)


def test_lambda_schedule_ramps_then_saturates():
    assert pl.lambda_schedule(0, 100, 0.5) == 0.0
    assert abs(pl.lambda_schedule(50, 100, 0.5) - 0.25) < 1e-6
    assert pl.lambda_schedule(200, 100, 0.5) == 0.5


def test_lam_zero_equals_plain_ddpm_loss():
    e0 = data.build_training_example(_shard(N=5, seed=0), 0, 2, 4)
    e1 = data.build_training_example(_shard(N=7, seed=1), 0, 2, 4)
    union = batching.union_collate([e0, e1])
    phys = pl.collate_physics([e0, e1])
    net = tm.PropagatorNet(hidden=16, layers=2)
    sched = NoiseSchedule(T=30)
    scale = torch.ones(6)

    torch.manual_seed(7)
    plain = tm.ddpm_loss_union(net, union["u_target"], union["node_feats"],
                               union["edge_index"], union["edge_feats"],
                               union["tau"], union["batch"], sched)
    torch.manual_seed(7)
    phys_loss = pl.ddpm_physics_loss(net, union, phys, scale, sched, lam=0.0)
    assert torch.allclose(plain, phys_loss, atol=1e-6)


def test_physics_term_raises_loss_when_lambda_positive():
    e0 = data.build_training_example(_shard(N=6, seed=2), 0, 2, 4)
    union = batching.union_collate([e0])
    phys = pl.collate_physics([e0])
    net = tm.PropagatorNet(hidden=16, layers=2)
    sched = NoiseSchedule(T=30)
    scale = torch.ones(6)

    torch.manual_seed(3)
    base = pl.ddpm_physics_loss(net, union, phys, scale, sched, lam=0.0)
    torch.manual_seed(3)
    with_phys = pl.ddpm_physics_loss(net, union, phys, scale, sched, lam=5.0)
    assert with_phys >= base
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_physics_loss.py::test_build_training_example_carries_current_frames -v`
Expected: FAIL with `KeyError: 'R_cur'`

- [ ] **Step 3a: Extend `data.build_training_example` (additive)**

In `lsmd/data.py`, in `build_training_example`, the return dict currently has keys `node_feats`, `edge_index`, `edge_feats`, `u_target`, `tau`. Add three keys (the current-frame data is already in scope as `R_i`, `t_i`, and `frames["chain_id"]`):

```python
    return {
        "node_feats": node_feats,
        "edge_index": edge_index,
        "edge_feats": edge_feats,
        "u_target": u_target,
        "tau": float(tau_frames) * float(frames["dt"]),
        "R_cur": R_i,
        "t_cur": t_i,
        "chain_id": frames["chain_id"],
    }
```

- [ ] **Step 3b: Append `collate_physics`, `ddpm_physics_loss`, `lambda_schedule` to `lsmd/physics_loss.py`**

```python
from lsmd.transfer_model import _scatter_mean


def collate_physics(examples):
    """Collate current-frame extras for the physics term (mirrors union order).

    Returns R_cur [ΣN,3,3], t_cur [ΣN,3], global_chain [ΣN] where global_chain
    = graph_idx*1000 + chain_id so chains never merge across proteins.
    """
    R_cur, t_cur, chains = [], [], []
    for gi, ex in enumerate(examples):
        R_cur.append(ex["R_cur"])
        t_cur.append(ex["t_cur"])
        chains.append(gi * 1000 + ex["chain_id"].long())
    return {
        "R_cur": torch.cat(R_cur, dim=0),
        "t_cur": torch.cat(t_cur, dim=0),
        "global_chain": torch.cat(chains, dim=0),
    }


def lambda_schedule(step, warmup_steps, lam_max):
    """Linear ramp 0 → lam_max over warmup_steps, then constant lam_max."""
    if warmup_steps <= 0:
        return float(lam_max)
    return float(lam_max) * min(1.0, step / warmup_steps)


def ddpm_physics_loss(net, union, physics, scale, schedule, *, rama_pot=None,
                      lam=0.0, w_bond=1.0, w_clash=1.0, w_rama=0.1):
    """Union DDPM score loss + lam * geometric_penalty on the clean estimate.

    Mirrors transfer_model.ddpm_loss_union's noising (same RNG order), recovers
    the model's clean-update estimate x̂₀, de-normalizes it, and adds the
    chain-aware geometric penalty. lam=0 reproduces ddpm_loss_union exactly.
    """
    u_target = union["u_target"]
    node_feats = union["node_feats"]
    edge_index = union["edge_index"]
    edge_feats = union["edge_feats"]
    tau = union["tau"]
    batch = union["batch"].to(u_target.device)

    G = tau.shape[0]
    T = schedule.T
    t_min = max(1, T // 20)
    t_idx = torch.randint(t_min, T + 1, (G,), device=u_target.device)   # [G]  (1st draw)
    t_nodes = t_idx[batch]
    eps = torch.randn_like(u_target)                                    # (2nd draw)

    sqrt_ab = schedule.sqrt_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    sqrt_1mab = schedule.sqrt_one_minus_alphas_bar[t_nodes].to(u_target.dtype).unsqueeze(-1)
    noisy = sqrt_ab * u_target + sqrt_1mab * eps

    s = (t_idx.float() / T).to(u_target.dtype)
    pred = net(noisy, s, node_feats, edge_index, edge_feats, tau, batch)

    node_se = ((pred - eps) ** 2).mean(dim=-1)
    score_loss = _scatter_mean(node_se, batch, G).mean()

    if lam == 0.0:
        return score_loss

    # model's clean-update estimate, de-normalized, then decoded
    u0_hat = (noisy - sqrt_1mab * pred) / sqrt_ab.clamp_min(1e-8)
    u_denorm = u0_hat * scale.to(u0_hat)
    pen = geometric_penalty(physics["R_cur"].to(u_denorm.device),
                            physics["t_cur"].to(u_denorm.device),
                            u_denorm, physics["global_chain"].to(u_denorm.device),
                            rama_pot=rama_pot, w_bond=w_bond, w_clash=w_clash,
                            w_rama=w_rama)
    return score_loss + lam * pen
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_physics_loss.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/data.py lsmd/physics_loss.py tests/test_physics_loss.py
git commit -m "feat: C1 soft geometric loss with lambda annealing"
```

---

### Task 3: C2 differentiable energy guidance

**Files:**
- Create: `lsmd/guidance.py`
- Test: `tests/test_guidance.py`

**Interfaces:**
- Consumes: `geometric_penalty` (Task 1), `lsmd.transfer_model.sample_ddpm_union` reverse-step math, `lsmd.model.NoiseSchedule`.
- Produces:
  - `guidance_step(u0_hat[ΣN,6], R_cur, t_cur, global_chain, scale, gamma, *, rama_pot=None, **pen_kw) -> [ΣN,6]` — one reconstruction-guidance nudge: `u0_hat − gamma · ∇_{u0_hat} geometric_penalty(u0_hat·scale)`. `gamma=0` returns `u0_hat` unchanged.
  - `sample_ddpm_union_guided(net, node_feats, edge_index, edge_feats, tau, batch, schedule, R_cur, t_cur, global_chain, scale, *, steps=50, eta=1.0, sigma_init=1.0, gamma=0.0, rama_pot=None) -> [ΣN,6]` — Plan-1 reverse loop with a `guidance_step` applied to `u0_hat` each step. `gamma=0` reproduces `sample_ddpm_union` exactly under identical RNG.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_guidance.py
import torch
from lsmd import guidance as gd
from lsmd import transfer_model as tm
from lsmd.model import NoiseSchedule
from lsmd import geometry as g
from lsmd import batching


def _frames(n):
    R = g.so3_exp(torch.zeros(n, 3))
    t = torch.zeros(n, 3); t[:, 0] = torch.arange(n).float() * 3.8
    return R, t


def test_guidance_step_lowers_energy_on_clashing_fixture():
    from lsmd import physics_loss as pl
    R, t = _frames(6)
    chain = torch.zeros(6, dtype=torch.long)
    scale = torch.ones(6)
    # an update that collapses residues together → clashes
    u0 = torch.zeros(6, 6); u0[:, 0] = -1.9 * torch.arange(6).float()
    before = pl.geometric_penalty(R, t, u0 * scale, chain)
    u0_g = gd.guidance_step(u0, R, t, chain, scale, gamma=0.05)
    after = pl.geometric_penalty(R, t, u0_g * scale, chain)
    assert after < before


def test_guidance_step_gamma_zero_is_identity():
    R, t = _frames(5)
    chain = torch.zeros(5, dtype=torch.long)
    u0 = torch.randn(5, 6)
    out = gd.guidance_step(u0, R, t, chain, torch.ones(6), gamma=0.0)
    assert torch.allclose(out, u0)


def test_guided_sampler_gamma_zero_matches_plain():
    n, k = 6, 4
    gr = {"node_feats": torch.randn(n, 24),
          "edge_index": torch.randint(0, n, (2, n * k)),
          "edge_feats": torch.randn(n * k, 13),
          "u_target": torch.randn(n, 6), "tau": 100.0}
    u = batching.union_collate([gr])
    net = tm.PropagatorNet(hidden=16, layers=2).eval()
    sched = NoiseSchedule(T=40)
    R, t = _frames(n)
    chain = torch.zeros(n, dtype=torch.long)

    torch.manual_seed(11)
    plain = tm.sample_ddpm_union(net, u["node_feats"], u["edge_index"],
                                 u["edge_feats"], u["tau"], u["batch"],
                                 sched, steps=5)
    torch.manual_seed(11)
    guided = gd.sample_ddpm_union_guided(net, u["node_feats"], u["edge_index"],
                                         u["edge_feats"], u["tau"], u["batch"],
                                         sched, R, t, chain, torch.ones(6),
                                         steps=5, gamma=0.0)
    assert torch.allclose(plain, guided, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guidance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsmd.guidance'`

- [ ] **Step 3: Write minimal implementation**

```python
# lsmd/guidance.py
"""C2 differentiable energy guidance for the transferable sampler (Plan 4).

Reconstruction guidance: at each reverse step the model's clean-update estimate
x̂₀ is nudged down the gradient of the chain-aware geometric energy, enforcing
physics during generation. gamma=0 recovers the plain sampler exactly.
"""
import torch

from lsmd.physics_loss import geometric_penalty


def guidance_step(u0_hat, R_cur, t_cur, global_chain, scale, gamma, *,
                  rama_pot=None, **pen_kw):
    """One reconstruction-guidance nudge on the clean-update estimate."""
    if gamma == 0.0:
        return u0_hat
    with torch.enable_grad():
        x0 = u0_hat.detach().requires_grad_(True)
        pen = geometric_penalty(R_cur, t_cur, x0 * scale.to(x0), global_chain,
                                rama_pot=rama_pot, **pen_kw)
        grad = torch.autograd.grad(pen, x0)[0]
    return u0_hat - gamma * grad


@torch.no_grad()
def sample_ddpm_union_guided(net, node_feats, edge_index, edge_feats, tau, batch,
                             schedule, R_cur, t_cur, global_chain, scale, *,
                             steps=50, eta=1.0, sigma_init=1.0, gamma=0.0,
                             rama_pot=None):
    """Plan-1 reverse sampler with per-step energy guidance on x̂₀.

    gamma=0 reproduces sample_ddpm_union exactly under identical RNG.
    """
    T = schedule.T
    N = node_feats.shape[0]
    device, dtype = node_feats.device, node_feats.dtype
    u = torch.randn(N, net.point_dim, device=device, dtype=dtype) * sigma_init

    t_full = torch.round(torch.linspace(T - 1, 0, steps + 1, device=device)).long().clamp(0, T - 1)
    for i in range(steps):
        t = t_full[i].item()
        t_prev = t_full[i + 1].item()
        s = torch.full((tau.shape[0],), t / T, dtype=dtype, device=device)
        eps_pred = net(u, s, node_feats, edge_index, edge_feats, tau, batch)

        sqrt_ab_t = schedule.sqrt_alphas_bar[t].to(dtype)
        sqrt_1mab_t = schedule.sqrt_one_minus_alphas_bar[t].to(dtype)
        ab_prev = schedule.alphas_bar[t_prev].to(dtype)
        ab_t = schedule.alphas_bar[t].to(dtype)

        u0_hat = (u - sqrt_1mab_t * eps_pred) / sqrt_ab_t.clamp_min(1e-8)
        u0_hat = guidance_step(u0_hat, R_cur, t_cur, global_chain, scale, gamma,
                               rama_pot=rama_pot)
        pv = (1 - ab_prev) / (1 - ab_t).clamp_min(1e-8) * (1 - ab_t / ab_prev.clamp_min(1e-8))
        sigma_t = eta * pv.clamp_min(0.0).sqrt()
        dir_coeff = (1 - ab_prev - sigma_t ** 2).clamp_min(0.0).sqrt()
        z = torch.randn_like(u) if t_prev > 0 else torch.zeros_like(u)
        u = ab_prev.sqrt() * u0_hat + dir_coeff * eps_pred + sigma_t * z
    return u
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_guidance.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add lsmd/guidance.py tests/test_guidance.py
git commit -m "feat: C2 differentiable energy guidance for sampling"
```

---

### Task 4: End-to-end physics integration + C3 design

**Files:**
- Create: `tests/test_physics_e2e.py`
- Modify: `docs/superpowers/specs/2026-06-20-transferable-training-system-design.md` (append a fleshed-out C3 design subsection)

**Interfaces:**
- Consumes: `ddpm_physics_loss` (C1), `sample_ddpm_union_guided` (C2), `collate_physics`, `geometric_penalty`.

This task adds no new library code; it proves C1 + C2 compose on synthetic data and that guided generation reduces the geometric energy of a rollout step relative to unguided. It also records the C3 (learned-energy / Boltzmann) design.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_physics_e2e.py
import torch
from lsmd import data, batching, geometry as g
from lsmd import physics_loss as pl
from lsmd import guidance as gd
from lsmd import transfer_model as tm
from lsmd.model import NoiseSchedule


def _shard(F=20, N=8, dt=200.0, seed=0):
    torch.manual_seed(seed)
    return {"R": g.so3_exp(torch.randn(F, N, 3) * 0.1),
            "t": torch.randn(F, N, 3) * 5.0,
            "res_type": torch.randint(0, 21, (N,)),
            "chain_id": torch.zeros(N, dtype=torch.long),
            "res_index": torch.arange(N), "dt": dt}


def test_c1_loss_backpropagates_through_net():
    e0 = data.build_training_example(_shard(N=6, seed=0), 0, 2, 4)
    e1 = data.build_training_example(_shard(N=8, seed=1), 0, 2, 4)
    union = batching.union_collate([e0, e1])
    phys = pl.collate_physics([e0, e1])
    net = tm.PropagatorNet(hidden=16, layers=2)
    sched = NoiseSchedule(T=30)
    loss = pl.ddpm_physics_loss(net, union, phys, torch.ones(6), sched, lam=1.0)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g_).all() for g_ in grads)


def test_c2_guidance_reduces_rollout_step_energy():
    # a net is untrained, but guidance must still pull the sampled step toward
    # lower geometric energy than no guidance, on average over seeds.
    n, k = 6, 4
    gr = {"node_feats": torch.randn(n, 24),
          "edge_index": torch.randint(0, n, (2, n * k)),
          "edge_feats": torch.randn(n * k, 13),
          "u_target": torch.randn(n, 6), "tau": 100.0}
    u = batching.union_collate([gr])
    net = tm.PropagatorNet(hidden=16, layers=2).eval()
    sched = NoiseSchedule(T=40)
    R = g.so3_exp(torch.zeros(n, 3))
    t = torch.zeros(n, 3); t[:, 0] = torch.arange(n).float() * 3.8
    chain = torch.zeros(n, dtype=torch.long)
    scale = torch.ones(6)

    def _energy(gamma, seed):
        torch.manual_seed(seed)
        out = gd.sample_ddpm_union_guided(net, u["node_feats"], u["edge_index"],
                                          u["edge_feats"], u["tau"], u["batch"],
                                          sched, R, t, chain, scale,
                                          steps=8, gamma=gamma)
        return pl.geometric_penalty(R, t, out * scale, chain).item()

    plain = sum(_energy(0.0, s) for s in range(5))
    guided = sum(_energy(0.2, s) for s in range(5))
    assert guided < plain
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_physics_e2e.py -v`
Expected: PASS (2 passed). If it errors with an import/attribute error, the named upstream task is incomplete.

- [ ] **Step 3: Append the C3 design subsection**

Append to `docs/superpowers/specs/2026-06-20-transferable-training-system-design.md`, under the Plan 4 section, the following subsection (documentation only — no code in this task):

```markdown
#### C3 — learned energy / Boltzmann targeting (design)

**Aim:** move beyond hand-weighted geometric penalties toward samples that
approach the equilibrium Boltzmann distribution of the reference MD.

**Hook:** the `StructuralEncoder` context (Plan 3) is the natural feature for an
energy head `E_θ(context, u) -> scalar`. Two candidate objectives, to be
bracketed against the C1/C2 baseline before any build:
1. **Energy matching** — regress `E_θ` to the differentiable geometric energy on
   training updates, then use `∇_u E_θ` as a learned guidance field in C2,
   replacing the fixed `geometric_penalty` (cheaper, smoother, learns terms not
   hand-coded).
2. **Boltzmann reweighting** — weight the DDPM score loss by `exp(−E/kT)` density
   ratios estimated from the corpus, nudging the marginal toward equilibrium.

**Risk:** highest of the three stages (training stability, density estimation).
**Gate:** only build if C1+C2 land and the zero-shot RMSF correlation / ensemble
metrics justify the added complexity. No implementation in v1.
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -q`
Expected: all pass (Plan-1, Plan-2, Plan-3, and single-protein tests unaffected — Global Constraint: non-destructive).

- [ ] **Step 5: Commit**

```bash
git add tests/test_physics_e2e.py docs/superpowers/specs/2026-06-20-transferable-training-system-design.md
git commit -m "test: physics C1+C2 end-to-end; document C3 learned-energy design"
```

---

## Self-Review

**Spec coverage** (against `2026-06-20-transferable-training-system-design.md`, Plan 4 section):
- C1 soft geometric losses: Cα–Cα chain-connectivity (per chain) + Ramachandran prior, from the predicted clean update, added as a λ-annealed auxiliary term; `λ=0` reproduces the score loss → Tasks 1–2. ✓
- C2 differentiable energy guidance: per reverse step, nudge x̂₀ down the geometric energy gradient; `γ=0` recovers the plain sampler → Task 3. ✓
- C3 learned energy / Boltzmann: design only, hung on the encoder context → Task 4 Step 3. ✓
- Reuses `RamachandranPotential` (φ/ψ via `decoder.build_structure`) and the `minimize_energy` bond/clash form → Task 1. ✓
- Shared penalty across C1 and C2 (`geometric_penalty`) → Tasks 1–3. ✓
- Non-destructive: new modules + additive `build_training_example` keys; `PropagatorNet`/loss/sampler/`validation.py` untouched → verified by full suite in Task 4. ✓

**Type consistency:** `geometric_penalty(R_cur, t_cur, u_denorm, global_chain, rama_pot, w_bond, w_clash, w_rama, bond_target, clash_dist)` signature is identical where called in C1 (`ddpm_physics_loss`) and C2 (`guidance_step`). `collate_physics` produces `R_cur`/`t_cur`/`global_chain` consumed by `ddpm_physics_loss`; `build_training_example` produces the `R_cur`/`t_cur`/`chain_id` that `collate_physics` consumes. `scale` is `UpdateNorm.scale` (`[6]`) everywhere; `point_dim=6` throughout. `_scatter_mean` is imported from `transfer_model` (Plan-1) and used exactly as in `ddpm_loss_union`, guaranteeing the `λ=0` equivalence.

**Placeholder scan:** No TBD/TODO; every code step is complete and runnable. The C3 subsection is intentionally design-only (the plan's final documented deliverable), not a code stub.

---

## Follow-up

With C1 + C2 green, the trainer (Plan 2) can swap `ddpm_loss_union` → `ddpm_physics_loss` (carrying `collate_physics` extras and a `lambda_schedule`), and the eval rollout can swap `sample_ddpm_union`/`_cached` → `sample_ddpm_union_guided`. Both are gated on **not regressing** zero-shot RMSF correlation while improving geometry validity (lower clash count, fewer bond violations). C3 remains a research option, gated on the C1/C2 results.
