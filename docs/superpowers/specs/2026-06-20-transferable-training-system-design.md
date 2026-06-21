# Transferable Multi-Protein Training System — Design

**Status:** Approved design, ready for implementation planning
**Date:** 2026-06-20
**Author:** Q. Shao (with Claude)
**Builds on:** `2026-06-20-transferable-protein-dynamics-design.md` (the propagator
core, now implemented as Plan 1) and `2026-06-20-transferable-propagator-core.md`
(the implementation plan for that core).

---

## Goal

Turn the implemented propagator **core** into a complete system that **trains one
model across many proteins** and generalizes **zero-shot** to unseen proteins —
with two additional, explicitly-prioritized concerns folded in from the start:

1. **Efficiency** — the model must train on a single GPU and roll out long
   trajectories cheaply. Priority order: **rollout/sampling speed → training
   throughput → memory**.
2. **Physics-awareness** — beyond the symmetry-invariance the core already has
   (E(3)-invariant features), the model should respect protein geometry and,
   progressively, energetics. Built in **three staged levels**: soft geometric
   losses → differentiable energy guidance → learned energy / Boltzmann.

This is **one integrated design** realized as **four sequenced implementation
plans**, each producing a working, measurable artifact.

## Decisions (fixed)

| Axis | Decision |
|---|---|
| Relationship to prior "Plan 2" | This spec **supersedes** the three-bullet Plan-2 sketch at the end of the core plan; it folds data + trainer + eval together with the efficiency and physics work. |
| Build order | **Plan 2 (baseline) → Plan 3 (fast rollout) → Plan 4 (physics, staged C1→C2→C3).** |
| Efficiency priority | rollout speed first, then training throughput, then memory. |
| Physics depth | all three levels, **staged**; each gated on the previous working. |
| Denoiser depth (Plan 3) | default **1 message layer** in the denoiser; `n_denoise_layers` is tunable (0 = pure MLP). |
| C1 physics form | **weighted auxiliary loss with λ-annealing**, not a hard constraint. |
| Eval brackets | lower = existing reference-only marginal prior; oracle = single-protein model trained on the test protein. |
| Temperature conditioning | **out of scope (v1)** — ATLAS is single-temperature (300 K); no signal to learn T-dependence. |
| Granularity | **CA-frame only** (per the core spec); sidechain beads deferred. |

## Non-Goals (v1)

- Temperature / multi-T conditioning (no multi-T training data).
- Sidechain beads in the transferable model (CA-frame only).
- ESM / MSA conditioning.
- Multi-GPU / distributed training.
- C3 (learned energy / Boltzmann) **implementation** — designed here, build deferred.

---

## Architecture overview

```
Plan 2 (foundation) ──> Plan 3 (fast rollout) ──> Plan 4 (physics: C1 → C2 → C3)
  data + trainer + eval     encoder/denoiser split     soft loss → energy guidance
  + AMP / grad-accum / cap   + reduced-step DDIM          → learned energy
```

Everything reuses the **Plan-1 core**, untouched:
`lsmd/vocab.py`, `lsmd/featurize.py` (`frame_graph`, `frame_node_features`),
`lsmd/batching.py` (`union_collate`), `lsmd/transfer_model.py`
(`PropagatorNet`, `ddpm_loss_union`, `sample_ddpm_union`), `lsmd/normalize.py`
(`UpdateNorm`), `lsmd/data.py` (`build_training_example`, `physical_lag_pairs`).
The single-protein pipeline (`FlowNet`, `infer.py`, `generate_md.py`,
committed checkpoints) stays working; all new code lives in new modules or new
classes alongside the old ones.

**The baseline number from Plan 2 is the yardstick.** Plans 3 and 4 are each
gated on **not regressing** zero-shot per-residue RMSF-profile correlation while
improving their own axis (speed for Plan 3, geometry validity for Plan 4).

---

## Plan 2 — Baseline cross-protein training system

Four independently-testable units.

### `lsmd/atlas.py` — per-protein preprocessor

Given an ATLAS entry (3 replicas × 100 ns all-atom MD + reference PDB), emit one
shard `data/atlas/<pdbid>.pt`:

```
{ R [F,N,3,3], t [F,N,3], res_type [N], chain_id [N], res_index [N],
  dt (ps/frame), seq (str), n_res (int) }
```

- Frames via `make_molecules_whole` → CA superpose → `geometry.build_frames`
  (reuses the existing `data.load_frames` extraction logic).
- **`res_type` is keyed through `lsmd.vocab.residue_indices` on residue *names*** —
  this closes the cross-cutting gap noted in the core review: `load_frames`
  produces per-protein alphabetical indices, which are **not** comparable across
  proteins; the preprocessor re-keys them to the fixed 21-token vocabulary.
- Residues lacking N–CA–C are skipped (existing loader behavior).
- Replicas stored as sibling shards (`<pdbid>_r0.pt`, …) or stacked; the trainer
  treats each replica as a trajectory.

### `lsmd/splits.py` — homology-aware by-protein split

- Input: list of shard ids + their CATH topology labels (ATLAS ships these).
- Assign **whole CATH clusters** to exactly one of train / val / test, so a test
  protein shares no cluster with any train protein (prevents homology leakage).
- Deterministic and seeded; returns `{train: [...], val: [...], test: [...]}`.
- Test set = the zero-shot evaluation set.

### `scripts/train_transfer.py` — cross-protein trainer

Loop:
1. **Protein/replica sampler** draws a protein and a physical lag τ (multi-τ
   schedule in ps via `physical_lag_pairs`).
2. `build_training_example(frames, i, tau_frames, k)` per drawn protein.
3. `union_collate` into a minibatch with a **ΣN cap**: greedily add proteins
   until the next would exceed `max_union_nodes`; defer it to the next batch.
   (Union batching does not pad, so the cap — not bucketing — is the memory
   lever that prevents an unlucky batch of large proteins from OOM-ing.)
4. `ddpm_loss_union` under **AMP autocast** + `GradScaler`.
5. **Gradient accumulation** over `accum_proteins` union-batches → optimizer step.
6. `UpdateNorm.fit` over a corpus sample **once at startup**; persisted in the
   checkpoint.

Checkpoint = `{ model_state, noise_schedule_cfg, update_norm: UpdateNorm.state_dict(),
n_aa_types, hparams }`.

### `scripts/eval_transfer.py` — zero-shot eval harness

- Rollout from a **held-out reference structure**: build (R₀, t₀) from the
  reference backbone, then autoregress — rebuild dynamic graph each step →
  `sample_ddpm_union` (G=1) → `apply_update` → next frames — for the requested
  steps at the chosen physical τ. No re-anchoring.
- **Headline metric:** `validation.rmsf_profile` correlation (Pearson r over
  residues) between generated and reference MD, averaged over the test set.
- **Supporting:** Cα–Cα pairwise distance-distribution JS (`validation`),
  geometry validity (bonds, clashes), Rg-distribution overlap.
- **Brackets:** lower = reference-only marginal prior (today's single-protein
  model); oracle = a single-protein model trained on that exact test protein.
- **Success:** transferable model lands well above the lower bracket and
  approaches the oracle on held-out RMSF correlation.

---

## Plan 3 — Efficient rollout

**Observation.** Within one propagation step the reverse diffusion runs `steps`
(~20–50) iterations; each calls the network with a fresh `(u, s)` but the same
`node_feats`, `edge_index`, `edge_feats`, `tau`, `batch`. The current
`PropagatorNet` mixes `u` into the node embedding at the first layer, so all `L`
message-passing layers depend on `u` and are recomputed every reverse step.

**Split** (new classes in `lsmd/transfer_model.py`; existing `PropagatorNet`
retained for compatibility):

- **`StructuralEncoder(node_feats, edge_index, edge_feats, tau, batch) →
  context [ΣN, H]`** — the `L` message-passing layers over the **static** graph.
  Run **once per propagation step**.
- **`Denoiser(u, s, context) → eps [ΣN, P]`** — default **1 `UnionMessageLayer`**
  + MLP head, consuming the cached `context`. Run **per reverse step**.
  `n_denoise_layers` is a constructor arg: `0` = pure per-node MLP (max speed,
  drops neighbor coupling during denoising); `≥1` = retains coupling.
- **`sample_ddpm_union_cached(...)`** — encode once, loop only the denoiser.
  Cost: `O(steps·L·E·H²)` → `O(L·E·H²) + O(steps·(1·E + N)·H²)`.

**Training reuses the same two modules** in a single encode→denoise pass, so the
split is *also* the training architecture — there is no second model to keep in
sync. A reduced-step DDIM profile (`eta=0`, fewer steps) is exposed as a sampling
option.

**Equivalence guarantee.** A combined `PropagatorNet`-equivalent wrapper
(encoder + denoiser with the structural layers in the encoder and the rest in the
denoiser) must reproduce a reference single-pass forward to numerical tolerance on
identical weights — this is the key correctness test, ensuring the refactor is a
pure speedup with no behavioral change.

**Dynamic-graph cost** (`torch.cdist` + `topk`, O(N²)) is left as-is for v1;
for the target sizes (N ≤ ~500) on GPU it is not expected to dominate the network
cost. Optimize (radius/cell lists) only if profiling shows it matters (YAGNI).

---

## Plan 4 — Physics-aware training & sampling (staged)

Each stage is its own sub-plan, gated on the previous landing and not regressing
the baseline.

### C1 — soft geometric losses — `lsmd/physics_loss.py`

- Inside the loss, recover the predicted **clean** update x̂₀ (already computable
  from `eps_pred` and the schedule), `apply_update` → predicted next frames →
  geometric observables:
  - **Chain-connectivity:** penalty on consecutive Cα–Cα distance vs ~3.8 Å,
    **per chain** (respect breaks via `chain_id`; no penalty across chain
    boundaries).
  - **Ramachandran prior:** φ/ψ from the N–CA–C frames scored through the
    existing `validation.RamachandranPotential`.
- Added as a **weighted auxiliary term** to `ddpm_loss_union`, with **λ
  annealing** (ramp from 0) so the geometric term never destabilizes the DDPM
  score target early in training.
- Test: a deliberately chain-breaking synthetic update incurs strictly higher
  C1 loss than a chain-preserving one; `λ=0` reproduces the Plan-2/3 loss exactly.

### C2 — differentiable energy guidance — `lsmd/guidance.py`

- During sampling, at each reverse step: estimate x̂₀ → next frames →
  **differentiable energy** (bonds + clashes + Rama, reusing the energy terms
  from `validation.minimize_energy_*`) → gradient w.r.t. `u` → guided step
  `u ← u − γ · ∇_u E`. Physics enforced **during** generation rather than as
  post-hoc L-BFGS.
- `γ` (guidance strength) is tunable; **`γ = 0` recovers Plan 3 exactly** (test
  asserts bitwise/this-tolerance equivalence).
- Reuses the existing differentiable energy machinery; no retraining required to
  add guidance to an already-trained model.

### C3 — learned energy / Boltzmann targeting — **design only**

- Sketch: an energy head (or exploiting the score↔energy relationship) trained
  toward the **equilibrium Boltzmann distribution** via an energy-matching /
  reweighted objective.
- **Highest research risk; build deferred** until C1/C2 land and the baseline
  justifies it. Documented so the architecture leaves room for it (the encoder
  context is a natural place to hang an energy head).

### Memory tier (folded in, not its own plan)

Activation checkpointing on the encoder's message layers and a neighbor-count cap
fold into whichever plan first hits a memory wall (likely Plan 2 at large
`max_union_nodes`, or C2 where the guidance gradient doubles activation memory).

---

## Data flow (end to end)

```
ATLAS entry ──atlas.py──> data/atlas/<pdbid>.pt  (fixed-vocab res_type via vocab)
                                  │
                            splits.py ──> {train, val, test} by CATH cluster
                                  │
              train_transfer.py: sample protein+τ → build_training_example
                  → union_collate (ΣN cap) → encode→denoise → ddpm_loss_union
                  (+C1 aux loss) under AMP + grad-accum → checkpoint
                                  │
              eval_transfer.py (held-out test protein):
                  reference → rollout (graph each step → sample cached
                  [+C2 guidance] → apply_update) → RMSF corr / JS / validity
                  vs lower + oracle brackets
```

---

## Testing strategy

- Every new module is unit-tested on **synthetic frames + the existing WT
  trajectory** — the *machinery* is testable without an ATLAS download; a tiny
  fixture shard stands in for the corpus.
- **Plan 2:** preprocessor produces a well-formed shard with fixed-vocab
  `res_type`; split is disjoint and homology-respecting on a toy cluster map;
  trainer takes one optimizer step on a 2-protein fixture without NaNs; eval
  harness produces a finite RMSF correlation on a short synthetic rollout.
- **Plan 3:** **encoder+denoiser equivalence** to a single-pass reference
  forward on shared weights (the central test); cached sampler returns correct
  shapes; measured per-step speedup recorded.
- **Plan 4:** C1 chain-break vs chain-preserve ordering + `λ=0` equivalence;
  C2 `γ=0` equivalence + a guided step strictly lowers energy on a clashing
  fixture.
- The **full existing suite stays green** (non-destructive constraint), as in
  Plan 1.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Zero-shot RMSF correlation near the lower bracket (no real transfer) | Brackets quantify it from Plan 2; if flat, grow corpus/capacity, revisit conditioning (structural descriptors before ESM). |
| Encoder/denoiser split changes outputs (subtle bug) | Equivalence test against single-pass reference on identical weights is a hard gate for Plan 3. |
| Pure-MLP denoiser (`n_denoise_layers=0`) loses too much accuracy | Default is **1 message layer**; `0` is opt-in and compared against the default on RMSF correlation. |
| C1 geometric term destabilizes DDPM training | λ-annealing from 0; `λ=0` equivalence test; monitor score-loss vs geom-loss separately. |
| C2 guidance gradient doubles activation memory | Memory tier (activation checkpointing) folds in here; `γ` tunable down. |
| Homology leakage inflates zero-shot scores | CATH-cluster by-protein split; report per-cluster held-out results. |
| ATLAS download/preprocess is slow/large | Per-protein shards are independent and resumable; machinery tested on a fixture shard, not the full corpus. |

## Reused vs new

- **Reused as-is:** the entire Plan-1 core; `validation.py`
  (`rmsf_profile`, distance/PCA JS, `RamachandranPotential`,
  `minimize_energy_*` energy terms); `geometry.build_frames` /
  `data.load_frames` extraction; `NoiseSchedule` / `tau_embedding`.
- **New:** `lsmd/atlas.py`, `lsmd/splits.py`, `scripts/train_transfer.py`,
  `scripts/eval_transfer.py` (Plan 2); `StructuralEncoder` / `Denoiser` /
  `sample_ddpm_union_cached` in `lsmd/transfer_model.py` (Plan 3);
  `lsmd/physics_loss.py`, `lsmd/guidance.py` (Plan 4 C1/C2); C3 energy head
  (designed, deferred).

---

## Build order (for the implementation plans)

1. **Plan 2** — `atlas.py` → `splits.py` → `train_transfer.py` (AMP, grad-accum,
   ΣN cap, `UpdateNorm.fit`, checkpointing) → `eval_transfer.py` (RMSF corr +
   supporting + brackets). *Deliverable: first zero-shot baseline number.*
2. **Plan 3** — `StructuralEncoder` / `Denoiser` / `sample_ddpm_union_cached`,
   equivalence-tested, default 1 denoiser message layer; reduced-step DDIM.
   *Deliverable: faster rollout, RMSF correlation not regressed.*
3. **Plan 4** — C1 soft losses (`physics_loss.py`) → C2 energy guidance
   (`guidance.py`); C3 design only. *Deliverable: improved geometry validity,
   RMSF correlation not regressed.*

---

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
