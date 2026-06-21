# Transferable Protein-Dynamics Propagator — Design

**Status:** Approved design, ready for implementation planning
**Date:** 2026-06-20
**Author:** Q. Shao (with Claude)

---

## Goal

Evolve LSMD from a **single-protein** displacement model into **one model that
learns protein-dynamics physics across many proteins** and generalizes
**zero-shot** to proteins it never saw during training. Given only a new
protein's reference structure and sequence, the model should generate plausible
backbone dynamics with no per-protein retraining.

The model must learn the physics *implicitly*: by conditioning each prediction
on the **current local geometric environment** of every residue, it learns the
transferable mapping "local environment + residue identity + elapsed time →
how this residue moves," rather than memorizing one protein's motions.

## Decisions (fixed)

| Axis | Decision |
|---|---|
| Training corpus | Public mid-scale MD — **ATLAS** (~1.4k proteins, 3×100 ns each) as primary, mixable with in-house trajectories |
| Inference mode | **Zero-shot** on unseen proteins (no fine-tune in v1) |
| Conditioning | **Structure + AA only** — reference/current geometry + 20-AA identity + positional encoding. **No ESM, no MSA.** |
| Compute | **Single GPU** — gradient accumulation over proteins, modest model size |
| Representation | **SE(3) backbone frames** (one frame per residue from N–CA–C) |
| Granularity (v1) | **CA-frame only** (backbone frame per residue); sidechain beads deferred |

## Non-Goals (v1)

- Sidechain beads (2-bead/4-bead) in the transferable model — deferred to a follow-up once CA-frame transfer is validated.
- Per-protein fine-tuning / few-shot adaptation.
- Sequence-language-model (ESM) or MSA conditioning.
- Learned force fields / energy supervision (Approach 3 from brainstorming).
- Equilibrium Boltzmann emulation as a primary objective (Approach 2) — possible later extension.
- Multi-GPU / distributed training.

---

## Background: why the current model does not transfer

The existing pipeline has two architectural properties that block transfer, plus
a representation already in the codebase that fixes most of them.

1. **Marginal, not conditional.** `FlowNet.forward` (`lsmd/model.py`) consumes
   the static frame-0 graph (`node_feats`, `edge_index`, `edge_feats`) + the
   noisy update + `τ`. It **never sees the current conformation Xₜ**. It learns
   `p(Δ | reference, τ)`, a marginal displacement prior — which is why "mimic"
   rollouts drift and require periodic re-anchoring to real MD frames.

2. **Per-protein, non-invariant features.** `res_type` is indexed per trajectory
   (`sorted(set(res_names))` in `lsmd/data.py`), so identities are not comparable
   across proteins. The production graph builder `ca_graph` emits
   `[rel_pos, dist]` in the **frame-0 global orientation** — *not* rotation
   invariant. Batching in `MessageLayer` assumes **one shared graph** across the
   batch (`[B, N, H]`), which cannot mix proteins of different size/topology.

3. **Frame representation already exists but is unused.** `lsmd/featurize.py`
   contains `relative_update`/`apply_update` (SE(3) local rigid updates) and
   `edge_features` (`[rel_pos, dist, rel_R]`, E(3)-invariant), and
   `lsmd/data.py:load_frames` already builds per-residue frames (R, t) from
   N–CA–C. This machinery is the correct, invariant, size-consistent basis for a
   transferable model — it is simply not wired into the generation pipeline.

---

## Architecture

### Per-residue representation: SE(3) frames

Each residue is an oriented frame **(Rᵢ ∈ SO(3), tᵢ ∈ ℝ³)** built from its
backbone N–CA–C atoms (`geometry.build_frames`). The dynamical variable the model
predicts is the **local rigid update** between time t and t+τ, expressed in the
residue's own frame:

```
u_i = [ local_translation(3) , axis_angle_rotation(3) ]      # point_dim = 6
```

via `relative_update(R_t, t_t, R_{t+τ}, t_{t+τ})`, and inverted at rollout by
`apply_update`. This update is **E(3)-invariant by construction** (computed in
the source frame's local coordinates) and **unit-consistent across proteins**
(Å for translation, rad for rotation), independent of protein size or global
orientation — the properties zero-shot transfer requires.

### State-conditional dynamic graph (the core change)

At **every training pair and every rollout step**, the graph is rebuilt from the
**current** frames (Rₜ, tₜ):

- **Edges:** kNN (k≈16) over current CA positions tₜ → `edge_index`.
- **Edge features:** `edge_features(R_t, t_t, edge_index)` →
  `[rel_pos(3), dist(1), rel_R(9)]` = 13-dim, all invariant in the source frame.
- **Node features:** `[ 20-AA one-hot + UNK (21) , sin/cos residue-index PE (2) ,
  chain_id (1) ]`. Fixed global AA vocabulary (canonical 20 + UNK), identical
  indexing for every protein.

The network predicts uₜ conditioned on this current-state graph + τ, yielding a
true Markov propagator `p(Xₜ₊τ | Xₜ)`. Rebuilding kNN each step is cheap for
N ≤ ~500 and is what eliminates the need for re-anchoring on long rollouts.

### Network

Reuse the existing `FlowNet` + `MessageLayer` core (DDPM ε-prediction, cosine
`NoiseSchedule`, multi-τ `tau_embedding`), with two structural edits:

- **`point_dim = 6`** (SE(3) update), `edge_dim = 13`,
  `node_dim = 21 + 2 + 1 = 24`.
- **Disjoint-union batching** (replaces the `[B, N, H]` shared-graph path) — see
  below.

Default size for single-GPU: `hidden = 128`, `layers = 4–6`. Tunable.

### Cross-protein batching (disjoint union)

Replace the batched `[B, N, H]` assumption with a flat `[ΣN, H]` union graph:

- Concatenate G proteins' nodes into one tensor; offset each protein's
  `edge_index` by its node-start so edges stay within-protein.
- A `batch` vector `[ΣN]` maps each node → its protein index.
- Message passing (`scatter_add` over the union `edge_index`) keeps proteins
  independent automatically (no cross-protein edges). Degree normalization and
  aggregation are unchanged in form, just over the flat node axis.
- Per-protein scalars (flow-time s, τ) are stored `[G]` and broadcast to nodes
  via `batch`.
- On single GPU, accumulate gradients over several union-batches per optimizer
  step to reach an effective protein count without exceeding memory.

### Physical-time τ

Lags are expressed in **picoseconds**, not frame counts. Each trajectory stores
`dt` (ps/frame); a training pair (i, j) carries lag `(j − i) · dt`. `tau_embedding`
already uses `log(τ)`, so feeding ps gives a smooth, trajectory-agnostic
embedding and lets ATLAS and in-house data with different strides mix correctly.
Multi-τ training (sampling several physical lags) is retained for time-flexible
inference.

### Update normalization

Compute one **corpus-level standard deviation** for the translation and rotation
components of u across a sample of training pairs; normalize targets to unit scale
for the DDPM prior, de-normalize at sampling. Because units are already
consistent across proteins (Å, rad), a single global scale suffices — no
per-protein rescaling.

---

## Data pipeline (ATLAS)

A per-protein preprocessor produces one shard per protein:

- Download an ATLAS entry (3 replicas × 100 ns all-atom MD) + its reference PDB.
- `make_molecules_whole`, superpose on CA, extract per-residue backbone frames
  (R, t) for all frames, plus sequence → fixed-vocab `res_type`, `chain_id`,
  `res_index`, and per-trajectory `dt`.
- Save `data/atlas/<pdbid>.pt` with `{R, t, res_type, chain_id, res_index, dt,
  seq, n_res}`. Replicas stored together or as sibling shards.

**Splitting is by protein, not by frame:**

- `train` / `val` / `test` are **disjoint protein sets**. The `test` set is the
  zero-shot evaluation set — proteins never seen in any form during training.
- To avoid homology leakage, split so that test proteins share no close
  structural/sequence cluster with train (e.g., CATH-topology- or
  sequence-identity-based clustering; ATLAS ships CATH labels).
- Within each training protein, an optional time-split reserves late frames for
  per-protein validation curves.

---

## Inference (zero-shot rollout)

1. Input: held-out protein's **reference PDB + sequence**.
2. Build initial per-residue frames (R₀, t₀) from the reference backbone; build
   node features from sequence (fixed vocab).
3. Autoregressive loop: rebuild dynamic graph from current frames → sample u via
   reverse diffusion (`sample_ddpm`) → `apply_update` → next frames. Repeat for
   the requested number of steps at the chosen physical τ.
4. No re-anchoring (the propagator is state-conditional). Existing geometry
   validity checks and optional energy minimization carry over unchanged.
5. Reconstruct all-atom / write trajectory using existing decoder/reconstruct
   paths (CA-frame → backbone; sidechain grafting reuses templates only if a
   reference trajectory is supplied).

---

## Evaluation

**Headline metric:** zero-shot **per-residue RMSF-profile correlation** between
generated and reference MD on held-out proteins (Pearson r over residues,
averaged across the test set).

**Supporting metrics (per held-out protein, vs its reference MD):**

- Cα–Cα pairwise distance-distribution Jensen–Shannon divergence.
- Secondary-structure retention (helix/sheet fraction stability over the rollout).
- Radius-of-gyration distribution overlap.
- Ensemble coverage / recall of reference conformational space.
- Geometry validity (bond lengths, clashes) — should pass without re-anchoring.

**Bracketing baselines:**

- **Upper (oracle):** single-protein model trained on that exact protein.
- **Lower:** today's reference-only marginal prior (no state conditioning).
- **Success criterion:** the transferable model lands well above the lower
  baseline and approaches the oracle on held-out RMSF correlation.

---

## Build order (for the implementation plan)

1. **Fixed AA vocabulary + frame wiring** — global vocab module; route the
   loader and featurizer through `load_frames` / `relative_update` /
   `edge_features` / `apply_update`; unit tests on a single protein
   (round-trip `relative_update`↔`apply_update`, invariance of edge features).
2. **Disjoint-union batching** — rewrite `MessageLayer`/`FlowNet` to flat
   `[ΣN, H]` + `batch` vector; equivalence test vs the old shared-graph path on
   one protein; multi-protein batch test (no cross-protein leakage).
3. **State-conditional graph + physical-τ + normalization** — per-pair dynamic
   graph from Xₜ; ps-lag pairs; corpus-level update normalization.
4. **ATLAS data pipeline** — downloader/preprocessor, per-protein shards,
   by-protein train/val/test split with homology-aware clustering.
5. **Cross-protein training loop** — protein sampler, gradient accumulation,
   multi-τ schedule, checkpointing.
6. **Zero-shot inference + evaluation harness** — rollout from reference;
   RMSF-correlation and supporting metrics on the held-out set; baseline
   comparison.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Zero-shot RMSF correlation stays near the lower baseline (no real transfer) | Bracket with oracle/lower baselines from step 6; if flat, increase corpus size and model capacity, revisit conditioning (structural descriptors as the first add, ESM only if necessary). |
| Long rollouts still drift despite state conditioning | Geometry validity gate + optional energy min already exist; add update-magnitude clamping; evaluate stability vs rollout length explicitly. |
| Homology leakage inflates zero-shot scores | Cluster-based by-protein split (CATH topology / sequence identity); report per-cluster held-out results. |
| Single-GPU memory with many proteins per step | Disjoint-union + gradient accumulation; cap per-union node count; mixed precision. |
| Nonstandard residues / missing backbone atoms | UNK vocabulary entry; skip residues lacking N–CA–C (existing loader behavior). |

## Reused vs new

- **Reused as-is:** `geometry.build_frames`, `relative_update`/`apply_update`/
  `edge_features` (featurize), `NoiseSchedule`, `ddpm_loss`/`sample_ddpm` core,
  `tau_embedding`, density-reweighting idea, validity/energy-min and
  reconstruct/decoder paths.
- **New/rewritten:** fixed AA vocabulary module; disjoint-union batching in
  `MessageLayer`/`FlowNet`; state-conditional (dynamic-graph) training and
  rollout loops; physical-τ pairs; corpus-level normalization; ATLAS pipeline;
  by-protein splitting; cross-protein trainer; zero-shot evaluation harness.
