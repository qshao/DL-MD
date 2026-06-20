# Long-Stride Protein MD — Residue-Frame Demo Design

**Date:** 2026-06-20
**Status:** Approved (design phase)
**Source blueprint:** `long_stride_protein_md_world_model_plan.pdf`

## 1. Purpose & goal

Build an end-to-end **proof-of-concept** for long-stride molecular dynamics: given one
protein conformation `x_t`, sample several *chemically valid* conformations many MD steps
ahead (`x_{t+τ}`) using a learned stochastic transition operator.

This is the first milestone of a larger research program (a transferable protein-dynamics
"world model"). The demo deliberately drops the transfer/world-model claim and focuses on a
single protein, but it adopts a **representation chosen so the multi-molecule assembly case is
the same object with more nodes** — nothing in the representation or core model is discarded
when we later study multiple molecules assembling.

**Priorities:** momentum and a visible working result over rigor. Correct foundations for the
representation (so we don't rewrite later) over feature completeness.

## 2. Scope

**In scope:**
- One protein, the user's existing trajectories (a few long runs).
- Backbone only (N, Cα, C, O), residue-frame representation.
- A single fixed stride `τ` (configurable; chosen via autocorrelation inspection).
- Conditional generative transition `x_t → {x_{t+τ}^(1..K)}` via flow matching on frames.
- Rigid all-atom backbone build + light idealization; PDB export.
- Validation harness for geometry, diversity, ensemble overlap; simple baselines.

**Explicitly out of scope (deferred to later phases):**
- Side chains (χ torsions), all-heavy-atom reconstruction.
- Multi-protein transfer / world-model generalization claims.
- Multi-molecule assembly *runs* (but the representation and model are built to extend to it).
- Multi-τ conditioning, VAMP/Koopman kinetic heads, Chapman–Kolmogorov consistency.
- Forces/velocities, explicit solvent, predictor–corrector MCMC acceptance.

## 3. Success criteria (the "win")

From a held-out frame, sample `K` futures that are:
1. **Chemically valid** — intra-residue geometry exact by construction; peptide-bond
   continuity within tolerance; low clash rate; on-Ramachandran; correct chirality.
2. **Diverse** — the `K` samples spread out; not collapsed to one point and not a copy of `x_t`.
3. **Plausible** — generated torsion / Ramachandran / contact-map distributions visibly
   overlap the MD ensemble.

Bonus framing: the stochastic model beats a deterministic predictor (predict-the-mean /
copy-input) and a noise-only control on ensemble metrics.

## 4. State representation

Everything lives in **one shared coordinate system**, so single-chain and multi-chain are the
same object with a different number of nodes.

**Per residue `i` (across all chains):**
- **Frame** `T_i = (R_i, t_i)`: `t_i` = Cα position; `R_i ∈ SO(3)` built by Gram–Schmidt from
  N–Cα–C. Carries both position and orientation (orientation matters at interfaces).
- **Identity features:** residue-type embedding, `chain_id` embedding, residue-index encoding.
  Identical chains share parameters ⇒ permutation symmetry for free.
- **Backbone atoms** (N, Cα, C, O) placed *rigidly* from `T_i` using ideal bond
  lengths/angles (O from ψ). Intra-residue geometry is exact by construction.

**Edges — dynamic interaction graph:**
- k-nearest residues by Cα distance, computed over *all* residues regardless of chain. For the
  demo (one chain) all edges are intra-chain; adding a second chain makes inter-molecule edges
  appear automatically — no new code path.
- Edge features: relative translation in the local frame, relative rotation, sequence
  separation, same-chain flag, soft contact probability.

**Conditioning:** stride `τ` embedding + current-state frames `x_t` (optional short history later).

**Why this representation (assembly rationale):**
- Inter-molecule relative pose is represented identically to intra-molecule geometry —
  docking is just "relative frames between chains."
- We quotient only the single global E(3) of the whole system (via an equivariant network on
  relative features), never per-molecule poses — docking poses stay live dynamical variables.
- Maps onto the blueprint's slow/fast split: relative pose = slow assembly coordinate,
  internal vibration = fast.

## 5. Component architecture

Six small, independently-testable modules.

| Module | Responsibility | Depends on |
|---|---|---|
| `data` | Load trajectory + topology (mdtraj); compute per-residue N/Cα/C/O and frames `T_i`; build `(x_t, x_{t+τ})` pairs; time-ordered train/val split (no leakage) | mdtraj |
| `representation` | Frame construction/featurization; SO(3) ↔ 6D rotation encode/decode; dynamic k-NN graph + edge features; terminal-residue masking | — |
| `model` | Conditional flow-matching network (equivariant graph net emitting per-residue translation+rotation velocity) + ODE sampler that draws K futures | representation |
| `decoder` | Rigid all-atom backbone build from frames; light idealization/relaxation (peptide-bond + clash fix); write PDB | representation |
| `validation` | Geometry metrics, diversity, ensemble-overlap; baselines | decoder |
| `demo` | CLI/notebook: load checkpoint, sample futures from a frame, dump PDBs + metrics + plots | all |

Each module has a clear interface (e.g. `data` emits `[T_frames, N_res, SE(3)]`; `decoder`
takes `[N_res, SE(3)]` → `[N_res, 4, 3]` coords), so modules are built and tested in isolation,
and the `model` can later be swapped for the full multi-molecule version without touching the rest.

## 6. Data flow

```
trajectory + topology
   → per-frame backbone frames {T_i}        [T_frames, N_res, SE(3)]
   → pairs (x_t, x_{t+τ}) at fixed stride τ, time-ordered train/val split
   → conditional flow-matching training
   → sample K future frame-sets from a held-out x_t
   → rigid all-atom build + light idealization
   → PDBs + validation metrics + plots
```

**Data source:** any mdtraj-loadable trajectory + topology. Concrete residue count, frame
count, and stride `τ` are configuration values filled from the user's data; `τ` is selected by
inspecting backbone-torsion / contact autocorrelation so that fast vibration has decayed but
slow structure is retained.

## 7. Generative transition model

Conditional **flow matching on frames**: learn `p(x_{t+τ} | x_t, τ)` by transporting a noise
distribution over frames to the data distribution. Stochastic by design — samples `K` diverse
futures, which is the correct target at long stride.

- **Network:** an SE(3)/E(3)-equivariant graph net (IPA-style frame attention, or an EGNN
  extended to emit per-residue frames). Reads the noisy frames at flow-time `s`, the condition
  frames `x_t`, identity features, and the dynamic graph; predicts a per-residue velocity —
  translation update (equivariant vector) + rotation update (SO(3) tangent).
- **Symmetry:** equivariance handles global E(3) by construction; shared per-type parameters
  give permutation symmetry. Both are exactly what multi-molecule assembly needs.
- **Sampling:** integrate the flow ODE from noise → `K` future frame-sets.
- **Single-GPU sizing:** small hidden width, a few message-passing/attention layers, sparse
  k-NN graph (not dense all-pairs).

**Deliberate simplification to de-risk the demo:** the fully principled version does flow
matching jointly on translations (ℝ³) and rotations (SO(3)); the SO(3) part is the fiddly bit.
The first demo represents rotation in a continuous **6D form** and projects back to SO(3) via
Gram–Schmidt — simpler to implement, sufficient to prove "valid jumps," and swappable for
proper SO(3) flow matching later without changing the data/decoder/validation modules.

## 8. Decoder & all-atom build

- Each residue's backbone (N, Cα, C, O) is placed **rigidly** from its frame `T_i` using ideal
  internal geometry (O from ψ). Intra-residue geometry is therefore exact.
- **Idealization / relaxation (light):** because frames are placed independently, the
  peptide bond `C_i–N_{i+1}` continuity is a soft property. A few cheap steps of geometry
  projection / minimization enforce peptide-bond length/angle and remove clashes. This is the
  blueprint's "relaxation head" in lightweight form.
- Export to PDB for visualization.

## 9. Training objectives (demo subset)

- **Generative (flow-matching) loss** on frames: translation + rotation-tangent velocity
  regression against the conditional flow target.
- **Frame-aware geometry loss (alignment-free):** FAPE-style / relative-frame loss between
  predicted and true `x_{t+τ}`. Alignment-free so no global superposition is needed and it
  generalizes directly to multi-molecule.
- **Physical/continuity penalties:** peptide-bond length/angle, clash, chirality.

Deferred to later phases: kinetic/VAMP loss, Chapman–Kolmogorov consistency, equilibrium
distribution loss, force auxiliary head.

## 10. Validation & baselines

**Geometry (the win):** bond-length / bond-angle distributions, peptide-bond continuity,
clash score, Ramachandran validity, chirality.

**Diversity:** spread of the `K` samples (e.g. pairwise RMSD spread), checked against two
degenerate failure modes — collapse to a single structure, and copying the input `x_t`.

**Ensemble overlap:** generated vs MD torsion histograms, Ramachandran density, contact map,
RMSF.

**Baselines to contextualize / beat:**
- Deterministic predictor (predict the mean future).
- Copy-input (`x_{t+τ} = x_t`).
- Noise-only control (perturb `x_t` with Gaussian noise; should fail ensemble/diversity tests
  in a structured way).

## 11. Multi-molecule extension path (documented, not built now)

The payoff of Path 2 — these are the changes to go from demo to assembly, none of which touch
the representation or the core model:
- Add chains ⇒ more nodes; the k-NN graph spans chains automatically ⇒ inter-molecule edges
  appear with no new code path.
- Per-molecule rigid pose = the collective of that chain's frames (the slow assembly coordinate).
- Add interface / inter-chain contact metrics and permutation-invariant evaluation.
- Later phases layer in the deferred objectives (VAMP/Koopman, CK consistency, equilibrium) and
  side chains for interface chemistry.

## 12. Testing approach

Test-driven, per module:
- `representation`: angle/rotation encode↔decode round-trip; frame construction recovers
  N/Cα/C; graph builder produces expected neighbors.
- `decoder`: rebuild from a real frame's torsions and confirm local geometry / recomputed
  torsions match; idealization reduces peptide-bond / clash violations.
- `model`: flow-matching sanity on a toy distribution; equivariance check (rotating/translating
  the input rotates/translates the output); sampler produces diverse outputs.
- `validation`: metrics return known values on synthetic inputs (perfect copy, pure noise).

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Deterministic collapse (samples identical) | Model a conditional distribution; monitor diversity metric; flow matching not regression |
| Pretty-but-invalid structures | Rigid intra-residue geometry + idealization + clash/chirality penalties |
| SO(3) generative complexity stalls the demo | Start with 6D-rotation simplification; upgrade to true SO(3) flow later |
| Undersampled slow transitions in the data | Demo claims validity + plausibility, not full kinetics; kinetics deferred |
| Frame independence breaks chain continuity | Continuity loss + light relaxation; report peptide-bond violation rate |

## 14. Staged roadmap toward the blueprint

1. **This demo:** single-chain residue-frame flow-matching transition, valid jumps + ensemble
   overlap, on one GPU.
2. Proper SO(3) flow matching; multi-τ conditioning.
3. Kinetic structure: VAMP/Koopman slow-latent head, Chapman–Kolmogorov consistency, implied
   timescales.
4. Side chains + all-heavy-atom reconstruction; equilibrium-distribution losses.
5. Multi-molecule assembly: multiple chains, inter-molecule edges, interface metrics.
6. Transfer / world-model: multiple proteins, family/fold-split evaluation, anti-noise null
   controls, active-learning loop.
