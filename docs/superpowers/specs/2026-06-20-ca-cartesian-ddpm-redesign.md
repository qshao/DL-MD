# CA-Cartesian DDPM Redesign

**Date:** 2026-06-20
**Status:** Approved (design), pending spec review
**Supersedes:** the SE(3) per-residue frame representation in the demo design and
the Boltzmann redesign spec (`2026-06-20-boltzmann-loss-eval-redesign.md`). The
DDPM loss, multi-lag τ-conditioning, density reweighting, and PCA/recall/novelty
metrics from that spec are **retained**; only the state representation, the
target construction, the decoder, and the torsion-based metric change.

## Motivation

Training on the real WT trajectory (`WT/WT-sol6.trr`, 5001 frames, 200 ps/frame)
exposed two problems:

1. **Periodic boundary corruption (root cause).** The protein is split across
   the periodic box (71.3 Å). Before correction, sequential CA–CA "bonds"
   reached 70 Å (exactly one box length) and CA displacements at τ=1 reached
   69 Å — physically impossible for 200 ps. This corrupted both the SE(3)
   frames and any Cartesian target. `mdtraj.Trajectory.make_molecules_whole()`
   fixes it: CA–CA bond mean drops from 6.95 Å to 3.85 Å, and CA displacement
   at τ=1 drops to mean 0.71 Å / max 2.08 Å.

2. **SE(3) per-residue frame fragility.** The rotation-log target space is
   poorly conditioned (a separate orientation per residue, singular near θ=π)
   and yields no physical reconstruction advantage at the CA level. Even with
   PBC fixed, it complicates training and evaluation.

## Decision

Switch the model state from per-residue SE(3) frames to a **Cartesian point
cloud of CA atoms**, predicting CA displacements with the existing DDPM
machinery. Keep the representation extensible so backbone (N, CA, C, O) or
all-atom can be added later without touching the loss, sampler, or network core.

## Scientific framing: timescale and the two regimes

- **Timescale.** The trajectory save interval is 200 ps/frame — already 10⁵×
  the 2 fs MD integration step. This 200 ps stride (τ=1) is the headline
  result; reaching it is the achievement, not 1 ns or 10 ns. Default lag
  schedule: `taus = [1, 2, 5]` → **200 ps, 400 ps, 1 ns**.
- **Fluctuation vs. transition.** These are differentiated by the multi-lag
  τ-conditioning (chosen: multi-lag only, no extra regime-specific loss
  weighting). Short τ learns within-basin thermal fluctuations; long τ begins
  to capture barrier crossings. Evaluation makes the distinction explicit by
  comparing the **distribution** of per-pair CA displacement magnitude ‖Δ‖ at
  each τ — the bulk is fluctuation, the tail is transition.

## Architecture

### Representation: point cloud of displacements

The system is a set of `P` points, each a 3-vector. Currently `P = N_res`
(one CA per residue). Adding atoms later increases `P` (more graph nodes), not
the per-point dimension (always 3). This is the single extensibility hook.

- State at frame `i`: CA coordinates `X_i ∈ ℝ^{P×3}` after (a) making molecules
  whole and (b) global CA superposition to frame 0.
- Training target for pair `(i, j=i+τ)`: **per-pair Kabsch-aligned displacement**
  (chosen: per-pair Kabsch). Align `X_j` onto `X_i` (rotation + translation that
  minimizes CA-RMSD), then `Δ = X_j^{aligned} − X_i ∈ ℝ^{P×3}`. This isolates
  internal conformational change from whole-protein tumbling.
- Inference: pick a source frame `X_i`, sample `Δ`, output `X_j = X_i + Δ`
  (expressed in the `X_i` frame; global orientation is unobservable and
  irrelevant to conformational dynamics).

### Conditioning and equivariance

- **Static reference graph.** As in the existing SE(3) pipeline, the kNN graph
  and edge features are built once from a reference structure (frame 0's CA
  positions) and shared across the batch — this keeps the batched message
  passing intact. The network learns the displacement distribution `P(Δ | τ)`
  conditioned on the reference graph and the lag embedding; sampled `Δ` are
  applied to the chosen source frame at decode time. Conditioning on the
  *specific* source conformation `X_i` (per-state graph or `X_i` node features)
  is a documented future improvement, deferred to keep this iteration a minimal,
  correct CA port.
- **Edge features (CA-only).** `[rel_pos(3), dist(1)]` in the frame-0 orientation
  → `edge_dim = 4`. There is no per-residue rotation, so the rotation-based
  terms of the SE(3) `edge_features` are dropped.
- **Equivariance.** Global CA superposition to frame 0 canonicalizes orientation
  across the whole dataset, so a plain (non-equivariant) graph net suffices —
  displacements live in the common frame-0 frame. No equivariant architecture
  is introduced.

### Components

| Component | Change |
|---|---|
| `data.py` | Add `make_molecules_whole()`; CA-only superposition (protein CA atoms, not all atoms). Return CA point cloud `X [F, P, 3]` plus residue node attributes. `compute_frame_weights` already operates on CA coordinates — unchanged. |
| `geometry.py` | Add `kabsch(X, Y) → (R, t)` returning the rigid transform aligning `Y` onto `X`. |
| `featurize.py` | Add `ca_displacement(X_i, X_j)` (Kabsch-align `X_j`→`X_i`, return `Δ`). Add CA kNN graph + node features (residue identity / index / chain). SE(3) `relative_update`/`apply_update` retained for the future backbone path but unused by the CA pipeline. |
| `model.py` | Add a `point_dim` parameter to `FlowNet.__init__` (default `6`, preserving the SE(3) path and all existing tests). Store it as `self.point_dim`; use it for the embed input width and head output width. `sample` and `sample_ddpm` read `net.point_dim` instead of the literal `6`. The CA pipeline constructs `FlowNet(point_dim=3)`. DDPM loss, schedule, and message passing are unchanged (already generalize over the last dim). |
| `decoder.py` | CA path: `X_i + Δ` → write CA trace (one CA per residue) as PDB with element C. SE(3) frame→atom decoder retained for the future backbone path. |
| `validation.py` | Drop Ramachandran JS from the CA path (needs N/C — revisit with backbone atoms). Keep PCA-JS and recall/novelty (CA-RMSD based). Add `distance_matrix_js` (CA–CA pairwise distance distribution), `rmsf_profile` (per-residue fluctuation, model vs MD), and `displacement_distribution` (‖Δ‖ histogram per τ, separating fluctuation bulk from transition tail). |
| `demo.py` | Wire the CA pipeline: load → whole+superpose → multi-lag Kabsch displacement targets → DDPM train → sample Δ → reconstruct CA → CA-appropriate metrics. |

## Interfaces (key signatures)

```python
# geometry.py
def kabsch(X, Y):
    """Rigid transform aligning Y onto X (both [P,3] or [B,P,3]).
    Returns (R [3,3] or [B,3,3], t [3] or [B,3]) s.t. X ≈ Y@R.T + t."""

# featurize.py
def ca_displacement(X_i, X_j):
    """Kabsch-align X_j onto X_i, return Δ = X_j_aligned - X_i. [P,3]."""

def ca_graph(X, k):
    """kNN graph + edge features from CA positions X [P,3]. Returns (edge_index, edge_feats)."""

# model.py  (point_dim default 6 = SE(3); CA path passes point_dim=3)
class FlowNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=64, layers=3,
                 tau_emb_dim=16, point_dim=6): ...
# sample_ddpm/sample read net.point_dim → return [K, P, net.point_dim]

# validation.py
# pca_js, ensemble_recall, ensemble_novelty accept EITHER [.,N,4,3] (extract CA)
# OR a CA point cloud [.,P,3] directly (rank-3 input used as-is).
def ca_geometry(ca) -> {"ca_bond_mean", "ca_bond_min", "ca_bond_max", "clash_count"}  # ca [P,3]
def distance_matrix_js(ca_model, ca_md, bins=30) -> float   # ca_* [.,P,3]
def rmsf_profile(ca_model, ca_md) -> {"model": [P], "md": [P], "corr": float}
def displacement_js(disp_model, disp_md, bins=30) -> {"js", "model_mean", "md_mean"}  # disp_* 1-D RMSD magnitudes
```

## Test plan

- **PBC:** after `make_molecules_whole`, all sequential same-chain CA–CA bonds
  ∈ [3.5, 4.2] Å; no CA displacement at τ=1 exceeds a physical bound (~5 Å).
- **Kabsch:** aligning a randomly rotated+translated copy recovers RMSD ≈ 0;
  `kabsch(X, X) ≈ (I, 0)`.
- **ca_displacement:** displacement of a frame with itself ≈ 0; rotation-only
  difference yields ≈ 0 after alignment.
- **model point_dim:** `ddpm_loss` and `sample_ddpm` run with `point_dim=3` and
  produce `[K, P, 3]`; the overfit-constant-target test passes at `point_dim=3`.
- **validation:** `distance_matrix_js` is 0 for identical ensembles and bounded
  in [0,1]; `rmsf_profile` correlation is 1.0 for identical ensembles;
  `displacement_distribution` JS is 0 for identical ensembles.
- **demo smoke:** end-to-end on a tiny synthetic trajectory produces a CA-trace
  PDB per sample and a report with the new metric keys.

## CLI changes

- Remove SE(3)-specific assumptions; `--taus` default becomes `1 2 5`.
- No `--atoms` flag this iteration (YAGNI): CA-only is hardwired. The
  point-cloud representation still permits a clean backbone extension later;
  the flag is added when that path is built.
- Report keys: `ca_geometry` (CA–CA bond stats dict), `pca_js`,
  `pca_var_explained`, `ensemble_recall`, `ensemble_novelty`,
  `distance_matrix_js`, `rmsf_corr`, `displacement_js`, `displacement_model_mean`,
  `displacement_md_mean`, `n_residues`, `n_md_reference`, `taus`, `infer_tau`.

## Limitations

- CA-only cannot produce φ/ψ, so Ramachandran agreement is deferred until
  backbone atoms are added. The point-cloud design makes that a data/featurize
  extension, not a model change.
- Output is a CA trace, not a full backbone; side-chain and backbone-oxygen
  geometry are out of scope for this iteration.
- 200 ps is the hard floor (trajectory save interval); finer strides require a
  more frequently saved trajectory.
