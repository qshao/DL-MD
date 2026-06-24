# Next Steps: Improving the SE(3) PropagatorNet

*Last updated: 2026-06-24. Based on v4 per-protein fine-tune results.*

## Constraint: Physics-Based Losses Are Not Ready

The Phase 3 physics-informed losses — FDT loss, score-matching CG energy, and
fluctuation-dissipation kinetic constraints — are implemented in `lsmd/` but
should not be incorporated into training yet. Prior explorations showed that the
current ATLAS dataset (6 proteins × 1001 frames × 1 replica) is too small and
insufficiently diverse to reliably fit a CG energy model or constrain kinetics
via these relations. Applying them risks overfitting the energy model to
trajectory noise rather than genuine physical signal.

All near-term improvements should be **data- and architecture-driven**.

---

## Prioritized Next Steps

### 1. Use All Three ATLAS Replicas per Protein

Each ATLAS entry provides 3 independent MD replicas. The current pipeline uses
only 1 replica per protein shard. Using all 3 triples the training frames
(3003 vs 1001 per protein), reduces overfitting to replica-specific
conformations, and better covers rare states.

**Implementation**: concatenate the 3 replica trajectories into a single shard,
or pass `--shard` three times in `train_transfer.py` (which already supports
multiple shards via its dataloader). No architectural changes needed.

**Expected impact**: moderate-to-large across all proteins, especially those
with high FES JS divergence (2y4x_B, 6ovk_R).

---

### 2. Longer Per-Protein Fine-Tune for Large Proteins

The uniform 5k-step budget was insufficient for the two largest proteins:

| Protein | Residues | Best RMSF corr | Steps |
|---------|----------|---------------|-------|
| 1z0b_A  | 207      | 0.983         | 5k    |
| 6ovk_R  | 219      | 0.715         | 5k    |

6ovk_R in particular shows a large gap from the rest. A targeted 20k-step
fine-tune on 6ovk_R (and optionally 1z0b_A) is likely to bring it in line with
the other proteins.

**Implementation**: re-run `train_transfer.py --shard data/atlas/6ovk_R.pt
--resume checkpoints/v4_longlags.pt --steps 20000 --out checkpoints/v4_6ovk_R_20k.pt`.

**Expected impact**: high for 6ovk_R; 1z0b_A is already at r=0.983 so the
gain there may be small.

---

### 3. Held-Out Generalization Test

Every protein was present in both the v4_longlags universal fine-tune and the
per-protein fine-tunes. The current metrics therefore measure **in-distribution
fit quality**, not generalization. A leave-one-out experiment is the most
scientifically important next step:

1. Train v4_longlags on 5 proteins (exclude 1 completely).
2. Evaluate zero-shot on the held-out protein at τ = 2000 ps.
3. Optionally fine-tune on the held-out protein and compare to the 6-protein
   fine-tune to quantify the contribution of per-protein adaptation.

This experiment directly tests whether the model has learned **transferable
protein dynamics** or merely memorized the fine-tuning trajectories.

---

### 4. Architecture Scaling

The current model (hidden=256, 6 layers) is conservative. Scaling the hidden
dimension to 512 while keeping 6 layers doubles parameter count (~4× compute
per step) and may improve the capacity to capture complex multi-domain motions
in larger proteins. This should be validated as a controlled experiment:

1. Train `v4_longlags_h512.pt` from `v2_256h_90k.pt` with `--hidden 512
   --layers 6` on all 6 ATLAS proteins (same 20k steps, same lag set).
2. Compare longlags validation against the h=256 baseline before committing to
   per-protein fine-tunes at the larger size.

Do not scale both hidden dim and layers simultaneously — it becomes hard to
attribute any improvement.

---

### 5. Longer Inference Rollout for Kinetic Assessment

The current validation uses 300 rollout steps × τ = 2000 ps = **600 ns** of
simulated time. For larger proteins (1z0b_A, 6ovk_R) whose characteristic
relaxation times may exceed 600 ns, the kinetic metrics (relax_ratio, ACF RMSE)
are computed over too short a window to be reliable. The consistently
sub-unity relax_ratio may partly reflect this measurement truncation rather
than a true model deficiency.

**Implementation**: add a `--steps 1000` flag in `validate_physics.py` for a
dedicated kinetics validation run (separate from the standard 300-step
structural run). This costs ~4× more GPU time per validation but produces
kinetic estimates with lower variance.

---

## Summary Table

| Step | Type | Effort | Expected impact | Risk |
|------|------|--------|----------------|------|
| 1. Multi-replica shards | Data | Low | Medium–High | Low |
| 2. Longer fine-tune (6ovk_R) | Training | Low | High (for 6ovk_R) | Low |
| 3. Leave-one-out generalization | Evaluation | Medium | Scientific validity | Low |
| 4. Hidden dim scaling (h=512) | Architecture | Medium | Medium | Medium |
| 5. Longer rollout for kinetics | Evaluation | Low | Measurement quality | Low |

Steps 1 and 2 can run in parallel and should be attempted first.
Step 3 is the highest scientific priority but requires re-running Phase 1
training from scratch without one protein.
Step 4 is a longer-term investment contingent on Steps 1–3 showing a ceiling.
