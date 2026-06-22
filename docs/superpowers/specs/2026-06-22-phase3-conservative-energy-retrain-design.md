# Phase 3 — Energy-Parameterized Conservative Retrain + FDT Loss Design

**Status:** Phase 3 of 4 (approved 2026-06-22)

**Date:** 2026-06-22

## Project context

This is Phase 3 of the physics-informed propagator roadmap.

1. Phase 1 ✓ — validation suite + baseline (`v2_256h_90k`)
2. Phase 2 ✓ — inference-only modes A (Noether) + B (Boltzmann reweighting)
3. Phase 3 (this doc) — energy-parameterized conservative retrain + FDT loss
4. Phase 4 — productionize

### What Phase 2 proved (and why Phase 3 is necessary)

Phase 2 added inference-only corrections without touching the weights. The
results set hard limits on what post-hoc correction can achieve:

| Metric | Baseline | Mode A | Target |
|---|---|---|---|
| relax_ratio | 14.1× | 9.86× (−30%) | < 5× |
| fes_js | 0.742 | 0.79 | < 0.5 |
| pop_tv | 0.535 | 0.564 | < 0.35 |

- **Mode A (Noether projection)** plateaued at relax_ratio 9.86. Removing
  spurious COM drift and rigid rotation per step cannot fix a model that
  diffuses ~14× too fast — that bias is in the weights.
- **Mode B (Boltzmann reweighting)** failed completely. Even at
  `kT_reweight=10.0` (17× physical temperature) the effective sample size was
  `n_eff` 1.2–3.6 out of 300 frames — fully degenerate on all 6 proteins. The
  model's generated distribution is *fundamentally* too far from the
  CG-energy Boltzmann distribution to rescue by importance reweighting.

The conclusion: both the thermodynamic bias (`fes_js`, `pop_tv`) and the
kinetic bias (`relax_ratio`) must be fixed **at training time**, not by
post-processing.

## Goal

Fine-tune `v2_256h_90k` so that:

- its **stationary distribution** approaches the MD Boltzmann ensemble
  (`fes_js` < 0.5, `pop_tv` < 0.35), and
- its **relaxation timescale** matches MD (`relax_ratio` < 5),

by adding two soft physics losses on top of the existing DDPM drift — with
**no hand-specified energetic parameter values**, and **without caging the
model**: it must remain free to explore physically-plausible conformations the
training trajectories never visited.

## Global Constraints

- **No hand-specified energy values.** All energetic parameters are *learned*
  from the MD data. Literature values (e.g. the Miyazawa–Jernigan matrix) may
  be used only as *initialization*, never as fixed inputs.
- **Fine-tune, do not retrain from scratch.** Stage 2 initializes from the
  existing `checkpoints/v2_256h_90k.pt`.
- **Exploration is a priority over strict distribution matching.** The energy
  is used as a *soft, mostly one-sided* guide. Genuine novel-state exploration
  remains the job of the DDPM sampler's stochasticity, which is kept intact.
- **Soft losses only.** Conservativeness is encouraged by a training loss, not
  enforced by reparameterizing the network's drift (no `drift = −∇U_θ` rewrite).
- **WT/ data must never be pushed to GitHub.** (Repo is public:
  https://github.com/qshao/DL-MD)

## Architecture: two decoupled stages with a hard gate

```
STAGE 1 — Energy calibration (propagator untouched)
  LearnedCGEnergy U_θ  (cg_energy.py functional form, ~6 learnable scalars)
  fit U_θ to corpus-pooled local CA statistics   →  denoising score-matching
  GATE: short Langevin/MCMC samples from U_θ reproduce the MD FES?
        pass → freeze U_θ (save energy_theta.pt) ;  fail → fallback energy form
        ↓
STAGE 2 — Propagator fine-tune (U_θ frozen, init from v2_256h_90k)
  loss = L_ddpm                                   (unchanged; anchors to data)
       + λ_E · L_energy_match(x_pred, U_θ)         (thermo → fes_js, pop_tv)
       + λ_F · L_FDT(Δt_pred, Σ_MD@τ)              (kinetics → relax_ratio)
  λ_E, λ_F warmed up from 0 ; DDPM stochasticity kept for exploration
        ↓
  validate scripts/validate_physics.py on the 6 held-out proteins
```

**Why a hard gate between stages:** if Stage 1's energy cannot reproduce the
MD free-energy surface, Stage 2 would pull the propagator toward a wrong
target. The gate makes that failure cheap and visible, and it is the natural
decoupling boundary of the pre-fit-then-freeze strategy.

## Why a learned *structured* energy (not a fixed teacher, not a neural head)

- **A fixed teacher (Phase 2's `cg_energy.py` with literature values) is ruled
  out** by both the "no hand-specified values" constraint and the empirical
  Mode B failure (its absolute scale is wrong at every temperature tried).
- **A free neural energy head is the wrong default** because of undersampling
  (see below): in MD-unvisited regions it is undefined / wrongly low and would
  actively mislead the propagator. Held as a documented fallback only.
- **A structured energy (sum of local physical terms) with learnable
  coefficients** is the right inductive bias: it extrapolates *sanely*
  (unphysical configs → high energy by construction) and it can be fit from
  *local, corpus-pooled* statistics that are well-sampled even when each
  protein is stuck in one global basin.

## Addressing trajectory undersampling

MD trajectories cover only a small fraction of the Boltzmann ensemble
(dominant basins over-represented, rare states < 1% of frames, barriers
> ~5 kT never crossed). This is the central risk to Stage 1. The resolution:

1. **Fit a *local*, corpus-pooled energy — not a global FES.** `U_θ` is a sum
   of local terms (contacts, angles, excluded volume). A given contact type
   (e.g. ILE–LEU at 6 Å) is sampled thousands of times across the ATLAS
   corpus even if every protein stays in one global basin. Fitting from local
   statistics pooled over the whole corpus turns a severe global-undersampling
   problem into a well-posed local one. (Miyazawa–Jernigan philosophy.)
2. **Physical functional form → sane extrapolation.** Where the corpus has no
   data (a clash, a snapped bond), the structured terms still return high
   energy by construction.
3. **The real metric risk is the opposite of intuition.** `fes_js` compares
   the model to the *empirical (also-undersampled)* MD ensemble, so the goal
   is to *match* MD coverage, not exceed it. The danger is the energy
   rewarding plausible-but-unvisited regions and pulling the model where MD
   has no density (→ `fes_js` worse). Mitigations: inverse-density weighting
   of the fit, and using the energy *softly* in Stage 2 (below).
4. **Multi-temperature reweighting is deferred.** The repo has multi-T data
   (`mdcath.py`, `repack_mdcath_temps.py`) and could reweight high-T replicas
   to 300 K (MBAR/WHAM) for broader coverage. Decision: **not in Phase 3** —
   the corpus-pooled local fit already mitigates most undersampling; add
   multi-T only if the Stage-1 gate shows poor coverage of known states.

---

## Stage 1 — Energy calibration

### `lsmd/learned_energy.py` (new) — `LearnedCGEnergy(nn.Module)`

Wraps the existing `cg_energy.py` terms, replacing fixed coefficients with a
small learnable set. Parameters are stored in log-space so they stay positive
and are initialized to reproduce the current `cg_energy.py` defaults / M&J
1996 matrix shape:

```
learnable (init from cg_energy defaults):
  log_alpha_mj          # global scale on the MJ contact matrix shape
  log_k_angle           # angle stiffness  (init log 10.0)
  log_wca_eps           # excluded-volume well depth (init log 0.3)
  log_w_mj, log_w_angle, log_w_wca   # per-term weights (init log 1.0)

U_θ(x) = w_mj·α_mj·Σ MJ·contact
       + w_angle·k_angle·Σ(θ − θ0)²
       + w_wca·eps·WCA(x)
```

~6 scalars to start — trivially fittable from CA data and robust to
undersampling. The MJ matrix (`cg_energy.MJ_MATRIX`) is the fixed *shape*;
only the scalar `α_mj` scales it. `theta0`, `mj_cutoff`, `wca_sigma` stay at
their `cg_energy.py` defaults (geometric, not energetic, parameters).

**Fallback (documented; only if the Stage-1 gate fails):** expand the contact
term to a learnable 20×20 matrix (MJ as prior) or a small neural energy head.

**Housekeeping:** relocate `_wca_energy` from `lsmd/transfer_eval.py` into
`lsmd/cg_energy.py` (the Phase 2 code comment already anticipates this:
"private; moved to cg_energy in Phase 3"). Keep a re-export shim in
`transfer_eval.py` so existing `rollout()` WCA guidance keeps working.

### `scripts/fit_energy.py` (new) — fitting procedure

- **Objective: denoising score-matching on corpus-pooled CA frames.** Perturb
  each MD CA frame with Gaussian noise of scale σ; train `U_θ` so that
  `−∇ₓ U_θ(x_noisy) / kT ≈ (x_clean − x_noisy) / σ²` (Vincent 2011). No
  partition function; stable with few parameters. Because `U_θ` is a sum of
  *local* terms its gradient is local → this is the corpus-pooled local fit.
- **Inverse-density weighting:** weight each frame's loss by inverse local
  density in PCA-CV space (the same shared-PCA CVs the validation suite uses,
  `transfer_validate.shared_pca`), so over-represented basins do not dominate.
- Pool across all training proteins/frames at native temperature.
- Output: `checkpoints/energy_theta.pt` (the fitted `LearnedCGEnergy` state).

### Freeze gate (hard go/no-go)

1. **Cheap pre-check — energy–population correlation:** bin MD frames into the
   2-D PCA-CV free-energy surface; mean `U_θ` per bin must anti-correlate with
   empirical bin population (Spearman ρ below a chosen negative threshold).
2. **Real gate — generative FES match:** run short Langevin/MCMC on the frozen
   `U_θ` (CA-space, a few thousand steps) to draw samples; compute
   `fes_js(U_θ-samples, MD)` via `transfer_validate.fes_comparison`. **Pass**
   if below a chosen threshold (target ≲ 0.5, the same scale as the final
   metric) → freeze and save. **Fail** → switch to the fallback energy form
   and refit; do **not** start Stage 2.

A `scripts/fit_energy.py --gate` mode prints both checks and the pass/fail
verdict so the gate is reproducible and recorded.

---

## Stage 2 — Propagator fine-tune

`U_θ` is frozen (loaded from `energy_theta.pt`); the network is initialized
from `checkpoints/v2_256h_90k.pt`.

### Extend `lsmd/physics_loss.py`

Both new losses reuse the exact `u0_hat → de-normalize → apply_update → score`
path already implemented in `ddpm_physics_loss` / `geometric_penalty`.

**`energy_match_loss(R_cur, t_cur, u_denorm, res_type, global_chain, energy, *,
w_hi, w_lo, u_cut)`** — soft, mostly one-sided:

```
x_pred = apply_update(R_cur, t_cur, u_denorm)        # decoded predicted frame
L_em = w_hi · relu(U_θ(x_pred) − u_cut)              # hinge: ONLY unphysical /
                                                     #   high-energy excursions
     + w_lo · weak_boltzmann(U_θ(x_pred), U_θ(x_MD_target))   # gentle reweight
```

- `u_cut` = high-percentile of `U_θ` over MD frames, size-normalized,
  precomputed once per protein/corpus. The hinge is **zero** anywhere at or
  below typical physical-energy levels → novel low-energy basins are free to
  be explored.
- `weak_boltzmann` is the only term that nudges relative basin populations
  toward `exp(−U_θ/kT)` (what actually moves `fes_js`/`pop_tv`). It is kept
  small (`w_lo ≪ w_hi`) so it corrects weights without caging the model.
- `x_MD_target` is the real next frame already present in the training pair
  (the DDPM target), so no extra data is needed.

**`fdt_loss(u_denorm, batch, sigma_md_tau, *, reduction)`** — step-variance
matching at lag τ (the chosen FDT form):

```
Σ_model  = Var(Δt_pred)   over the translational component of u_denorm,
                          per protein (grouped by batch)
Σ_target = Cov(Δx_MD) @ τ  precomputed per shard at the SAME lag τ used in
                          training (Einstein/FDT relation: Σ = 2·D·τ, D = kT/γ)
L_FDT    = ‖Σ_model − Σ_target‖²
```

- The data-anchored target `Σ_target` is computed once per shard from the MD
  trajectory at lag τ. The implied friction `γ = kT·τ / D` is reported as a
  diagnostic. A learnable scalar `γ` is **optional** (off by default — the
  data-anchored target already pins the diffusion scale).
- This directly fixes the diffusion scale that drives `relax_ratio`: if the
  per-step CA displacement variance matches MD's, CV autocorrelations decay at
  the MD rate, so `relax_model_ps ≈ relax_md_ps`.

### Modify `lsmd/transfer_train.py` — `train()`

New keyword arguments: `energy_ckpt=None`, `lam_energy=0.0`, `lam_fdt=0.0`,
`phys_warmup_steps`, plus `w_hi`, `w_lo` for the energy hinge.

```
net = load_checkpoint("checkpoints/v2_256h_90k.pt")     # fine-tune
energy = LearnedCGEnergy.load("energy_theta.pt").eval()  # frozen, no grad

loss = ddpm_physics_loss(net, union, physics, scale, schedule, lam=0.0)  # L_ddpm
     + lambda_schedule(step, phys_warmup_steps, lam_energy)
         · energy_match_loss(..., energy, w_hi=w_hi, w_lo=w_lo, u_cut=u_cut)
     + lambda_schedule(step, phys_warmup_steps, lam_fdt)
         · fdt_loss(u_denorm, batch, sigma_md_tau)
```

- `lambda_schedule` (already in `physics_loss.py`) warms both physics λ from 0
  so the pretrained weights are not shocked.
- `collate_physics` already supplies `R_cur`, `t_cur`, `global_chain`,
  `protein_id`; extend it to also carry `res_type` (for `U_θ`) and the
  precomputed `sigma_md_tau` per protein.
- Output: `checkpoints/v3_<tag>.pt`, with periodic step checkpoints as in the
  existing training script.

---

## Validation & success criteria

Run `scripts/validate_physics.py` on the same 6 held-out proteins used for the
Phase 1 baseline and Phase 2 (`3u7t_A`, `4p3a_B`, `1b2s_F`, `2y4x_B`,
`1z0b_A`, `6ovk_R`).

**Pass criteria:**

| Metric | Baseline | Phase 3 target |
|---|---|---|
| mean relax_ratio | 14.1× | < 5× |
| mean fes_js | 0.742 | < 0.5 |
| mean pop_tv | 0.535 | < 0.35 |

**Plus an exploration check (explore-over-match priority):** any model density
in MD-unvisited PCA-CV regions must be *low-`U_θ`* (physically valid). Novel
density at *high* `U_θ` is the only kind flagged as a regression. This is
recorded alongside the metrics, not gated by a single threshold, because
genuine valid exploration can legitimately keep `fes_js` above zero.

**Comparison:** extend / reuse `scripts/compare_modes.py` to produce a delta
table `baseline → Mode A → Phase 3` so the retrain's effect is visible against
the inference-only modes.

---

## File layout

| File | Change | Stage |
|---|---|---|
| `lsmd/cg_energy.py` | Move `_wca_energy` in from `transfer_eval.py` | 1 |
| `lsmd/transfer_eval.py` | Re-export shim for `_wca_energy` | 1 |
| `lsmd/learned_energy.py` | **New** — `LearnedCGEnergy` module | 1 |
| `scripts/fit_energy.py` | **New** — score-matching fit + `--gate` | 1 |
| `lsmd/physics_loss.py` | Add `energy_match_loss`, `fdt_loss` | 2 |
| `lsmd/transfer_train.py` | Wire energy + FDT losses into `train()`; extend `collate_physics` | 2 |
| `scripts/train_transfer.py` | New CLI flags for energy/FDT/warmup | 2 |
| `scripts/compare_modes.py` | Add Phase 3 column to delta table | validate |

## Non-goals (Phase 3)

- No hard conservative reparameterization (`drift = −∇U_θ`) — soft loss only.
- No multi-temperature / MBAR reweighting (deferred unless Stage-1 gate fails).
- No full 20×20 learnable contact matrix or neural energy head unless the
  Stage-1 gate fails with the ~6-scalar structured form (documented fallback).
- No rollout-in-the-training-loop (the FDT loss is per-step, no multi-step
  rollout — that was the rejected Green-Kubo option).
- No new validation metrics; reuse the Phase 1 suite.

## Decisions log (from brainstorming)

1. **Mechanism:** soft energy-consistency + FDT loss on the existing DDPM
   drift (not hard reparameterization), reusing `physics_loss.py` C1 machinery.
2. **Energy form:** learned *structured* energy (cg_energy functional form,
   learnable coefficients); neural head is a fallback.
3. **Calibration:** decoupled — pre-fit and freeze `U_θ` (Stage 1), then
   fine-tune the propagator (Stage 2), with a hard gate between.
4. **FDT loss:** step-variance matching at lag τ, anchored to the MD one-step
   covariance (cheap, no rollout); learnable γ optional.
5. **Coverage:** corpus-pooled local fit only at native temperature;
   multi-temperature deferred.
6. **Softness:** energy used softly and asymmetrically (hinge-dominant) so the
   model stays free to explore physically-valid novel conformations.
