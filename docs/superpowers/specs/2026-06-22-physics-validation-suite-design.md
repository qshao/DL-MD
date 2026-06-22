# Physics Validation Suite + Baseline — Design

**Status:** Phase 1 of 4 (validation-first sequencing approved)

**Date:** 2026-06-22

## Project context (the larger goal)

We want the transferable propagator (`checkpoints/v2_256h_90k.pt`) to produce
trajectories that are **fast to generate**, **structurally physical** (valid
conformations), and **dynamically physical** in two senses the user asked for
as separate modes:

- **Kinetic mode (Mode A):** time-evolution matches MD — relaxation timescales,
  autocorrelations, diffusion behavior.
- **Thermodynamic mode (Mode B):** the visited ensemble is Boltzmann-distributed
  — correct free-energy surfaces and metastable-state populations.

The agreed backbone for later phases is an **energy-parameterized conservative
propagator** (network outputs a scalar energy `E_θ`; score = `−∇E_θ`), which
gives curl-free forces (a Hamiltonian principle) and an explicit energy for the
equilibrium mode. Full project decomposition:

1. **Phase 1 — Validation suite + baseline** (this document)
2. Phase 2 — Inference-only modes (Noether momentum projection; Metropolis /
   reweighting), no retraining
3. Phase 3 — Energy-parameterized conservative retrain + physics losses
4. Phase 4 — Productionize the two sampling modes

Each phase is justified by a gap measured with the Phase 1 suite. **No
retraining happens until Phases 1–2 quantify the residual gap.**

## Phase 1 goal

Build a reusable validation suite that quantifies the kinetic and thermodynamic
physicality of generated trajectories against reference MD, and record baseline
numbers for the current `v2_256h_90k` checkpoint. Everything here is
**inference/analysis only — no training, no model changes.**

## Non-goals (Phase 1)

- No new sampling modes (Phase 2).
- No model architecture or training changes (Phase 3).
- No new MD/analysis dependencies beyond `numpy`, `torch`, `scipy` (already
  transitively available). Specifically **no** `pyemma`/`deeptime`; a full MSM
  is out of scope. TICA-style implied timescales are deferred to a later phase
  if the lightweight metrics prove insufficient.

## Reference-data caveat (must be stated in the report)

There is currently no `split.json`, and all ATLAS shards were seen during
training. Phase 1 therefore measures **fit quality**, not generalization. The
report JSON must carry a `"heldout": false` flag and the protein IDs used, so we
never mistake these baselines for zero-shot generalization numbers. Creating a
proper held-out split is tracked separately and is a prerequisite for any
generalization claim.

## Time-axis alignment (the central correctness point)

Model and MD live on different physical time steps and must be compared on a
shared **physical time axis** before any kinetic metric is computed:

- Model: one rollout step advances physical time by `tau_ps` (the lag, e.g.
  2000 ps). Frame `i` is at time `i * tau_ps`.
- MD: consecutive shard frames are `dt` apart (`shard["dt"]`, e.g. 200 ps for
  ATLAS, 1000 ps for mdCATH). Frame `j` is at time `j * dt`.

All kinetic metrics (MSD, ACF) are returned as `(time_ps, value)` curves, and
comparison happens after interpolating both curves onto a common `time_ps` grid.
Structural and thermodynamic metrics are time-independent (computed over the
pooled ensemble) and need no alignment.

## Module layout

### New: `lsmd/transfer_validate.py`

Pure, individually testable functions. All operate on CA coordinate arrays
`[F, N, 3]` (F frames, N residues) as `torch.Tensor` (float32, CPU).

Structural (consolidate existing `lsmd/validation.py` calls; add Rg):

- `rg_distribution_js(ca_model, ca_md, bins=30) -> float`
  Radius of gyration per frame for each ensemble; JS divergence of the two
  histograms. Rg(frame) = sqrt(mean_i ||x_i − mean(x)||²).

Thermodynamic:

- `shared_pca(ca_ref, n_components=2) -> (mean[N*3], components[n_components, N*3])`
  Fit PCA on a reference ensemble (the MD frames) via `torch.linalg.svd` on the
  mean-centered, flattened `[F, N*3]` matrix. Returns mean and top components to
  define a **shared CV basis** so both ensembles project into the same space.

- `project_cv(ca, mean, components) -> cv[F, n_components]`
  Flatten, center by `mean`, project onto `components`.

- `free_energy_surface(cv, bins=30, kT=1.0, ranges=None) -> (F_grid, edges)`
  2D histogram of CV → probability `P` → free energy `F = −kT * ln(P)` with
  empty bins set to `nan`. `ranges` lets caller pin both ensembles to the same
  grid extents.

- `fes_comparison(cv_model, cv_md, bins=30, kT=1.0) -> dict`
  Build a shared grid from the union of CV ranges; compute both FES; return
  `{"fes_js": <JS of densities>, "fes_rmse_kT": <RMSE over bins well-sampled in
  both ensembles>}`. "Well-sampled" = count ≥ `min_count` (default 5) in both.

- `state_populations(cv_model, cv_md, n_states=6, seed=0) -> dict`
  k-means (simple Lloyd's, fixed seed, 50 iters) on the **MD** CV points to fix
  cluster centers; assign both ensembles to nearest center; return
  `{"pop_model": [...], "pop_md": [...], "pop_tv": <total-variation distance>}`.

Kinetic (all consume a physical timestep and return `(time_ps, value)` arrays):

- `msd_curve(ca, dt_ps, max_lag=None) -> (time_ps[L], msd[L])`
  Mean squared displacement vs lag: for lag `l`, `mean over t,i of
  ||x[t+l,i] − x[t,i]||²`. Caps `max_lag` at `F//2`. CA traces are first
  superimposed to frame 0 (Kabsch) so MSD reflects internal motion, not global
  drift.

- `cv_autocorrelation(cv_1d, dt_ps, max_lag=None) -> (time_ps[L], acf[L])`
  Normalized time autocorrelation of a 1-D CV series:
  `C(l) = mean_t[δq(t) δq(t+l)] / var(q)`, `δq = q − mean(q)`.

- `relaxation_time_ps(time_ps, acf) -> float`
  Integral relaxation time: trapezoidal integral of `acf` up to its first
  zero-crossing (or full range if none), in ps.

Comparison helpers:

- `interp_to_grid(time_ps, value, grid_ps) -> value_on_grid`  (linear interp)
- `curve_rmse(time_a, val_a, time_b, val_b, n=50) -> float`
  Interp both onto a shared grid (the overlapping time range, `n` points),
  return RMSE.

Top-level driver:

- `validate(ca_model, ca_md, *, tau_ps, dt_md_ps, kT=1.0, n_states=6) -> dict`
  Runs the full metric set and returns one nested dict (schema below). Pure:
  takes coordinate tensors, returns numbers. Rollout is the caller's job.

### New: `scripts/validate_physics.py`

CLI that wires checkpoint + shard(s) → rollout → `validate` → JSON report.

Flags (mirror `eval_transfer.py` where they overlap):
`--checkpoint`, `--shard` (repeatable), `--steps`, `--tau_ps`, `--diff_steps`,
`--eta`, `--temp_K`, `--wca_sigma/--wca_eps/--wca_lam`,
`--bond_constraint_iters`, `--max_update_norm`, `--n_states`, `--kT`,
`--out`, `--device`.

Behavior per shard: build `R0,t0` (supports compact `R_aa` and legacy `R`),
roll out with `transfer_eval.rollout`, read `dt_md_ps = shard["dt"]`, call
`validate`, collect into a per-protein dict. Report carries `"heldout": false`
and the list of protein IDs. Writes pretty JSON and prints a summary table.

### New: `tests/test_transfer_validate.py`

Unit tests with analytically known answers (see Testing).

### Baseline artifact

`validation_baseline.json` — full report for `v2_256h_90k` over the 6 proteins
already used (`3u7t_A, 4p3a_B, 1b2s_F, 2y4x_B, 1z0b_A, 6ovk_R`), committed so
later phases diff against it.

## Report JSON schema

```json
{
  "heldout": false,
  "checkpoint": "checkpoints/v2_256h_90k.pt",
  "settings": {"steps": 100, "tau_ps": 2000, "diff_steps": 20, "eta": 1.0,
               "temp_K": 300, "wca_lam": 0.05, "bond_constraint_iters": 5},
  "proteins": {
    "3u7t_A": {
      "n_res": 46,
      "structural":   {"rmsf_corr": 0.71, "dist_js": 0.006,
                       "ca_bond_mean": 3.83, "clash_count": 0.0, "rg_js": 0.04},
      "thermodynamic":{"fes_js": 0.12, "fes_rmse_kT": 0.8,
                       "pop_tv": 0.18},
      "kinetic":      {"msd_rmse": 1.4, "relax_model_ps": 4200.0,
                       "relax_md_ps": 3800.0, "relax_ratio": 1.10,
                       "acf_rmse": 0.09}
    }
  },
  "summary": {"mean_rmsf_corr": 0.43, "mean_dist_js": 0.014,
              "mean_fes_js": 0.15, "mean_relax_ratio": 1.1}
}
```

`relax_ratio = relax_model_ps / relax_md_ps` (1.0 = perfect kinetic match) is
the headline kinetic number; `fes_js` and `pop_tv` are the headline
thermodynamic numbers.

## Testing strategy

Synthetic data with closed-form answers — no model needed:

1. **MSD of a static structure** (all frames identical) → MSD ≈ 0 at all lags.
2. **MSD of pure diffusion** (`x[t] = x[0] + cumsum(gaussian)`) → MSD grows
   ~linearly; assert monotonic increase and slope > 0.
3. **ACF of an Ornstein–Uhlenbeck series** with known correlation time `θ`
   (`q[t+1] = (1−1/θ) q[t] + noise`) → fitted `relaxation_time_ps` within
   tolerance of `θ * dt_ps`.
4. **Free energy of a 2-D Gaussian** ensemble → FES is approximately parabolic;
   `fes_rmse_kT` between two independent draws from the same Gaussian is small
   (< ~0.5 kT in well-sampled bins).
5. **state_populations** on two identical ensembles → `pop_tv ≈ 0`; on disjoint
   clusters → `pop_tv ≈ 1`.
6. **interp_to_grid / curve_rmse**: RMSE of a curve against itself = 0; against
   a constant offset = the offset.
7. **shared_pca / project_cv**: projecting the fitting ensemble reproduces
   variance ordering (component-0 variance ≥ component-1 variance).

All tests are CPU, deterministic (seeded), and run in the existing `pytest`
suite.

## Risks / open points

- **Sampling sufficiency for FES/MSM.** 100 model frames and ~1000 MD frames
  are thin for 2-D free-energy estimation. Mitigation: keep `bins` modest (≤30),
  report only well-sampled-bin RMSE, and treat `fes_js`/`pop_tv` as coarse
  indicators in Phase 1. If they prove too noisy, raise `--steps` for the
  baseline run (cheap at `diff_steps=20`).
- **Relaxation-time estimation on short series.** Integral relaxation time is
  biased when the series is short relative to the relaxation time. We report it
  with the raw ACF curve so the bias is visible, and rely on `relax_ratio`
  (model vs MD on the *same* series length) rather than absolute values.
- **Kabsch in MSD** removes global drift; this is the intended definition
  (internal MSD), consistent with how RMSF is computed.
```
