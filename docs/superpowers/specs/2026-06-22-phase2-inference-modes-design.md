# Phase 2 — Inference-Only Sampling Modes Design

**Status:** Phase 2 of 4 (approved 2026-06-22)

**Date:** 2026-06-22

## Project context

This is Phase 2 of the physics-informed propagator roadmap. Phase 1 built the
validation suite and recorded a baseline for `v2_256h_90k`. The gaps to close:

| Metric | Baseline | Target |
|---|---|---|
| mean relax_ratio | 14.1× (should be ~1) | < 5× |
| mean fes_js | 0.742 (0=identical) | < 0.5 |
| mean pop_tv | 0.532 (0=identical) | < 0.35 |

Phase 2 is **inference-only** — the checkpoint is unchanged. Phase 3 will add
the energy-parameterized conservative retrain.

Full roadmap:
1. Phase 1 ✓ — validation suite + baseline
2. Phase 2 (this doc) — inference-only modes A + B
3. Phase 3 — energy-parameterized conservative retrain + FDT loss
4. Phase 4 — productionize both sampling modes

## Two modes, two different physical properties

**Mode A (kinetics):** Noether momentum projection. Removes the spurious COM
drift and rigid-body rotation that the diffusion model adds each step. Targets
`relax_ratio`. Preserves the time axis — trajectories remain ordered and
kinetic metrics are valid.

**Mode B (thermodynamics):** CG statistical contact energy + Boltzmann
reweighting. Reweights frames from a Mode A trajectory toward the Boltzmann
distribution defined by a Miyazawa–Jernigan statistical contact potential.
Targets `fes_js` and `pop_tv`. **Breaks the time axis** — resampled
trajectories cannot be used for kinetic metrics, which are reported as `null`
in Mode B reports.

The modes are independent and composable: `--noether` enables Mode A alone;
`--noether --reweight` enables Mode A rollout followed by Mode B reweighting.

## Go-model rejected

A Go-model contact energy was considered and rejected: it creates a single
energy minimum at t₀, prevents exploration of non-native conformations, and
is inappropriate for a general accelerated MD tool trained on thermally
diverse trajectories. The Miyazawa–Jernigan statistical potential is used
instead — it is sequence-aware but reference-structure-agnostic.

## Non-goals (Phase 2)

- No model architecture or weight changes.
- No full MD force field (bonds, Lennard-Jones 12-6, electrostatics).
- No TICA/MSM implied timescales (deferred to Phase 3+ if needed).
- No MH rollout wired into the CLI — `mh_rollout()` is provided in
  `lsmd/transfer_modes.py` as a library function but not exposed as a CLI
  mode in Phase 2.

## File layout

### New: `lsmd/noether.py`

Single exported function:

```python
def noether_project(t_old, t_new, chain_id):
    """Remove net linear and angular momentum from a per-step displacement.

    For each chain independently:
      1. Subtract mean displacement (zero net linear momentum).
      2. Solve for angular velocity from inertia tensor; subtract its rotational
         contribution (zero net angular momentum).

    Args:
        t_old:    [N, 3] CA positions before the step.
        t_new:    [N, 3] CA positions after apply_update + bond_constraint.
        chain_id: [N] long, chain assignment.

    Returns:
        [N, 3] corrected CA positions.
    """
```

**Math (per chain c):**

Let `Δᵢ = t_new_i − t_old_i` for residues i in chain c, `nᶜ` the chain size.

Step 1 — zero linear momentum:
```
Δ_com = (1/nᶜ) Σᵢ Δᵢ
Δᵢ ← Δᵢ − Δ_com
```

Step 2 — zero angular momentum:
```
centroid = (1/nᶜ) Σᵢ t_old_i
rᵢ = t_old_i − centroid
L  = Σᵢ rᵢ × Δᵢ                       # [3] angular momentum
I  = Σᵢ (|rᵢ|² I₃ − rᵢ rᵢᵀ)           # [3,3] inertia tensor
ω  = pinv(I) L                         # [3] angular velocity (pinv for safety)
Δᵢ ← Δᵢ − ω × rᵢ
```

`t_new = t_old + Δ` (per-chain corrections concatenated back).

**Implementation note:** use `torch.linalg.pinv` rather than `inv` — very small
chains (N=1) or degenerate configurations produce singular inertia tensors.
`pinv` returns a least-squares solution safely.

**Integration point in `rollout()`:** one line after `_apply_bond_constraint`,
before `traj.append`:

```python
t_prev = traj[-1]               # position at start of this step (already appended)
# ... existing apply_update + bond_constraint ...
if noether:
    t = noether_project(t_prev, t, chain_id)
traj.append(t.clone())
```

`t_prev` is `traj[-1]` because frame 0 is appended before the loop.

**New `rollout()` signature change:** add `noether: bool = False` keyword
argument. No other change to the existing signature.

### New: `lsmd/cg_energy.py`

Three pure energy functions plus a combined `total_cg_energy`. All return a
scalar tensor in **kcal/mol** (differentiable w.r.t. `t`). `res_type` values
follow the `lsmd.vocab.CANONICAL` ordering (0–19); UNK (index 20) is excluded
from contact pairs.

**`angle_energy(t, chain_id, k_angle=10.0, theta0=2.094)`**

For each consecutive in-chain triplet (i, i+1, i+2):
```
v1 = t[i]   − t[i+1]
v2 = t[i+2] − t[i+1]
cos_θ = dot(v1, v2) / (|v1| |v2|)   (clamped to [-1+ε, 1-ε])
θ = acos(cos_θ)
V_angle += k_angle * (θ − theta0)²
```

θ₀ = 2.094 rad = 120° is the canonical CG Cα equilibrium angle (literature
consensus for all-Cα models). k_angle = 10.0 kcal/mol/rad².

**`mj_contact_energy(t, res_type, chain_id, cutoff=8.0)`**

Uses the Miyazawa–Jernigan 1996 (J. Mol. Biol. 256:623-644, Table 3) 20×20
symmetric contact energy matrix, stored as a `torch.Tensor` constant in the
module. The matrix is indexed by `lsmd.vocab.CANONICAL` order:

```
ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
```

A residue pair (i, j) contributes to the energy when:
- CA–CA distance `d_ij < cutoff` (default 8.0 Å)
- Sequence separation `|i − j| > 3` (exclude bonded neighbors)
- Neither i nor j is UNK (index 20)

Energy: `V_mj = 0.5 * Σ_{i<j, in_contact} MJ[res_type_i, res_type_j]`
(factor 0.5 because the matrix is symmetric and pairs are counted once).

Note: MJ matrix values are negative for favorable contacts (in kcal/mol after
converting from the original kT units at 300 K; multiply by kT=0.593).

**`total_cg_energy(t, res_type, chain_id, *, wca_sigma=4.5, wca_eps=0.3,
k_angle=10.0, theta0=2.094, mj_cutoff=8.0, w_wca=1.0, w_angle=1.0,
w_mj=1.0)`**

```python
from lsmd.transfer_eval import _wca_energy
return (w_wca   * _wca_energy(t, chain_id, sigma=wca_sigma, eps=wca_eps)
      + w_angle * angle_energy(t, chain_id, k_angle=k_angle, theta0=theta0)
      + w_mj    * mj_contact_energy(t, res_type, chain_id, cutoff=mj_cutoff))
```

### New: `lsmd/transfer_modes.py`

**`reweight_boltzmann(traj, res_type, chain_id, kT, **energy_kwargs)`**

```
traj:      [F, N, 3] CA positions (output of rollout).
res_type:  [N] long residue type indices.
chain_id:  [N] long chain assignment.
kT:        float, in kcal/mol (e.g. 0.593 at 300 K).
**energy_kwargs: passed to total_cg_energy.

Returns: {"weights": Tensor[F], "n_eff": float, "degenerate": bool}
```

Algorithm:
```python
energies = torch.stack([total_cg_energy(traj[i], ...) for i in range(F)])
log_w = -energies / kT
log_w -= log_w.max()
w = torch.exp(log_w); w /= w.sum()
n_eff = w.sum().pow(2) / w.pow(2).sum()
degenerate = n_eff < 0.1 * F
```

**`resample_trajectory(traj, weights, n_samples=500)`**

```python
idx = torch.multinomial(weights, n_samples, replacement=True)
return traj[idx]
```

Used internally by `validate_physics.py` to get a fixed-size resampled
trajectory that can be passed to the existing `tv.validate()` unchanged.

**`mh_rollout(net, schedule, update_norm, R0, t0, res_type, chain_id,
res_index, *, steps, tau_ps, k, diff_steps, eta, temp_K, kT,
noether=True, **energy_kwargs)`**

Wraps `rollout()` step-by-step:
1. Propose `t_prop` by calling one internal step of `rollout()` logic
   (extracted as a helper `_rollout_step()`).
2. Compute `ΔU = U_cg(t_prop) − U_cg(t_current)`.
3. Accept with `min(1, exp(−ΔU/kT))`; if rejected, keep `t_current`.
4. Append accepted frame to trajectory.

Returns `[steps+1, N, 3]` — same shape as `rollout()`. Library function only;
not wired into the CLI in Phase 2.

### Modified: `lsmd/transfer_eval.py`

- Add `noether: bool = False` to `rollout()` signature (keyword-only, after
  existing kwargs).
- Add `import` of `noether_project` from `lsmd.noether` at top of file.
- Add one line inside the loop after `_apply_bond_constraint`:
  ```python
  if noether:
      t = noether_project(traj[-1], t, chain_id)
  ```
  (`traj[-1]` is the position at the start of this step because frame 0 is
  appended before the loop begins, so `traj[-1]` is always one step behind.)
- No other changes.

### Modified: `scripts/validate_physics.py`

New CLI flags:
```
--noether              Pass noether=True to rollout (Mode A).
--reweight             Post-process trajectory with Boltzmann reweighting (Mode B).
--kT_reweight FLOAT    kT for reweighting, kcal/mol (default 0.593).
--w_angle FLOAT        Angle term weight in CG energy (default 1.0).
--w_mj FLOAT           MJ contact term weight (default 1.0).
--w_wca_cg FLOAT       WCA weight in CG energy for reweighting (default 1.0;
                       distinct from --wca_lam used for diffusion guidance).
```

`build_report()` change: after rollout, if `--reweight`:
1. Call `reweight_boltzmann(traj, ...)` → weights + n_eff + degenerate flag.
2. Call `resample_trajectory(traj, weights, n_samples=500)` → resampled.
3. Call `tv.validate(resampled, shard["t"].float(), ...)`.
4. Set kinetic fields (msd_rmse, acf_rmse, relax_model_ps, relax_md_ps,
   relax_ratio) to `null` in the report JSON.
5. Add `"reweight": {"n_eff": float, "degenerate": bool}` to the protein dict.

### New: `scripts/compare_modes.py`

```
python scripts/compare_modes.py baseline.json modeA.json modeB.json
```

Reads two or more report JSONs, prints a delta table:

```
Metric           baseline    Mode A    Mode B    A-vs-base   B-vs-base
relax_ratio        14.1       3.2       null        -77%        n/a
fes_js              0.74      0.73      0.41        -1%         -45%
pop_tv              0.53      0.52      0.28        -2%         -47%
rmsf_corr           0.27      0.31      0.27        +15%        0%
dist_js             0.014     0.014     0.015       0%          +7%
```

No model inference — pure JSON comparison.

## Report schema additions

```json
{
  "mode": "A",
  "proteins": {
    "3u7t_A": {
      "reweight": null,
      "structural": {"rmsf_corr": 0.71, "dist_js": 0.006, "rg_js": 0.04,
                     "ca_bond_mean": 3.83, "clash_count": 0.0},
      "thermodynamic": {"fes_js": 0.62, "fes_rmse_kT": 0.9, "pop_tv": 0.45},
      "kinetic": {"msd_rmse": 1.1, "acf_rmse": 0.07, "relax_model_ps": 4100.0,
                  "relax_md_ps": 3800.0, "relax_ratio": 1.08}
    }
  }
}
```

For Mode B (`--reweight`):
```json
{
  "reweight": {"n_eff": 87.3, "degenerate": false},
  "kinetic": {"msd_rmse": null, "acf_rmse": null, "relax_model_ps": null,
              "relax_md_ps": null, "relax_ratio": null}
}
```

## Testing

### `tests/test_noether.py`

Analytically-known answers:

1. **Pure rigid translation** — Δ is uniform; after projection Δ = 0 (all
   displacement was COM motion).
2. **Pure rigid rotation** — Δ_i = ω × r_i for fixed ω; after projection
   Δ ≈ 0 (all displacement was rotation). Tolerance 1e-5 (float32 pinv noise).
3. **Random update — linear momentum zero** — `assert (delta_after.sum(dim=0).abs() < 1e-5).all()`
4. **Random update — angular momentum zero** — compute L after projection,
   assert `L.norm() < 1e-5`.
5. **Two-chain protein** — chains projected independently; verify each chain's
   COM and angular momentum are zero separately.

### `tests/test_cg_energy.py`

1. **Angle — linear chain at θ₀** — place N CAs along a helix such that all
   triplet angles equal 2.094 rad; assert `angle_energy ≈ 0`.
2. **Angle — bent chain** — force one triplet to 180°; assert energy > 0.
3. **MJ — all-GLY contact** — place two GLY CAs at 6 Å distance, seq_sep > 3;
   assert energy equals `MJ_MATRIX[7,7]` (GLY–GLY entry, index 7).
4. **MJ — beyond cutoff** — place pair at 9 Å; assert energy = 0.
5. **MJ — bonded neighbor excluded** — place pair at 5 Å with seq_sep = 2;
   assert energy = 0.
6. **MJ — UNK excluded** — res_type = [20, 0]; assert energy = 0.
7. **Total CG energy — weights** — verify `total_cg_energy(..., w_mj=0)` equals
   `wca + angle` only (no MJ contribution).
8. **Reweighting — uniform energies** — all U_i equal → uniform weights →
   N_eff = F.
9. **Reweighting — one dominant frame** — U_0 = −1000, rest 0 → w_0 ≈ 1,
   degenerate = True (N_eff < 0.1 * F).

## Risks and open points

- **MJ unit conversion.** The original MJ matrix is in kT units at 298 K.
  Multiply by 0.592 kcal/mol (kT at 298 K) to obtain kcal/mol before storing
  the constant tensor, so the energy is consistent with WCA and angle terms
  which are already in kcal/mol.

- **N_eff degeneracy at long rollouts.** With 300 steps at τ=2 ns = 600 ns
  equivalent, many frames will have very high MJ energy (unphysical contacts),
  concentrating weight on a few frames. The `degenerate` flag alerts users to
  reduce `--steps` or increase `--kT_reweight`. The report warns explicitly
  when `n_eff < 0.1 * F`.

- **relax_ratio outlier (1b2s_F at 58×).** This is a multi-chain protein with
  unusual dynamics in the baseline. Noether projection may reduce it somewhat
  (COM drift removed) but the outlier likely has a different root cause
  (interface dynamics, large-scale motion). Do not over-interpret if it
  remains high after Mode A.

- **Mode A does not change thermodynamics by design.** `fes_js` and `pop_tv`
  should be essentially unchanged by `--noether`. Any change (< 2%) is
  numerical noise. If they change substantially, investigate the integration
  point (Noether should be identity for a drifted-but-otherwise-correct
  trajectory in the long-run average).

- **`_wca_energy` import coupling.** `total_cg_energy` imports `_wca_energy`
  from `lsmd.transfer_eval` (a private symbol). This is acceptable for Phase 2;
  Phase 3 will move all CG energy terms into `cg_energy.py` when the energy
  parameterization is added.
