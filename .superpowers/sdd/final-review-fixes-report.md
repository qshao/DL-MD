# Final Review Fixes — Branch `master`

**Date:** 2026-06-22
**Status:** COMPLETE — all fixes applied, all tests pass

---

## Summary

Three issues identified in the final whole-branch review were fixed.

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | Critical | `lsmd/cg_energy.py` | WCA double-count: divide by 2 in `total_cg_energy` |
| 2 | Critical | `scripts/validate_physics.py` | Null structural/thermodynamic metrics when reweighting is degenerate |
| 3 | Minor    | `lsmd/transfer_modes.py` | Guard against all-zero/negative weights in `resample_trajectory` |

---

## Fix 1 — WCA double-count in `total_cg_energy`

**File:** `lsmd/cg_energy.py`, line 158

`_wca_energy` sums a full symmetric NxN mask (every pair counted twice). `mj_contact_energy` uses an upper-triangle mask (each pair once). The WCA contribution in `total_cg_energy` was 2x too large relative to MJ and angle terms.

**Change:** `w_wca * _wca_energy(...) / 2` — the `/2` is local to `total_cg_energy` only. The standalone `_wca_energy` is unchanged, preserving correct gradient guidance in `rollout()`.

No existing test asserts a WCA-inclusive `total_cg_energy` value (`test_total_cg_energy_w_mj_zero` passes `w_wca=0.0`), so no test expected-values required updating.

**Note:** `validation_modeB.json` (committed in chore/Phase-1 baseline) was computed with the old double-counted WCA and is now stale. It must be re-run before using Mode B numbers.

---

## Fix 2 — Degenerate reweighting silently produces misleading metrics

**File:** `scripts/validate_physics.py`, lines 79-84

When `rw_info["degenerate"]` is True, the resampled trajectory collapses to ~1 frame, making structural (rmsf_corr, dist_js) and thermodynamic (fes_js) metrics meaningless. They were previously reported as real numbers.

**Change:** After nulling kinetic fields, also null structural and thermodynamic fields when `rw_info` exists and `rw_info["degenerate"]` is True. `summarize()` already skips None values, so the Mode B summary will no longer include misleading means from degenerate proteins.

Existing tests in `test_validate_physics_modes.py` do not assert structural values under degenerate reweighting, so no test updates were needed.

---

## Fix 3 — Guard against all-zero weights in `resample_trajectory`

**File:** `lsmd/transfer_modes.py`, lines 49-50

`torch.multinomial` raises a cryptic `RuntimeError` when all weights are zero. Added an explicit check:

```python
if weights.sum() <= 0 or (weights < 0).any():
    raise ValueError("resample_trajectory: weights must be non-negative with positive sum")
```

---

## Test Results

```
pytest tests/test_cg_energy.py tests/test_validate_physics_modes.py -v
→ 20 passed in 6.16s

pytest tests/ -v --tb=short
→ 211 passed, 5 warnings in 62.36s
```

No failures. The 5 warnings are pre-existing (non-contiguous tensor, empty splits, UserWarning in atlas.py).

---

## Commit

See git log for commit hash. Staged files:
- `lsmd/cg_energy.py`
- `lsmd/transfer_modes.py`
- `scripts/validate_physics.py`
- `.superpowers/sdd/final-review-fixes-report.md`

## Concerns

1. **`validation_modeB.json` is stale** — the committed baseline was produced with the 2x WCA bug. Re-run `scripts/validate_physics.py --reweight` to regenerate valid Mode B numbers.
2. **Finding #3 (mh_rollout proposal/target mismatch)** — not fixed per instructions; acceptable for Phase 2 as it is not CLI-exposed.
3. **Finding #5 (noether collinear chain edge case)** — not fixed per instructions; not triggered on real proteins.
