# Active Learning Loop for Single-Protein Conformational Exploration — Design Spec

**Date:** 2026-06-28  
**Status:** Approved for implementation

---

## Goal

Build a self-improving active learning loop that explores the full conformational ensemble of a single protein starting from only a static PDB structure (crystal or AlphaFold). The loop iteratively generates diverse proposals with the PropagatorNet, validates them with OpenMM MD, and fine-tunes the model on accumulated MD data — converging when coverage, budget, or FES stability criteria are met.

## Architecture

The loop is a thin sequential orchestrator over existing components. No existing module is substantially rewritten.

```
Round N
  ├─ 1. Generate proposals      → explore_conformations logic (existing)
  ├─ 2. Filter novel            → min-RMSD vs all accumulated frames
  ├─ 3. Reconstruct all-atom    → AllAtomReconstructor (existing)
  ├─ 4. Run MD (parallel)       → run_md() (existing)
  ├─ 5. Extract Cα frames       → shard_from_md_runs()  [NEW]
  ├─ 6. Build replay shard      → build_replay_shard()  [NEW]
  ├─ 7. Fine-tune model         → train_transfer.py subprocess (existing)
  ├─ 8. Refit CV space          → CVSpace.fit() (existing, minor guard)
  └─ 9. Check stopping          → check_convergence()   [NEW]
```

Round 0 prepends a one-time **bootstrap check** before step 1.

---

## New Files

| File | Purpose |
|---|---|
| `lsmd/active_loop.py` | Pure logic: `bootstrap_check`, `shard_from_md_runs`, `build_replay_shard`, `check_convergence` |
| `scripts/active_learning.py` | CLI orchestrator — runs the round loop, handles checkpointing |

## Changed Files

| File | Change |
|---|---|
| `lsmd/cv_guidance.py` | `CVSpace.fit()` — skip PCA when F < n_pc; use Rg+RMSD only |

---

## Section 1 — Bootstrap Check (Round 0 only)

Decides whether the universal model can cold-start from the static PDB, or needs a short MD run first.

```
Input: static PDB
       ↓
Build single-frame shard (F=1) from PDB Cα coords
       ↓
Run 20 DDIM proposals with universal checkpoint, no CV guidance
       ↓
Geometry check: fraction with Cα bond deviation < 0.15 Å AND 0 clashes
       ↓
≥ 80% pass  →  proceed (zero-MD path)
< 80% pass  →  run bootstrap MD (--bootstrap-ns ns via run_md())
              extract Cα frames at 200 ps intervals → initial shard
              then proceed
```

**Single-frame edge cases handled automatically:**

- **CVSpace**: F=1 < n_pc → skip PCA, use 2D CV (Rg + RMSD only). Refit to full n_pc=5 once ≥ 20 frames accumulated.
- **AllAtomReconstructor**: nearest-frame lookup works at F=1; all proposals use the input PDB as template. Quality improves each round as MD frames accumulate.
- **ref_bond**: per-frame mean of Cα–Cα bond lengths — correct at F=1 (no compression artifact).

**`bootstrap_check(pdb_path, checkpoint, device, bootstrap_ns, out_dir) -> shard`**

Returns a shard dict `{t, res_type, chain_id, res_index, seq, n_res, dt}` with either F=1 (zero-MD path) or F=bootstrap_frames (after short MD).

---

## Section 2 — Proposal Generation and Novel Filtering

Uses existing `rollout()` + CV guidance from `explore_conformations.py` logic, called with the current round's checkpoint and CV basis.

**Novel filter** — after generating `--proposals` Cα structures:

```python
# For each proposal p, compute min RMSD to all accumulated frames
min_rmsd = min(kabsch_rmsd(p, f) for f in accumulated_frames)
novel = [p for p in proposals if min_rmsd[p] > novel_threshold]
selected = random.sample(novel, min(batch_size, len(novel)))
```

If `len(novel) < min_batch` (default: `batch_size // 2`), the round logs a warning and uses all novel structures. If `len(novel) == 0`, the loop terminates early with `converged=True` (landscape exhausted before stopping criterion).

---

## Section 3 — Replay Shard Builder

**`shard_from_md_runs(md_run_dirs, dt_ps=200) -> Tensor[F, N, 3]`**

For each `md_run_dir` where `metrics.json` has `error == null`:
- Load `trajectory.dcd` + `topology.pdb` via mdtraj
- Extract Cα positions at `dt_ps` intervals
- Return concatenated `(F_total, N, 3)` float32 tensor in Å

Skips failed runs silently (they will auto-retry on next loop launch via existing checkpoint logic).

**`build_replay_shard(new_frames, accumulated_pt, protein_meta, replay_cap=5000) -> dict`**

`protein_meta` is the fixed per-protein dict `{res_type, chain_id, res_index, seq, n_res}` extracted from the initial shard during `bootstrap_check` and reused unchanged every round (residue identity never changes, only frames do).

```python
history = torch.load(accumulated_pt)["t"]          # all prior frames
n_history = min(replay_cap - len(new_frames), len(history))
sampled_history = history[torch.randperm(len(history))[:n_history]]
combined = torch.cat([new_frames, sampled_history], dim=0)
shard = {**protein_meta, "t": combined, "dt": 200}  # dt in ps matches extraction interval
```

Appends `new_frames` to `accumulated_pt["t"]` (grows every round; metadata is not stored redundantly). Returns a shard dict ready for `train_transfer.py`.

**Fine-tuning call (subprocess):**

```bash
python scripts/train_transfer.py \
    --shard   round_N/replay_shard.pt \
    --resume  checkpoints/v2_256h_90k.pt \
    --steps   {fine_tune_steps} \
    --lr      1e-4 \
    --hidden  256 \
    --layers  6 \
    --device  {device} \
    --out     round_N/checkpoint.pt
```

Always resumes from the universal base checkpoint (not the previous round's) to prevent compounding fine-tuning drift.

---

## Section 4 — Convergence Checkers

**`check_convergence(criterion, threshold, state) -> (bool, float)`**

Returns `(converged, metric_value)`. Logs `metric_value` every round regardless of convergence.

### `budget`
```python
converged = state["total_md_ns"] >= threshold
metric    = state["total_md_ns"]
```

### `coverage`
```python
# novel_fraction = fraction of last round's selected proposals
# that were > novel_threshold Å from all pre-round accumulated frames
converged = state["last_novel_fraction"] < threshold   # default 0.10
metric    = state["last_novel_fraction"]
```

### `fes`
Builds a 50×50 2D histogram over (PC1, PC2) of all accumulated frames; computes JS divergence against the previous round's histogram.

```python
# Requires: round >= 2 AND total_accumulated_frames >= 50
# Otherwise: converged=False, metric=nan
fes_js    = jensen_shannon(hist_current, hist_previous)
converged = fes_js < threshold    # default 0.05
metric    = fes_js
```

---

## Section 5 — CLI and File Layout

### CLI

```bash
python scripts/active_learning.py \
    --pdb              input.pdb              \  # required
    --checkpoint       checkpoints/v2_256h_90k.pt \
    --out              my_protein_loop        \
    --rounds           10                     \  # max rounds
    --proposals        100                    \  # DDIM proposals per round
    --batch-size       20                     \  # MD runs per round
    --md-ns            10                     \  # MD ns per structure
    --replay-cap       5000                   \  # max frames in replay shard
    --novel-threshold  1.5                    \  # Å, novelty RMSD cutoff
    --stop             coverage               \  # budget | coverage | fes
    --stop-threshold   0.10                   \  # criterion-specific value
    --bootstrap-ns     10                     \  # bootstrap MD length (ns)
    --fine-tune-steps  2000                   \  # training steps per round
    --n-parallel       4                      \  # parallel MD workers
    --device           cuda
```

### Output Layout

```
my_protein_loop/
  input.pdb                      # copy of input PDB
  accumulated_frames.pt          # growing Cα frame store (all rounds)
  loop_summary.json              # per-round metrics table
  round_0/
    .bootstrap_used              # present if bootstrap MD was run
    bootstrap_shard.pt           # initial shard (F=1 or bootstrap frames)
    proposals/                   # Cα PDB proposals (all, pre-filter)
    allatom/                     # reconstructed heavy-atom PDBs (selected)
    md_runs/                     # one subdir per validated structure
    replay_shard.pt              # shard used for fine-tuning this round
    checkpoint.pt                # model for round 1
    cv_basis.pt                  # CVSpace fitted on accumulated frames
    round_summary.json           # per-round metrics
    .done                        # written last — resume skips this round
  round_1/
    ...
  final_checkpoint.pt            # symlink → last round's checkpoint.pt
  final_shard.pt                 # symlink → accumulated_frames.pt
```

### `round_summary.json` schema

```json
{
  "round": 3,
  "bootstrap_used": false,
  "n_proposals_generated": 100,
  "n_novel_filtered": 24,
  "n_md_attempted": 20,
  "n_md_success": 19,
  "new_frames_this_round": 190,
  "total_frames_accumulated": 570,
  "total_md_ns": 190.0,
  "last_novel_fraction": 0.24,
  "fes_js": 0.031,
  "converged": false,
  "stop_criterion": "coverage",
  "stop_threshold": 0.10
}
```

### `loop_summary.json` schema

Top-level file with one entry per completed round:

```json
[
  {"round": 0, "n_md_success": 18, "total_md_ns": 180.0, "novel_fraction": 0.90, "fes_js": null,  "converged": false},
  {"round": 1, "n_md_success": 19, "total_md_ns": 370.0, "novel_fraction": 0.55, "fes_js": 0.14,  "converged": false},
  {"round": 2, "n_md_success": 20, "total_md_ns": 570.0, "novel_fraction": 0.24, "fes_js": 0.031, "converged": false}
]
```

---

## Global Constraints

- Python 3.10+; no new dependencies beyond what the project already uses (torch, mdtraj, openmm, numpy, scipy)
- All new logic goes in `lsmd/active_loop.py` and `scripts/active_learning.py`
- The only change to an existing module is a single guard in `CVSpace.fit()` in `lsmd/cv_guidance.py`
- Fine-tuning always calls `train_transfer.py` as a subprocess (not imported inline) to isolate GPU memory lifecycle
- Resume is round-level: a `.done`-stamped round is never re-run; failed rounds restart from scratch
- Failed MD runs within a round are not cached (`error != null` check in existing `run_md()`) and will be retried if the round restarts
- `accumulated_frames.pt` is the single source of truth for all prior conformations; it is append-only
- All RMSD computations use Kabsch superposition (already available via mdtraj)

---

## Testing Targets

| Test | What it covers |
|---|---|
| `test_bootstrap_check_zero_md` | Good universal model → skip bootstrap |
| `test_bootstrap_check_triggers_md` | Poor geometry rate → bootstrap MD runs |
| `test_cvspace_single_frame` | F=1 → PCA skipped, Rg+RMSD CV only |
| `test_shard_from_md_runs` | Extracts Cα frames, skips failed runs |
| `test_build_replay_shard_capped` | Never exceeds replay_cap |
| `test_build_replay_shard_small_history` | History < cap → uses all history |
| `test_convergence_budget` | Triggers at correct total_md_ns |
| `test_convergence_coverage` | Triggers at correct novel_fraction |
| `test_convergence_fes_insufficient_data` | Returns nan before round 2 / 50 frames |
| `test_convergence_fes` | Triggers at correct JS divergence |
| `test_active_loop_resume` | Skips .done rounds, resumes mid-loop |
| `test_active_loop_early_termination` | novel=0 → converged=True |
