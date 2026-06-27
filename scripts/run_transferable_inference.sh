#!/usr/bin/env bash
# =============================================================================
# Template: Transferable Propagator — fine-tune one protein, validate, rollout
#
# Starting from a pre-trained base checkpoint (e.g. checkpoints/v2_256h_90k.pt),
# this script fine-tunes on a single ATLAS-compatible protein shard, then runs
# physics validation at three inference temperatures.
#
# Usage
# -----
#   bash scripts/run_transferable_inference.sh
#   bash scripts/run_transferable_inference.sh --dry-run
#   bash scripts/run_transferable_inference.sh --skip-train   # validate only
#
# Edit the CONFIG section below, then run.
#
# Prerequisites
# -------------
#   BASE_CKPT          pre-trained checkpoint (hidden=256, 6 layers)
#   SHARD              atlas-compatible protein shard (.pt)
#                      For ATLAS proteins: python scripts/download_atlas_full.py --out data/atlas
#                      For custom protein: see scripts/prepare_kras_shard.py as template
#
# Outputs
# -------
#   PROTEIN_CKPT                    fine-tuned checkpoint
#   validation_${PROTEIN}_T*.json   three-temperature validation reports
#   rollout_${PROTEIN}/             autoregressive trajectory + PDB
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

# Protein identifier (used for output filenames)
PROTEIN="3u7t_A"

# Checkpoint to fine-tune from (the pre-trained base)
BASE_CKPT="checkpoints/v2_256h_90k.pt"

# Atlas-compatible shard for this protein
SHARD="data/atlas/${PROTEIN}.pt"

# Output checkpoint
PROTEIN_CKPT="checkpoints/ft_${PROTEIN}.pt"

# Fine-tuning hyperparameters
# Recommended: lower LR than pretraining to prevent catastrophic forgetting
LR="1e-4"
STEPS=5000
ACCUM=4
LAG_PS="100 200 500 1000 2000 5000 10000 20000 50000"

# Architecture — must match BASE_CKPT
HIDDEN=256
LAYERS=6
TEMP_EMB_DIM=8

# Validation settings
VAL_STEPS=300           # rollout steps (300 × 2 ns = 600 ns)
VAL_TAU_PS=2000         # lag per step in picoseconds
VAL_DIFF_STEPS=20       # denoising steps (DDPM subsampled)
VAL_TEMPS="300 375 450" # inference temperatures to sweep

# Rollout output (long trajectory for inspection)
ROLLOUT_DIR="rollout_${PROTEIN}"
ROLLOUT_STEPS=200        # 200 × 2 ns = 400 ns
ROLLOUT_DIFF_STEPS=20
ROLLOUT_ETA=1.0
ROLLOUT_TEMP_K=300

# ---------------------------------------------------------------------------

DRY_RUN=0; SKIP_TRAIN=0
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]]   && DRY_RUN=1
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
done

run() { echo "+  $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

echo "======================================================================"
echo "  Transferable Propagator: $PROTEIN"
echo "  Base  : $BASE_CKPT"
echo "  Shard : $SHARD"
echo "  Output: $PROTEIN_CKPT"
echo "======================================================================"
echo ""

mkdir -p checkpoints logs "$ROLLOUT_DIR"

# ---------------------------------------------------------------------------
# Step 1 — Fine-tune the transferable propagator on this protein
# ---------------------------------------------------------------------------
if [[ $SKIP_TRAIN -eq 1 ]]; then
    echo "=== Step 1: Skipping fine-tune (--skip-train) ==="
elif [[ -f "$PROTEIN_CKPT" ]]; then
    echo "=== Step 1: $PROTEIN_CKPT already exists — skipping ==="
else
    echo "=== Step 1: Fine-tune → $PROTEIN_CKPT ==="
    run python scripts/train_transfer.py \
        --shard         "$SHARD" \
        --resume        "$BASE_CKPT" \
        --lags_ps       $LAG_PS \
        --hidden        "$HIDDEN" \
        --layers        "$LAYERS" \
        --temp_emb_dim  "$TEMP_EMB_DIM" \
        --lr            "$LR" \
        --steps         "$STEPS" \
        --accum         "$ACCUM" \
        --grad_clip     1.0 \
        --time_reversal \
        --lam           0.0 \
        --log_every     250 \
        --out           "$PROTEIN_CKPT" \
        2>&1 | tee "logs/train_${PROTEIN}.log"
fi

# ---------------------------------------------------------------------------
# Step 2 — Physics validation at three inference temperatures
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Physics validation at T={$VAL_TEMPS} K ==="
for TEMP in $VAL_TEMPS; do
    OUT_JSON="validation_${PROTEIN}_T${TEMP}.json"
    if [[ -f "$OUT_JSON" && $SKIP_TRAIN -eq 0 ]]; then
        echo "  $OUT_JSON already exists — skipping (pass --skip-train to re-validate)"
        continue
    fi
    echo "  Validating at T=${TEMP} K → $OUT_JSON"
    run python scripts/validate_physics.py \
        --checkpoint         "$PROTEIN_CKPT" \
        --shard              "$SHARD" \
        --steps              "$VAL_STEPS" \
        --tau_ps             "$VAL_TAU_PS" \
        --diff_steps         "$VAL_DIFF_STEPS" \
        --eta                1.0 \
        --temp_K             "$TEMP" \
        --noether \
        --wca_sigma          4.5 \
        --wca_eps            0.3 \
        --wca_lam            0.05 \
        --bond_constraint_iters 5 \
        --max_update_norm    3.0 \
        --out                "$OUT_JSON"
done

# ---------------------------------------------------------------------------
# Step 3 — Generate a long rollout trajectory for inspection
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Autoregressive rollout (${ROLLOUT_STEPS} steps × ${VAL_TAU_PS} ps) ==="
echo "    Output: $ROLLOUT_DIR/trajectory.pdb"
run python scripts/eval_transfer.py \
    --checkpoint             "$PROTEIN_CKPT" \
    --shard                  "$SHARD" \
    --steps                  "$ROLLOUT_STEPS" \
    --tau_ps                 "$VAL_TAU_PS" \
    --diff_steps             "$ROLLOUT_DIFF_STEPS" \
    --eta                    "$ROLLOUT_ETA" \
    --temp_K                 "$ROLLOUT_TEMP_K" \
    --graph_rebuild_interval 5 \
    --wca_sigma              4.5 \
    --wca_eps                0.3 \
    --wca_lam                0.05 \
    --bond_constraint_iters  5 \
    --max_update_norm        3.0 \
    --out                    "${ROLLOUT_DIR}/eval_${PROTEIN}.json"

echo ""
echo "======================================================================"
echo "  Done: $PROTEIN"
echo ""
echo "  Checkpoint : $PROTEIN_CKPT"
for TEMP in $VAL_TEMPS; do
    echo "  Validation : validation_${PROTEIN}_T${TEMP}.json"
done
echo "  Rollout    : $ROLLOUT_DIR/"
echo ""
echo "  Key metrics to check (validation JSON → 'summary'):"
echo "    mean_rmsf_corr  > 0.90  (flexibility correlation)"
echo "    mean_dist_js    < 0.005 (Cα pairwise distances)"
echo "    mean_fes_js     < 0.50  (free-energy surface)"
echo "    mean_relax_ratio 0.5–2  (kinetics)"
echo "======================================================================"
