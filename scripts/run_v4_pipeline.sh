#!/bin/bash
# V4 training pipeline: long-lag universal fine-tune + per-protein fine-tunes
# with inference temperature sweep.
#
# Usage: bash scripts/run_v4_pipeline.sh [--dry-run]
#   --dry-run: print commands without executing (for verification)
#
# Prerequisites:
#   checkpoints/v3_lam0.pt         (Phase 1 base)
#   data/atlas/{protein}.pt        (per-protein shards)
#
# Outputs:
#   checkpoints/v4_longlags.pt
#   checkpoints/v4_{protein}.pt    (6 files)
#   validation_v4_*.json           (19 files)
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

run() {
    echo "+ $*"
    if [[ $DRY_RUN -eq 0 ]]; then "$@"; fi
}

LOG_DIR=logs
mkdir -p "$LOG_DIR" checkpoints

PROTEINS="3u7t_A 4p3a_B 1b2s_F 2y4x_B 1z0b_A 6ovk_R"
VFLAGS="--steps 300 --tau_ps 2000 --diff_steps 20 --eta 1.0 --noether"
BASE_TRAIN="--hidden 256 --layers 6 --lags_ps 100 200 500 1000 2000 5000 10000 20000 50000 --lam 0.0"

echo "=== V4 PIPELINE START $(date) ===" | tee "$LOG_DIR/v4_pipeline_$(date +%Y%m%d_%H%M%S).log"

# ── Phase 1: Long-lag universal fine-tune ─────────────────────────────────────
echo "[Phase 1] Train v4_longlags (lags=0.1k..50k ps, 20k steps)"
run python scripts/train_transfer.py \
    --shards_dir data/atlas \
    --resume checkpoints/v3_lam0.pt \
    $BASE_TRAIN \
    --steps 20000 \
    --out checkpoints/v4_longlags.pt \
    2>&1 | tee "$LOG_DIR/train_v4_longlags.log"

echo "[Phase 1] Validate v4_longlags at T=300 K"
run python scripts/validate_physics.py \
    --checkpoint checkpoints/v4_longlags.pt \
    $(for p in $PROTEINS; do echo "--shard data/atlas/${p}.pt"; done) \
    $VFLAGS --temp_K 300 \
    --out validation_v4_longlags_T300.json

# ── Phase 2: Per-protein fine-tune ───────────────────────────────────────────
for protein in $PROTEINS; do
    echo "[Phase 2] Train v4_${protein} (5k steps on ${protein} only)"
    run python scripts/train_transfer.py \
        --shard "data/atlas/${protein}.pt" \
        --resume checkpoints/v4_longlags.pt \
        $BASE_TRAIN \
        --steps 5000 \
        --out "checkpoints/v4_${protein}.pt" \
        2>&1 | tee "$LOG_DIR/train_v4_${protein}.log"

    for temp in 300 375 450; do
        echo "[Phase 2] Validate v4_${protein} at T=${temp} K"
        run python scripts/validate_physics.py \
            --checkpoint "checkpoints/v4_${protein}.pt" \
            --shard "data/atlas/${protein}.pt" \
            $VFLAGS --temp_K "$temp" \
            --out "validation_v4_${protein}_T${temp}.json"
    done
done

echo "=== V4 PIPELINE COMPLETE $(date) ==="
echo "Checkpoints: checkpoints/v4_longlags.pt + checkpoints/v4_{protein}.pt x6"
echo "Validation:  validation_v4_*.json (19 files)"
