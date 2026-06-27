#!/usr/bin/env bash
# =============================================================================
# Fine-tune the transferable SE(3) propagator on one protein shard.
#
# Start from a pre-trained base checkpoint and adapt to a single protein with
# a low learning rate to prevent catastrophic forgetting. Works equally for
# ATLAS shards and custom shards built with prepare_kras_shard.py.
#
# Usage
# -----
#   bash scripts/finetune_transfer.sh
#   bash scripts/finetune_transfer.sh --dry-run
#
# For Phase-1 universal fine-tuning across all ATLAS proteins, set
# USE_SHARDS_DIR=1 and point SHARDS_DIR at data/atlas. See docs/tutorial.md.
#
# Inputs
# ------
#   BASE_CKPT    pre-trained checkpoint (must match HIDDEN/LAYERS below)
#   SHARD        single atlas-compatible protein shard (.pt)
#
# Outputs
# -------
#   OUT_CKPT     fine-tuned checkpoint (.pt)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Base checkpoint to resume from
BASE_CKPT="checkpoints/v2_256h_90k.pt"

# Single-protein fine-tune (recommended for per-protein adaptation)
SHARD="data/atlas/3u7t_A.pt"

# Set to 1 to do a directory-wide Phase-1 fine-tune instead of single-shard
USE_SHARDS_DIR=0
SHARDS_DIR="data/atlas"

# Output
OUT_CKPT="checkpoints/ft_3u7t_A.pt"

# Lag times in picoseconds — keep consistent with the base checkpoint
# Recommended full set for production; reduce for faster iteration
LAG_PS="100 200 500 1000 2000 5000 10000 20000 50000"

# Architecture — must match BASE_CKPT exactly
HIDDEN=256
LAYERS=6
TEMP_EMB_DIM=8    # temperature conditioning; use 0 for old checkpoints

# Fine-tuning hyperparameters
LR="1e-4"          # 10× lower than pre-training (1e-3) to prevent forgetting
STEPS=5000         # ~3 dataset epochs at accum=4 for a 5000-frame shard
ACCUM=4            # gradient accumulation (effective batch ≈ 4 × max_union_nodes)
GRAD_CLIP=1.0

# Physics options
TIME_REVERSAL=1    # 1 = enable (recommended); doubles data + enforces reversibility
LAM=0.0            # C1 geometry penalty; keep 0.0 for ATLAS fine-tuning

# Logging
LOG_EVERY=250

# ---------------------------------------------------------------------------

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
run() { echo "+  $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

# Build source flag
if [[ $USE_SHARDS_DIR -eq 1 ]]; then
    SRC_FLAG="--shards_dir $SHARDS_DIR"
    echo "Mode: Phase-1 universal fine-tune (all proteins in $SHARDS_DIR)"
else
    SRC_FLAG="--shard $SHARD"
    echo "Mode: Per-protein fine-tune ($SHARD)"
fi

TR_FLAG=""
[[ $TIME_REVERSAL -eq 1 ]] && TR_FLAG="--time_reversal"

echo "  Base     : $BASE_CKPT"
echo "  Output   : $OUT_CKPT"
echo "  Steps    : $STEPS   LR: $LR   Accum: $ACCUM"
echo ""

mkdir -p checkpoints logs

run python scripts/train_transfer.py \
    $SRC_FLAG \
    --resume        "$BASE_CKPT" \
    --lags_ps       $LAG_PS \
    --hidden        "$HIDDEN" \
    --layers        "$LAYERS" \
    --temp_emb_dim  "$TEMP_EMB_DIM" \
    --lr            "$LR" \
    --steps         "$STEPS" \
    --accum         "$ACCUM" \
    --grad_clip     "$GRAD_CLIP" \
    --lam           "$LAM" \
    --log_every     "$LOG_EVERY" \
    $TR_FLAG \
    --out           "$OUT_CKPT" \
    2>&1 | tee "logs/finetune_$(basename "$OUT_CKPT" .pt).log"

echo ""
echo "Done. Checkpoint: $OUT_CKPT"
echo "Next: bash scripts/infer_transfer.sh  (update CKPT in CONFIG)"
