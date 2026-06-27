#!/usr/bin/env bash
# =============================================================================
# Train a per-protein DDPM checkpoint from an MD trajectory.
#
# Usage
# -----
#   bash scripts/train_ddpm.sh
#   bash scripts/train_ddpm.sh --dry-run
#
# Inputs
# ------
#   TRAJ       GROMACS .trr/.xtc trajectory  (or .dcd for NAMD/OpenMM)
#   TOPO       GROMACS .gro topology          (or .pdb for NAMD/OpenMM)
#
# Outputs
# -------
#   DATA_PT    bead point cloud (.pt)
#   CKPT       trained checkpoint (.pt)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TRAJ="WT/WT-sol6.trr"
TOPO="WT/WT-sol6.gro"

# "ca" | "2bead" (recommended) | "4bead"
BEAD_MODE="2bead"

DATA_PT="data/my_protein_${BEAD_MODE}.pt"
CKPT="checkpoints/my_protein_${BEAD_MODE}.pt"

# Lag times in frames (1 frame = 200 ps by default)
TAUS="1 2 5"           # 200 ps, 400 ps, 1 ns

# Architecture
EPOCHS=200
HIDDEN=64
LAYERS=3
LR="1e-3"
T_DIFF=200             # DDPM noise levels

# ---------------------------------------------------------------------------

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
run() { echo "+  $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

echo "=== Preprocess: $TRAJ → $DATA_PT ==="
if [[ ! -f "$DATA_PT" ]]; then
    run python scripts/preprocess.py \
        --traj  "$TRAJ" \
        --top   "$TOPO" \
        --atoms "$BEAD_MODE" \
        --out   "$DATA_PT"
else
    echo "  $DATA_PT exists — skipping preprocess"
fi

echo ""
echo "=== Train: $DATA_PT → $CKPT ==="
run python scripts/train.py \
    --frames   "$DATA_PT" \
    --taus     $TAUS \
    --epochs   "$EPOCHS" \
    --hidden   "$HIDDEN" \
    --layers   "$LAYERS" \
    --lr       "$LR" \
    --T_diff   "$T_DIFF" \
    --out      "$CKPT"

echo ""
echo "Done. Checkpoint: $CKPT"
