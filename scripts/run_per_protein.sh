#!/usr/bin/env bash
# =============================================================================
# Template: Per-Protein DDPM — preprocess, train, validate, generate trajectory
#
# This script trains a DDPM model on a single MD trajectory and generates
# a long generative trajectory at 10^5–10^6× speedup.
#
# Usage
# -----
#   bash scripts/run_per_protein.sh            # full run (GPU recommended)
#   bash scripts/run_per_protein.sh --dry-run  # print commands without running
#
# Edit the CONFIG section below, then run.
#
# Inputs
# ------
#   TRAJ_FILE   GROMACS .trr / .xtc trajectory (or .dcd for NAMD/OpenMM)
#   TOPO_FILE   GROMACS .gro topology (or .pdb for NAMD/OpenMM)
#
# Outputs
# -------
#   DATA_PT                 bead point cloud (.pt)
#   CHECKPOINT              trained model checkpoint (.pt)
#   INFER_DIR/metrics.json  8-sample validation metrics
#   GENMD_DIR/trajectory.pdb  long generative trajectory (multi-MODEL PDB)
#   GENMD_DIR/allatom.pdb     all-atom reconstruction (requires TRAJ_FILE)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

# Input trajectory and topology (GROMACS format; use .dcd/.pdb for NAMD)
TRAJ_FILE="WT/WT-sol6.trr"
TOPO_FILE="WT/WT-sol6.gro"

# Bead representation: "ca" (fastest), "2bead" (recommended), "4bead" (full backbone)
BEAD_MODE="2bead"

# Output locations
DATA_PT="data/my_protein_${BEAD_MODE}.pt"
CHECKPOINT="checkpoints/my_protein_${BEAD_MODE}_200ep.pt"
INFER_DIR="infer_out"
GENMD_DIR="genmd_out"

# Training hyperparameters
TAUS="1 2 5"      # lag times in frames (200 ps/frame → 200 ps, 400 ps, 1 ns)
EPOCHS=200
HIDDEN=64
LAYERS=3

# Inference
TAU_INF=2         # lag for inference (must be in TAUS)
SAMPLES=8         # K snapshots for quick validation

# Trajectory generation
GENMD_TAU=2       # lag per generative step
GENMD_STEPS=2500  # steps = simulated time (2500 × 400 ps = 1 μs at τ=2)
GENMD_MODE="mimic"   # "mimic" (anchored) or "explore" (free exploration)
ANCHOR_EVERY=50   # re-anchor interval for mimic mode

# ---------------------------------------------------------------------------

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

run() {
    echo "+  $*"
    [[ $DRY_RUN -eq 1 ]] || "$@"
}

echo "======================================================================"
echo "  Per-Protein DDPM Pipeline"
echo "  Bead mode : $BEAD_MODE"
echo "  Checkpoint: $CHECKPOINT"
echo "======================================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1 — Preprocess: convert MD trajectory to bead point cloud
# ---------------------------------------------------------------------------
if [[ ! -f "$DATA_PT" ]]; then
    echo "=== Step 1: Preprocess → $DATA_PT ==="
    run python scripts/preprocess.py \
        --traj  "$TRAJ_FILE" \
        --top   "$TOPO_FILE" \
        --atoms "$BEAD_MODE" \
        --out   "$DATA_PT"
else
    echo "=== Step 1: $DATA_PT already exists — skipping ==="
fi

# ---------------------------------------------------------------------------
# Step 2 — Train the per-protein DDPM
# ---------------------------------------------------------------------------
if [[ ! -f "$CHECKPOINT" ]]; then
    echo ""
    echo "=== Step 2: Train → $CHECKPOINT ==="
    run python scripts/train.py \
        --frames  "$DATA_PT" \
        --taus    $TAUS \
        --epochs  "$EPOCHS" \
        --hidden  "$HIDDEN" \
        --layers  "$LAYERS" \
        --out     "$CHECKPOINT"
else
    echo ""
    echo "=== Step 2: $CHECKPOINT already exists — skipping ==="
fi

# ---------------------------------------------------------------------------
# Step 3 — Quick validation: 8 samples, compute metrics
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Quick validation (K=$SAMPLES) → $INFER_DIR ==="
run python scripts/infer.py \
    --checkpoint "$CHECKPOINT" \
    --frames     "$DATA_PT" \
    --tau        "$TAU_INF" \
    --K          "$SAMPLES" \
    --diff_steps 50 \
    --out        "$INFER_DIR"

# ---------------------------------------------------------------------------
# Step 4 — Generate long trajectory
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Generate ${GENMD_STEPS}-step trajectory → $GENMD_DIR ==="
run python scripts/generate_md.py \
    --checkpoint   "$CHECKPOINT" \
    --frames       "$DATA_PT" \
    --tau          "$GENMD_TAU" \
    --steps        "$GENMD_STEPS" \
    --sample_mode  "$GENMD_MODE" \
    --anchor_every "$ANCHOR_EVERY" \
    --min_energy \
    --k_clash      5.0 \
    --diff_steps   50 \
    --out          "$GENMD_DIR"

# ---------------------------------------------------------------------------
# Step 5 — All-atom reconstruction (requires original trajectory)
# ---------------------------------------------------------------------------
if [[ -f "$TRAJ_FILE" ]]; then
    echo ""
    echo "=== Step 5: All-atom reconstruction → $GENMD_DIR/allatom.pdb ==="
    run python scripts/reconstruct.py \
        --beads      "$GENMD_DIR/trajectory.pdb" \
        --traj       "$TRAJ_FILE" \
        --top        "$TOPO_FILE" \
        --checkpoint "$CHECKPOINT" \
        --out        "$GENMD_DIR/allatom.pdb"
else
    echo ""
    echo "=== Step 5: Skipping reconstruction (TRAJ_FILE not found) ==="
fi

echo ""
echo "======================================================================"
echo "  Done."
echo ""
echo "  Metrics      : $INFER_DIR/metrics.json"
echo "  Trajectory   : $GENMD_DIR/trajectory.pdb"
if [[ -f "$TRAJ_FILE" ]]; then
    echo "  All-atom PDB : $GENMD_DIR/allatom.pdb"
fi
echo ""
echo "  Visualise:"
echo "    pymol $GENMD_DIR/trajectory.pdb"
if [[ -f "$GENMD_DIR/allatom.pdb" ]]; then
    echo "    pymol $GENMD_DIR/allatom.pdb"
fi
echo "======================================================================"
