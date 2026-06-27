#!/usr/bin/env bash
# =============================================================================
# Template: CV-Guided Conformational Exploration
#
# Runs the history-dependent repulsion explorer on a fine-tuned checkpoint.
# The model is steered away from previously accepted structures in PCA + Rg +
# RMSD collective-variable space, systematically filling the conformational
# landscape.
#
# Usage
# -----
#   bash scripts/run_explore.sh            # full exploration run
#   bash scripts/run_explore.sh --resume   # extend an existing run
#   bash scripts/run_explore.sh --dry-run  # print commands without running
#
# Edit the CONFIG section below, then run.
#
# Prerequisites
# -------------
#   CHECKPOINT     fine-tuned transferable propagator checkpoint (.pt)
#   SHARD          atlas-compatible protein shard (.pt)
#
# Outputs
# -------
#   OUT_DIR/candidates/NNNNN.pdb   Cα PDB files for accepted structures
#   OUT_DIR/summary.json           per-structure metrics (RMSD, CV, clashes)
#   OUT_DIR/cv_coords.npy          CV coordinates [M, n_pc+2]
#   OUT_DIR/structures.pt          stacked Cα tensors [M, N, 3]
#   OUT_DIR/cv_coverage.png        PC1 vs PC2 scatter (training grey, gen colour)
#   OUT_DIR/cv_basis.pt            saved PCA basis (required for --resume)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

# Checkpoint and shard (must be atlas-compatible: has res_type, chain_id, dt, etc.)
CHECKPOINT="checkpoints/v4_3u7t_A.pt"
SHARD="data/atlas/3u7t_A.pt"

# Output directory
OUT_DIR="explore_out"

# Exploration scale
N_EXPLORE=500          # total exploration attempts (accepted + rejected)
N_STEPS=100            # rollout steps per attempt (one step = one tau)
TAU_PS=2000            # physical lag per rollout step (ps); must be in training lags
TEMP_K=310             # simulation temperature (K); physiological = 310 K

# CV-space guidance
N_PC=5                 # PCA components in CV basis (3–5 for small proteins, 5–8 for large)
K_GUIDE=0.10           # CV repulsion strength; increase to 0.15–0.20 for stronger steering
SIGMA_CV=1.0           # Gaussian width in normalised CV units (smaller = finer resolution)
GUIDE_WARMUP=50        # activate repulsion after this many accepted structures

# Inference speed (DDPM vs DDIM)
DIFF_STEPS=20          # denoising steps; 20 is fast with good quality; 50–200 for best quality
ETA=1.0                # 1.0 = stochastic DDPM (diverse); 0.0 = deterministic DDIM (faster)
                       # Use --ddim flag (below) instead of manually setting both
USE_DDIM=0             # set to 1 for 10× faster inference via DDIM (eta=0, diff_steps=10)

# Excluded-volume guidance (WCA)
WCA_SIGMA=4.5          # Cα–Cα excluded-volume diameter (Å); set 0 to disable
WCA_EPS=0.3            # WCA well depth (kcal/mol)
WCA_LAM=0.05           # guidance step size; increase to 0.08–0.10 for flexible loops

# Performance
GRAPH_REBUILD=5        # rebuild kNN topology every N steps; 5–10 gives ~5× speedup

# Reproducibility
SEED=42

# ---------------------------------------------------------------------------

DRY_RUN=0; RESUME_FLAG=""
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
    [[ "$arg" == "--resume" ]]  && RESUME_FLAG="--resume"
done

run() { echo "+  $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

# Build --ddim flag if requested
DDIM_FLAG=""
if [[ $USE_DDIM -eq 1 ]]; then
    DDIM_FLAG="--ddim"
    DIFF_STEPS=10
    ETA=0.0
fi

echo "======================================================================"
echo "  CV-Guided Conformational Exploration"
echo "  Checkpoint : $CHECKPOINT"
echo "  Shard      : $SHARD"
echo "  Output     : $OUT_DIR"
echo "  Attempts   : $N_EXPLORE  (steps/attempt=$N_STEPS, tau=${TAU_PS} ps)"
echo "  CV guidance: n_pc=$N_PC  k_guide=$K_GUIDE  sigma=$SIGMA_CV  warmup=$GUIDE_WARMUP"
[[ -n "$RESUME_FLAG" ]] && echo "  Mode       : RESUME (continuing from existing summary.json)"
echo "======================================================================"
echo ""

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Step 1 — Run exploration
# ---------------------------------------------------------------------------
echo "=== Step 1: Exploring conformational space ==="
run python scripts/explore_conformations.py \
    --checkpoint             "$CHECKPOINT" \
    --shard                  "$SHARD" \
    --n_explore              "$N_EXPLORE" \
    --n_steps                "$N_STEPS" \
    --tau_ps                 "$TAU_PS" \
    --temp_K                 "$TEMP_K" \
    --n_pc                   "$N_PC" \
    --k_guide                "$K_GUIDE" \
    --sigma_cv               "$SIGMA_CV" \
    --guide_warmup           "$GUIDE_WARMUP" \
    --graph_rebuild_interval "$GRAPH_REBUILD" \
    --wca_sigma              "$WCA_SIGMA" \
    --wca_eps                "$WCA_EPS" \
    --wca_lam                "$WCA_LAM" \
    --diff_steps             "$DIFF_STEPS" \
    --eta                    "$ETA" \
    $DDIM_FLAG \
    --out                    "$OUT_DIR" \
    --seed                   "$SEED" \
    $RESUME_FLAG

# ---------------------------------------------------------------------------
# Step 2 — Summarize accepted structures
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Summarizing exploration results ==="
if [[ -f "$OUT_DIR/summary.json" ]]; then
    run python scripts/summarize_exploration.py --out "$OUT_DIR"
else
    echo "  summary.json not found (no structures were accepted)"
fi

echo ""
echo "======================================================================"
echo "  Done."
echo ""
echo "  Accepted structures  : $OUT_DIR/candidates/"
echo "  Summary JSON         : $OUT_DIR/summary.json"
echo "  CV coordinates (npy) : $OUT_DIR/cv_coords.npy"
echo "  Stacked structures   : $OUT_DIR/structures.pt"
echo "  Coverage plot        : $OUT_DIR/cv_coverage.png"
echo ""
echo "  Next steps:"
echo "    - Run short MD relaxation on candidates/ to validate stability"
echo "    - Update md_pass / md_rmsd_final in summary.json"
echo "    - Re-run summarize_exploration.py for classification + MD survivor plot"
echo ""
echo "  Extend this run (add more attempts):"
echo "    bash $0 --resume"
echo "======================================================================"
