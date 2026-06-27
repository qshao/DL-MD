#!/usr/bin/env bash
# =============================================================================
# KRAS-WT fine-tuning + enhanced conformational exploration pipeline
#
# Steps
# -----
#   1. Prepare atlas-compatible KRAS shard from legacy wt_frames.pt
#   2. Fine-tune the pretrained SE(3) PropagatorNet on KRAS-WT at low LR
#   3. Run CV-guided conformation explorer to sample diverse Cα structures
#
# Reproducibility
# ---------------
#   All random seeds are fixed.  Re-running produces the same exploration
#   trajectory.  Pass --resume to step 3 to extend an existing run.
#
# Requirements
# ------------
#   data/wt_frames.pt         KRAS-WT legacy trajectory (5001 × 169 × 3)
#   WT/WT_fixed.pdb           PDB with canonical KRAS-WT sequence (169 CA atoms)
#   checkpoints/v2_256h_90k.pt  Pretrained transferable propagator
#
# Usage
# -----
#   bash scripts/run_kras_finetune_explore.sh            # full run (GPU)
#   bash scripts/run_kras_finetune_explore.sh --resume   # extend exploration
#
# Outputs
# -------
#   data/kras_wt_shard.pt       atlas-compatible KRAS shard (created once)
#   checkpoints/kras_ft.pt      KRAS fine-tuned checkpoint
#   kras_exploration/           CV-guided exploration results
#     candidates/NNNNN.pdb      accepted Cα PDB files
#     summary.json              per-structure metrics (RMSD, CV, clashes)
#     cv_coords.npy             CV vectors of all accepted structures [M, n_pc+2]
#     structures.pt             stacked Cα tensors [M, N, 3]
#     cv_coverage.png           PC1 vs PC2 scatter (training grey, generated colour)
#     cv_basis.pt               saved PCA basis for resume consistency
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

RESUME=0
for arg in "$@"; do
    [[ "$arg" == "--resume" ]] && RESUME=1
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WF_FRAMES="data/wt_frames.pt"
PDB="WT/WT_fixed.pdb"
SHARD="data/kras_wt_shard.pt"
PRETRAINED="checkpoints/v2_256h_90k.pt"
FINETUNED="checkpoints/kras_ft.pt"
EXPLORE_OUT="kras_exploration"

# ---------------------------------------------------------------------------
# Step 1 — Prepare shard (idempotent: skip if already exists)
# ---------------------------------------------------------------------------
if [[ ! -f "$SHARD" ]]; then
    echo "=== Step 1: Preparing KRAS shard ==="
    python scripts/prepare_kras_shard.py \
        --wt_frames "$WF_FRAMES" \
        --pdb       "$PDB"      \
        --dt        200.0       \
        --out       "$SHARD"
else
    echo "=== Step 1: Shard already exists at $SHARD — skipping ==="
fi

# ---------------------------------------------------------------------------
# Step 2 — Fine-tune pretrained model on KRAS-WT
#
# Rationale for hyperparameters:
#   --lr 1e-4          10× lower than ATLAS pre-training (1e-3) to prevent
#                      catastrophic forgetting of transferable geometry priors
#   --steps 5000       ~3 full-dataset epochs at accum=4 (5001 frames ÷ 4)
#                      enough to adapt lag-specific fluctuation scale without
#                      memorizing the KRAS ensemble
#   --lags_ps          same as pre-training so normalisation is consistent
#   --time_reversal    doubles effective data via microscopic reversibility
#   --temp_emb_dim 8   must match pre-trained checkpoint's embedding size
#   --no-amp           keep fp32 on CPU; remove flag if running on GPU
# ---------------------------------------------------------------------------
if [[ ! -f "$FINETUNED" ]]; then
    echo ""
    echo "=== Step 2: Fine-tuning on KRAS-WT ==="
    python scripts/train_transfer.py \
        --shard        "$SHARD"       \
        --resume       "$PRETRAINED"  \
        --lags_ps      2000 5000 10000 \
        --hidden       256            \
        --layers       6              \
        --lr           1e-4           \
        --steps        5000           \
        --accum        4              \
        --grad_clip    1.0            \
        --temp_emb_dim 8              \
        --time_reversal               \
        --log_every    250            \
        --out          "$FINETUNED"
else
    echo ""
    echo "=== Step 2: Fine-tuned checkpoint already exists at $FINETUNED — skipping ==="
fi

# ---------------------------------------------------------------------------
# Step 3 — Enhanced CV-guided conformational exploration
#
# Rationale for hyperparameters:
#   --n_explore 1000   generate up to 1000 accepted Cα conformations
#   --n_steps 200      200-step rollout × 2 ns/step = 400 ns net simulated time
#                      per attempt; long enough to leave the native basin
#   --tau_ps 2000      2 ns lag — same as pre-training lag; model is well-calibrated
#   --temp_K 310       near-physiological (37 °C) for realistic fluctuations
#   --k_guide 0.15     3× stronger CV repulsion than default to push harder away
#                      from already-visited regions
#   --sigma_cv 0.8     tighter Gaussian than default 1.0; distinguishes finer
#                      structural differences in KRAS switch I/II conformations
#   --n_pc 5           5 PCs captures >85% of KRAS conformational variance
#                      (first 3 cover active/inactive, switch loop, helix 3)
#   --guide_warmup 20  activate CV repulsion after 20 accepts (default 50);
#                      KRAS basin is narrow so early guidance is productive
#   --graph_rebuild_interval 5
#                      reuse kNN topology for 5 steps; recompute edge features
#                      each step — ~5× faster rollout with negligible accuracy loss
#   --wca_lam 0.08     slightly stronger WCA than default 0.05 to prevent clashes
#                      in the flexible switch II loop (residues 58-72)
#   --diff_steps 20    20 DDPM steps; good balance of diversity and structure
#   --eta 1.0          stochastic DDPM (not DDIM) for maximal diversity
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: CV-guided conformational exploration ==="
RESUME_FLAG=""
[[ $RESUME -eq 1 ]] && RESUME_FLAG="--resume"

DEVICE="cuda"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || DEVICE="cpu"

python scripts/explore_conformations.py \
    --checkpoint  "$FINETUNED"   \
    --shard       "$SHARD"       \
    --n_explore   1000           \
    --n_steps     200            \
    --tau_ps      2000           \
    --temp_K      310            \
    --k_guide     0.15           \
    --sigma_cv    0.8            \
    --n_pc        5              \
    --guide_warmup 20            \
    --graph_rebuild_interval 5   \
    --wca_sigma   4.5            \
    --wca_eps     0.3            \
    --wca_lam     0.08           \
    --diff_steps  20             \
    --eta         1.0            \
    --device      "$DEVICE"      \
    --out         "$EXPLORE_OUT" \
    --seed        42             \
    $RESUME_FLAG

echo ""
echo "=== Done ==="
echo "Accepted conformations : $EXPLORE_OUT/candidates/"
echo "Summary JSON           : $EXPLORE_OUT/summary.json"
echo "CV coordinates (npy)   : $EXPLORE_OUT/cv_coords.npy"
echo "Stacked structures     : $EXPLORE_OUT/structures.pt"
echo "CV coverage plot       : $EXPLORE_OUT/cv_coverage.png"

# ---------------------------------------------------------------------------
# Optional post-processing (uncomment to run immediately after exploration)
# ---------------------------------------------------------------------------
# echo ""
# echo "=== Summarize exploration ==="
# python scripts/summarize_exploration.py \
#     --explore_dir "$EXPLORE_OUT" \
#     --shard       "$SHARD"
