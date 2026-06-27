#!/usr/bin/env bash
# =============================================================================
# Run inference with the transferable SE(3) propagator.
#
# Produces:
#   1. Physics validation report (structural + thermodynamic + kinetic metrics)
#   2. Autoregressive Cα trajectory written as a multi-MODEL PDB
#
# Usage
# -----
#   bash scripts/infer_transfer.sh                   # validate + generate PDB
#   bash scripts/infer_transfer.sh --validate-only   # skip PDB generation
#   bash scripts/infer_transfer.sh --rollout-only    # skip validation
#   bash scripts/infer_transfer.sh --dry-run         # print commands only
#
# Inputs
# ------
#   CKPT       transferable propagator checkpoint (.pt)
#   SHARD      atlas-compatible protein shard (.pt)
#
# Outputs
# -------
#   VAL_JSON            physics validation report (JSON)
#   TRAJ_DIR/traj.pdb   autoregressive Cα trajectory (multi-MODEL PDB)
#   TRAJ_DIR/eval.json  four-metric quick eval
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CKPT="checkpoints/v4_3u7t_A.pt"
SHARD="data/atlas/3u7t_A.pt"

# Validation output
VAL_JSON="validation_$(basename "$SHARD" .pt)_T300.json"

# Rollout output directory
TRAJ_DIR="traj_$(basename "$SHARD" .pt)"

# Simulation settings
TAU_PS=2000            # physical lag per rollout step (ps)
TEMP_K=300             # inference temperature (K); try 300, 375, 450 to find best
VAL_STEPS=300          # validation rollout length (300 × 2 ns = 600 ns)
TRAJ_STEPS=200         # trajectory rollout length

# Denoising (DDPM vs DDIM)
# DDPM  (diff_steps=200, eta=1.0) — full stochastic, highest quality
# DDIM  (diff_steps=20,  eta=0.0) — 10× faster, slightly less diverse
DIFF_STEPS=20
ETA=1.0                # 1.0 = DDPM, 0.0 = DDIM deterministic

# WCA excluded-volume guidance (prevents steric clashes during denoising)
WCA_SIGMA=4.5          # Cα–Cα diameter (Å); 0 to disable
WCA_EPS=0.3            # well depth (kcal/mol)
WCA_LAM=0.05           # guidance step size

# Structural constraints
BOND_ITERS=5           # SHAKE pseudo-bond iterations; 0 to disable
MAX_UPDATE_NORM=3.0    # per-residue update norm clip

# Graph topology reuse (set > 1 for ~5× speedup with minimal quality loss)
GRAPH_REBUILD=5

# Noether projection: removes net linear/angular momentum per chain
# Recommended for long trajectories to prevent centre-of-mass drift
NOETHER=1

# ---------------------------------------------------------------------------

DRY_RUN=0; VALIDATE=1; ROLLOUT=1
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]]      && DRY_RUN=1
    [[ "$arg" == "--validate-only" ]] && ROLLOUT=0
    [[ "$arg" == "--rollout-only" ]]  && VALIDATE=0
done

run() { echo "+  $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

NOETHER_FLAG=""; [[ $NOETHER -eq 1 ]] && NOETHER_FLAG="--noether"

echo "======================================================================"
echo "  Transferable Propagator Inference"
echo "  Checkpoint : $CKPT"
echo "  Shard      : $SHARD"
echo "  tau_ps=$TAU_PS  temp_K=$TEMP_K  diff_steps=$DIFF_STEPS  eta=$ETA"
echo "======================================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1 — Physics validation
# ---------------------------------------------------------------------------
if [[ $VALIDATE -eq 1 ]]; then
    echo "=== Validation (${VAL_STEPS} steps × ${TAU_PS} ps = $(( VAL_STEPS * TAU_PS / 1000 )) ns) → $VAL_JSON ==="
    run python scripts/validate_physics.py \
        --checkpoint             "$CKPT" \
        --shard                  "$SHARD" \
        --steps                  "$VAL_STEPS" \
        --tau_ps                 "$TAU_PS" \
        --diff_steps             "$DIFF_STEPS" \
        --eta                    "$ETA" \
        --temp_K                 "$TEMP_K" \
        --wca_sigma              "$WCA_SIGMA" \
        --wca_eps                "$WCA_EPS" \
        --wca_lam                "$WCA_LAM" \
        --bond_constraint_iters  "$BOND_ITERS" \
        --max_update_norm        "$MAX_UPDATE_NORM" \
        $NOETHER_FLAG \
        --out                    "$VAL_JSON"

    echo ""
    echo "  Metric summary (target values):"
    echo "    rmsf_corr   > 0.90  (per-residue flexibility vs MD)"
    echo "    dist_js     < 0.005 (Cα pairwise distance distributions)"
    echo "    fes_js      < 0.50  (free-energy surface in PCA space)"
    echo "    relax_ratio 0.5–2.0 (model kinetics vs MD kinetics)"
    echo "    clash_count < 0.5   (steric clashes per frame)"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2 — Generate autoregressive Cα trajectory
# ---------------------------------------------------------------------------
if [[ $ROLLOUT -eq 1 ]]; then
    mkdir -p "$TRAJ_DIR"
    echo "=== Rollout (${TRAJ_STEPS} steps × ${TAU_PS} ps = $(( TRAJ_STEPS * TAU_PS / 1000 )) ns) → $TRAJ_DIR/ ==="

    # eval_transfer.py runs rollout and writes a four-metric JSON
    run python scripts/eval_transfer.py \
        --checkpoint             "$CKPT" \
        --shard                  "$SHARD" \
        --steps                  "$TRAJ_STEPS" \
        --tau_ps                 "$TAU_PS" \
        --diff_steps             "$DIFF_STEPS" \
        --eta                    "$ETA" \
        --temp_K                 "$TEMP_K" \
        --wca_sigma              "$WCA_SIGMA" \
        --wca_eps                "$WCA_EPS" \
        --wca_lam                "$WCA_LAM" \
        --graph_rebuild_interval "$GRAPH_REBUILD" \
        --bond_constraint_iters  "$BOND_ITERS" \
        --max_update_norm        "$MAX_UPDATE_NORM" \
        --out                    "$TRAJ_DIR/eval.json"

    # Write trajectory frames as a multi-MODEL PDB using Python one-liner
    # (eval_transfer does not write PDB directly; pipe through decoder)
    run python - <<PYEOF
import torch
from lsmd import geometry as g, transfer_eval as te, decoder as dec
import os

ckpt = torch.load("$CKPT", map_location="cpu", weights_only=False)
shard = torch.load("$SHARD", map_location="cpu", weights_only=False)
net, sched, norm = te.load_checkpoint(ckpt, device="cpu")

R0 = g.so3_exp(shard["R_aa"][0].float()) if "R_aa" in shard else shard["R"][0]
t0 = shard["t"][0].float()
N = t0.shape[0]

traj = te.rollout(
    net, sched, norm, R0, t0,
    shard["res_type"], shard["chain_id"], shard["res_index"],
    steps=$TRAJ_STEPS, tau_ps=$TAU_PS, k=ckpt["hparams"].get("k", 12),
    diff_steps=$DIFF_STEPS, eta=$ETA, temp_K=$TEMP_K,
    wca_sigma=$WCA_SIGMA, wca_eps=$WCA_EPS, wca_lam=$WCA_LAM,
    bond_constraint_iters=$BOND_ITERS, max_update_norm=$MAX_UPDATE_NORM,
    graph_rebuild_interval=$GRAPH_REBUILD,
    device="cpu",
)  # [steps+1, N, 3]

res_names = shard.get("seq", ["ALA"] * N)
pdb_path = os.path.join("$TRAJ_DIR", "traj.pdb")
with open(pdb_path, "w") as fh:
    for model_i, frame in enumerate(traj):
        fh.write(f"MODEL     {model_i + 1:4d}\n")
        for j, (pos, name) in enumerate(zip(frame.tolist(), res_names)):
            fh.write(
                f"ATOM  {j+1:5d}  CA  {name:3s} A{j+1:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C\n"
            )
        fh.write("ENDMDL\n")
print(f"Wrote {len(traj)}-frame trajectory → {pdb_path}")
PYEOF

    echo ""
    echo "  Visualise: pymol $TRAJ_DIR/traj.pdb"
fi

echo ""
echo "======================================================================"
echo "  Done."
[[ $VALIDATE -eq 1 ]] && echo "  Validation : $VAL_JSON"
[[ $ROLLOUT -eq 1  ]] && echo "  Trajectory : $TRAJ_DIR/traj.pdb"
echo ""
echo "  Tip: to sweep temperatures, re-run with TEMP_K=375 or TEMP_K=450"
echo "======================================================================"
