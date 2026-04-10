#!/bin/bash
# =============================================================================
# Local Ablation: 14 tasks, 10 epochs (warmup=3), no Docker, num_workers=0
# Delta measurement only — same epoch count across all scenarios for fair comparison.
# Full training (20+ epochs) is done separately for the distillation teacher.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHASE0="outputs/phase0_v3"
RESULTS="outputs/ablation_v3"
CONFIG="configs/santander/pipeline.yaml"
EPOCHS=10
WARMUP=3
BATCH=4096
LR=0.0005
SEED=42
AMP=true
PATIENCE=$EPOCHS  # no early stop for ablation — run full epochs for fair comparison

mkdir -p "$RESULTS"

# ---------------------------------------------------------------------------
# Helper: run one scenario
# ---------------------------------------------------------------------------
run_one() {
    local NAME="$1"
    local EXTRA_HPS="$2"
    local OUT="$RESULTS/$NAME"

    # Skip if already completed
    if [ -f "$OUT/model/model.pth" ]; then
        echo "[SKIP] $NAME: model.pth already exists"
        return 0
    fi

    rm -rf "$OUT"
    mkdir -p "$OUT/model" "$OUT/logs"

    echo "[RUN ] $NAME ($(date '+%H:%M:%S')) ..."
    local START=$(date +%s)

    SM_CHANNEL_TRAIN="$PHASE0" \
    SM_OUTPUT_DATA_DIR="$OUT" \
    SM_MODEL_DIR="$OUT/model" \
    SM_HPS="{\"config\":\"$CONFIG\",\"epochs\":$EPOCHS,\"batch_size\":$BATCH,\"learning_rate\":$LR,\"seed\":$SEED,\"amp\":$AMP,\"early_stopping_patience\":$PATIENCE,\"warmup_epochs\":$WARMUP,\"num_workers\":0,\"ablation_scenario\":\"$NAME\"$EXTRA_HPS}" \
    python -u containers/training/train.py \
        > "$OUT/logs/stdout.log" 2> "$OUT/logs/stderr.log"

    local RC=$?
    local END=$(date +%s)
    local MINS=$(( (END - START) / 60 ))

    # Verify
    local EPOCH_COUNT=$(grep -c "Epoch.*val_loss" "$OUT/logs/stdout.log" 2>/dev/null)
    local REAL_ERRORS=$(grep -i "error\|traceback\|exception" "$OUT/logs/stderr.log" 2>/dev/null | grep -v "FutureWarning\|UserWarning\|SageMakerTracker" | wc -l)

    # Extract key metrics
    local AVG_AUC=$(grep "avg_auc" "$OUT/logs/stdout.log" | tail -1 | grep -oP "avg_auc=\K[0-9.]+")
    local AVG_F1=$(grep "avg_f1_macro" "$OUT/logs/stdout.log" | tail -1 | grep -oP "avg_f1_macro=\K[0-9.]+")

    if [ "$RC" -eq 0 ] && [ "$EPOCH_COUNT" -ge "$EPOCHS" ]; then
        echo "[OK  ] $NAME: ${EPOCH_COUNT} epochs, ${MINS}m, AUC=${AVG_AUC:-?}, F1=${AVG_F1:-?}"
    else
        echo "[FAIL] $NAME: rc=$RC, epochs=$EPOCH_COUNT, errors=$REAL_ERRORS (${MINS}m)"
        grep -i "error\|traceback" "$OUT/logs/stderr.log" 2>/dev/null | grep -v "FutureWarning\|UserWarning" | tail -3
    fi
}

# ---------------------------------------------------------------------------
# DRY RUN: 1 epoch to verify scenario settings before full ablation
# ---------------------------------------------------------------------------
dry_run() {
    local NAME="$1"
    local EXTRA_HPS="$2"
    local OUT="$RESULTS/dryrun_$NAME"

    rm -rf "$OUT"
    mkdir -p "$OUT/model" "$OUT/logs"

    echo "[DRY ] $NAME ..."

    SM_CHANNEL_TRAIN="$PHASE0" \
    SM_OUTPUT_DATA_DIR="$OUT" \
    SM_MODEL_DIR="$OUT/model" \
    SM_HPS="{\"config\":\"$CONFIG\",\"epochs\":1,\"batch_size\":$BATCH,\"learning_rate\":$LR,\"seed\":$SEED,\"amp\":$AMP,\"early_stopping_patience\":1,\"warmup_epochs\":0,\"num_workers\":0,\"ablation_scenario\":\"$NAME\"$EXTRA_HPS}" \
    python -u containers/training/train.py \
        > "$OUT/logs/stdout.log" 2> "$OUT/logs/stderr.log"

    local RC=$?

    # Verify scenario was actually applied
    local SHARED=$(grep "Expert Basket:" "$OUT/logs/stdout.log" | head -1)
    local ROUTING=$(grep "Feature routing:" "$OUT/logs/stdout.log" | head -1)
    local TASKS=$(grep "Tasks:" "$OUT/logs/stdout.log" | head -1)
    local EPOCH_OK=$(grep -c "Epoch.*val_loss" "$OUT/logs/stdout.log" 2>/dev/null)

    if [ "$RC" -eq 0 ] && [ "$EPOCH_OK" -ge 1 ]; then
        echo "  [OK] $SHARED"
        echo "       $ROUTING"
    else
        echo "  [FAIL] rc=$RC"
        grep -i "error\|traceback" "$OUT/logs/stderr.log" 2>/dev/null | grep -v "FutureWarning" | tail -2
        return 1
    fi

    rm -rf "$OUT"
    return 0
}

# ==========================================================================
# SCENARIOS
# ==========================================================================

echo "============================================================"
echo "LOCAL ABLATION: ${EPOCHS} epochs, $(date)"
echo "Phase0: $PHASE0"
echo "Config: $CONFIG"
echo "============================================================"

# --- Phase 2: Structure ablation (6 scenarios) ---
echo ""
echo "=== Phase 2: Structure Ablation (6 scenarios) ==="

# S1: shared_bottom (no PLE, no adaTT)
run_one "struct_14_shared_bottom" ",\"use_ple\":\"false\",\"use_adatt\":\"false\""

# S2: PLE softmax (no adaTT)
run_one "struct_14_ple_softmax" ",\"use_ple\":\"true\",\"use_adatt\":\"false\",\"gate_type\":\"softmax\""

# S3: PLE sigmoid (no adaTT)
run_one "struct_14_ple_sigmoid" ",\"use_ple\":\"true\",\"use_adatt\":\"false\",\"gate_type\":\"sigmoid\""

# S4: PLE softmax + adaTT
run_one "struct_14_ple_softmax_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"softmax\""

# S5: PLE sigmoid + adaTT (full system)
run_one "struct_14_ple_sigmoid_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"sigmoid\""

# S6: adaTT only (no PLE layering)
run_one "struct_14_adatt_only" ",\"use_ple\":\"false\",\"use_adatt\":\"true\""

# --- Phase 1: Feature + Expert joint ablation (auto-generated scenarios) ---
echo ""
echo "=== Phase 1: Feature + Expert Joint Ablation ==="

# Full system (all features, all experts) — baseline
run_one "joint_full" ""

# Base features only (demographics + product_holdings)
run_one "joint_base_only" ",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"merchant_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\",\\\"txn_behavior\\\",\\\"derived_temporal\\\"]\""

# DeepFM base (only DeepFM expert)
run_one "joint_deepfm_base" ",\"shared_experts\":\"[\\\"deepfm\\\"]\""

# DeepFM + each advanced expert
run_one "joint_deepfm+temporal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\"]\""
run_one "joint_deepfm+hgcn" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"hgcn\\\"]\""
run_one "joint_deepfm+tda" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"perslay\\\"]\""
run_one "joint_deepfm+lightgcn" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"lightgcn\\\"]\""
run_one "joint_deepfm+causal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"causal\\\"]\""
run_one "joint_deepfm+ot" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"optimal_transport\\\"]\""
run_one "joint_deepfm+gmm" ",\"shared_experts\":\"[\\\"deepfm\\\"]\"" # GMM is a feature, not expert

# Full minus one expert
run_one "joint_full-temporal" ",\"removed_experts\":\"[\\\"temporal_ensemble\\\"]\""
run_one "joint_full-hgcn" ",\"removed_experts\":\"[\\\"hgcn\\\"]\""
run_one "joint_full-tda" ",\"removed_experts\":\"[\\\"perslay\\\"]\""
run_one "joint_full-lightgcn" ",\"removed_experts\":\"[\\\"lightgcn\\\"]\""
run_one "joint_full-causal" ",\"removed_experts\":\"[\\\"causal\\\"]\""
run_one "joint_full-ot" ",\"removed_experts\":\"[\\\"optimal_transport\\\"]\""

# All features, all experts (full system) — same as joint_full but
# with all advanced feature groups (explicit, for comparison)
run_one "joint_deepfm_all_features" ",\"shared_experts\":\"[\\\"deepfm\\\"]\""

echo ""
echo "============================================================"
echo "ABLATION COMPLETE: $(date)"
echo "Results: $RESULTS/"
echo "============================================================"

# Summary table
echo ""
echo "=== Summary ==="
for d in "$RESULTS"/*/; do
    name=$(basename "$d")
    [[ "$name" == dryrun_* ]] && continue
    if [ -f "$d/logs/stdout.log" ]; then
        auc=$(grep "avg_auc" "$d/logs/stdout.log" | tail -1 | grep -oP "avg_auc=\K[0-9.]+")
        f1=$(grep "avg_f1_macro" "$d/logs/stdout.log" | tail -1 | grep -oP "avg_f1_macro=\K[0-9.]+")
        echo "  $name: AUC=${auc:-?} F1=${f1:-?}"
    fi
done
