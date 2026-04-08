#!/bin/bash
# On-prem rehearsal: 4 scenarios x 2 epochs (dry-run)
# Validates that all scenarios run without errors before on-prem deployment.
# NOT for results — just for pipeline/config verification.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

PHASE0="outputs/phase0"
RESULTS="outputs/onprem_rehearsal"
CONFIG="configs/santander/pipeline.yaml"
EPOCHS=2
BATCH=4096
LR=0.008
SEED=42

run_one() {
    local NAME="$1"
    local EXTRA_HPS="$2"
    local OUT="$RESULTS/$NAME"

    rm -rf "$OUT"
    mkdir -p "$OUT/model" "$OUT/logs"

    echo "[RUN ] $NAME ($(date '+%H:%M:%S')) ..."
    local START=$(date +%s)

    SM_CHANNEL_TRAIN="$PHASE0" \
    SM_OUTPUT_DATA_DIR="$OUT" \
    SM_MODEL_DIR="$OUT/model" \
    SM_HPS="{\"config\":\"$CONFIG\",\"epochs\":$EPOCHS,\"batch_size\":$BATCH,\"learning_rate\":$LR,\"seed\":$SEED,\"amp\":false,\"early_stopping_patience\":$EPOCHS,\"ablation_scenario\":\"$NAME\"$EXTRA_HPS}" \
    python -u containers/training/train.py \
        > "$OUT/logs/stdout.log" 2> "$OUT/logs/stderr.log"

    local RC=$?
    local END=$(date +%s)
    local MINS=$(( (END - START) / 60 ))

    # Verify critical checkpoints
    local CHECKS=""
    # 1. Config applied?
    grep -q "AdaTT config:" "$OUT/logs/stdout.log" 2>/dev/null && CHECKS="${CHECKS}adatt_cfg " || true
    grep -q "FeatureRouter" "$OUT/logs/stdout.log" 2>/dev/null && CHECKS="${CHECKS}router " || true
    grep -q "UncertaintyWeighting" "$OUT/logs/stdout.log" 2>/dev/null && CHECKS="${CHECKS}uw " || true
    # 2. Training ran?
    local EPOCH_COUNT=$(grep -c "Epoch.*val_loss" "$OUT/logs/stdout.log" 2>/dev/null)
    # 3. No errors?
    local ERRORS=$(grep -ci "error\|traceback\|exception" "$OUT/logs/stderr.log" 2>/dev/null)
    # Exclude FutureWarning false positives
    local REAL_ERRORS=$(grep -i "error\|traceback\|exception" "$OUT/logs/stderr.log" 2>/dev/null | grep -v "FutureWarning\|UserWarning" | wc -l)

    if [ "$RC" -eq 0 ] && [ "$EPOCH_COUNT" -ge "$EPOCHS" ] && [ "$REAL_ERRORS" -eq 0 ]; then
        echo "[OK  ] $NAME: ${EPOCH_COUNT} epochs, ${MINS}m, checks=[${CHECKS}]"
    else
        echo "[FAIL] $NAME: rc=$RC, epochs=$EPOCH_COUNT, errors=$REAL_ERRORS (${MINS}m)"
        echo "  Last stderr:"
        grep -i "error\|traceback" "$OUT/logs/stderr.log" 2>/dev/null | grep -v "FutureWarning\|UserWarning" | tail -3
    fi
}

echo "============================================================"
echo "ON-PREM REHEARSAL: 4 scenarios x ${EPOCHS} epochs ($(date))"
echo "Purpose: verify pipeline, NOT for results"
echo "============================================================"

# Scenario 1: shared_bottom (baseline, simplest)
echo ""
echo "--- Scenario 1/4: shared_bottom (baseline) ---"
run_one "rehearsal_shared_bottom" ",\"use_ple\":\"false\",\"use_adatt\":\"false\""

# Scenario 2: ple_sigmoid (FeatureRouter + PLE 3-layer)
echo ""
echo "--- Scenario 2/4: ple_sigmoid (PLE only) ---"
run_one "rehearsal_ple_sigmoid" ",\"use_ple\":\"true\",\"use_adatt\":\"false\",\"gate_type\":\"sigmoid\""

# Scenario 3: ple_sigmoid_adatt (full system: uncertainty + adaTT sequential)
echo ""
echo "--- Scenario 3/4: ple_sigmoid_adatt (full system) ---"
run_one "rehearsal_ple_sigmoid_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"sigmoid\""

# Scenario 4: deepfm+temporal (expert subset + removed_feature_groups)
echo ""
echo "--- Scenario 4/4: deepfm+temporal (expert subset) ---"
run_one "rehearsal_deepfm_temporal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\"]\""

echo ""
echo "============================================================"
echo "REHEARSAL COMPLETE. $(date)"
echo "============================================================"
echo ""
echo "Summary:"
for d in "$RESULTS"/rehearsal_*/; do
    name=$(basename "$d")
    epoch_count=$(grep -c "Epoch.*val_loss" "$d/logs/stdout.log" 2>/dev/null)
    real_errors=$(grep -i "error\|traceback" "$d/logs/stderr.log" 2>/dev/null | grep -v "FutureWarning\|UserWarning" | wc -l)
    if [ "$epoch_count" -ge "$EPOCHS" ] && [ "$real_errors" -eq 0 ]; then
        echo "  [PASS] $name ($epoch_count epochs)"
    else
        echo "  [FAIL] $name (epochs=$epoch_count, errors=$real_errors)"
    fi
done

echo ""
echo "If all 4 PASS, the pipeline is ready for on-prem deployment."
echo "Check adaTT config: grep 'AdaTT config' outputs/onprem_rehearsal/rehearsal_ple_sigmoid_adatt/logs/stdout.log"
