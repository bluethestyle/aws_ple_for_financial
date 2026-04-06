#!/bin/bash
# Joint ablation runner — 17 scenarios x 5 epochs
# Usage: nohup bash scripts/run_joint_ablation.sh > outputs/joint_ablation.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

PHASE0="outputs/phase0"
RESULTS="outputs/ablation_results"
CONFIG="configs/santander/pipeline.yaml"

EPOCHS=5
BATCH=4096
LR=0.008
SEED=42

run_one() {
    local NAME="$1"
    local EXTRA_HPS="$2"
    local OUT="$RESULTS/$NAME"

    if [ -f "$OUT/eval_metrics.json" ]; then
        echo "[SKIP] $NAME (cached)"
        return
    fi

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

    local END=$(date +%s)
    local ELAPSED=$((END - START))
    local MINS=$((ELAPSED / 60))

    if [ -f "$OUT/eval_metrics.json" ]; then
        local AUC=$(python -c "import json; m=json.load(open('$OUT/eval_metrics.json')); fm=m.get('final_metrics',m); print(fm.get('auc',m.get('aggregate_score','N/A')))" 2>/dev/null)
        local VLOSS=$(python -c "import json; m=json.load(open('$OUT/eval_metrics.json')); fm=m.get('final_metrics',m); print(round(fm.get('loss',0),4))" 2>/dev/null)
        echo "[OK  ] $NAME: AUC=$AUC val_loss=$VLOSS (${MINS}m)"
    else
        echo "[FAIL] $NAME (${MINS}m) — check $OUT/logs/"
    fi
}

echo "============================================================"
echo "JOINT ABLATION: 17 scenarios x $EPOCHS epochs ($(date))"
echo "Batch=$BATCH  LR=$LR  FP32  patience=$EPOCHS"
echo "============================================================"

# Delete old joint results
echo "Cleaning old joint results..."
rm -rf "$RESULTS"/joint_*

# Baselines
run_one "joint_deepfm_base" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_one "joint_deepfm_all_features" ",\"shared_experts\":\"[\\\"deepfm\\\"]\""
run_one "joint_full" ""

# DeepFM + one expert
run_one "joint_deepfm+tda" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"perslay\\\"]\",\"removed_feature_groups\":\"[\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\"]\""
run_one "joint_deepfm+temporal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\"]\""
run_one "joint_deepfm+hgcn" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"hgcn\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"graph_collaborative\\\"]\""
run_one "joint_deepfm+lightgcn" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"lightgcn\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\"]\""
run_one "joint_deepfm+causal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"causal\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"graph_collaborative\\\"]\""
run_one "joint_deepfm+ot" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\"]\""
run_one "joint_deepfm+gmm" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"model_derived\\\"]\""
run_one "joint_deepfm+model_derived" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\"]\""

# Top-down: full minus one
run_one "joint_full-tda_perslay" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\"]\""
run_one "joint_full-temporal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"hmm_states\\\",\\\"mamba_temporal\\\"]\""
run_one "joint_full-hgcn_hierarchy" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\""
run_one "joint_full-lightgcn_graph" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"graph_collaborative\\\"]\""
run_one "joint_full-causal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\""
run_one "joint_full-ot" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\"]\""

echo ""
echo "============================================================"
echo "JOINT ABLATION DONE. $(date)"
echo "============================================================"

echo ""
echo "Summary:"
for d in "$RESULTS"/joint_*/; do
    name=$(basename "$d")
    if [ -f "$d/eval_metrics.json" ]; then
        auc=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(fm.get('auc',m.get('aggregate_score','N/A')))" 2>/dev/null)
        vloss=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(round(fm.get('loss',0),4))" 2>/dev/null)
        echo "  $name: AUC=$auc val_loss=$vloss"
    else
        echo "  $name: INCOMPLETE"
    fi
done
