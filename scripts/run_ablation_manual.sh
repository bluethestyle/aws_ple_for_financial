#!/bin/bash
# Manual ablation runner — runs scenarios sequentially without subprocess wrapper
# Usage: bash scripts/run_ablation_manual.sh

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

PHASE0="outputs/phase0"
RESULTS="outputs/ablation_results"
CONFIG="configs/santander/pipeline.yaml"

# Read training defaults from config
EPOCHS=5
BATCH=2048
LR=0.008
SEED=42

run_one() {
    local NAME="$1"
    local EXTRA_HPS="$2"
    local OUT="$RESULTS/$NAME"

    # Skip if already completed
    if [ -f "$OUT/eval_metrics.json" ]; then
        echo "[SKIP] $NAME (cached)"
        return
    fi

    mkdir -p "$OUT/model" "$OUT/logs"

    echo "[RUN ] $NAME ..."
    local START=$(date +%s)

    SM_CHANNEL_TRAIN="$PHASE0" \
    SM_OUTPUT_DATA_DIR="$OUT" \
    SM_MODEL_DIR="$OUT/model" \
    SM_HPS="{\"config\":\"$CONFIG\",\"epochs\":$EPOCHS,\"batch_size\":$BATCH,\"learning_rate\":$LR,\"seed\":$SEED,\"amp\":true,\"early_stopping_patience\":3,\"ablation_scenario\":\"$NAME\"$EXTRA_HPS}" \
    python -u containers/training/train.py \
        > "$OUT/logs/stdout.log" 2> "$OUT/logs/stderr.log"

    local END=$(date +%s)
    local ELAPSED=$((END - START))

    if [ -f "$OUT/eval_metrics.json" ]; then
        local AUC=$(python -c "import json; m=json.load(open('$OUT/eval_metrics.json')); print(m.get('auc','N/A'))" 2>/dev/null)
        echo "[OK  ] $NAME: AUC=$AUC (${ELAPSED}s)"
    else
        echo "[FAIL] $NAME (${ELAPSED}s) — check $OUT/logs/"
    fi
}

echo "============================================================"
echo "ABLATION: Feature + Expert Joint (manual runner)"
echo "============================================================"

# Baselines
run_one "joint_deepfm_base" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_one "joint_deepfm_all_features" ",\"shared_experts\":\"[\\\"deepfm\\\"]\""
run_one "joint_full" ""

# DeepFM + one expert (with matching features)
run_one "joint_deepfm+tda" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"perslay\\\"]\",\"removed_feature_groups\":\"[\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_one "joint_deepfm+temporal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_one "joint_deepfm+hgcn" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"hgcn\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_one "joint_deepfm+lightgcn" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"lightgcn\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_one "joint_deepfm+causal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"causal\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_one "joint_deepfm+ot" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_one "joint_deepfm+gmm" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"model_derived\\\"]\""
run_one "joint_deepfm+model_derived" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\"]\""

# Top-down: full minus one (expert + matching features)
run_one "joint_full-tda_perslay" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\"]\""
run_one "joint_full-temporal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"hmm_states\\\",\\\"mamba_temporal\\\"]\""
run_one "joint_full-hgcn_hierarchy" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"product_hierarchy\\\"]\""
run_one "joint_full-lightgcn_graph" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"graph_collaborative\\\"]\""
run_one "joint_full-causal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\""
run_one "joint_full-ot" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\"]\""

# ============================================================
# Phase 2: Task x Structure (PLE/adaTT 구조 효과)
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 2: Task x Structure Cross Ablation"
echo "============================================================"

TASKS_4='["has_nba","churn_signal","product_stability","nba_primary"]'
TASKS_8='["has_nba","churn_signal","product_stability","nba_primary","tenure_stage","spend_level","cross_sell_count","engagement_score"]'
TASKS_18=""  # empty = all tasks

for TIER in "tasks_4:$TASKS_4" "tasks_8:$TASKS_8" "tasks_18:$TASKS_18"; do
    TIER_NAME="${TIER%%:*}"
    TIER_TASKS="${TIER#*:}"

    TASK_HP=""
    if [ -n "$TIER_TASKS" ]; then
        TASK_HP=",\"active_tasks\":\"$TIER_TASKS\""
    fi

    run_one "struct_${TIER_NAME}_shared_bottom" ",\"use_ple\":\"false\",\"use_adatt\":\"false\"${TASK_HP}"
    run_one "struct_${TIER_NAME}_ple_only"      ",\"use_ple\":\"true\",\"use_adatt\":\"false\"${TASK_HP}"
    run_one "struct_${TIER_NAME}_adatt_only"     ",\"use_ple\":\"false\",\"use_adatt\":\"true\"${TASK_HP}"
    run_one "struct_${TIER_NAME}_full"           ",\"use_ple\":\"true\",\"use_adatt\":\"true\"${TASK_HP}"
done

echo ""
echo "============================================================"
echo "DONE. Results in $RESULTS/"
echo "============================================================"

# Summary
echo ""
echo "Summary:"
for d in "$RESULTS"/joint_*/; do
    name=$(basename "$d")
    if [ -f "$d/eval_metrics.json" ]; then
        auc=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); print(f'{m.get(\"auc\",\"N/A\")}')" 2>/dev/null)
        echo "  $name: AUC=$auc"
    fi
done
