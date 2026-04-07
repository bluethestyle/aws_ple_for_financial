#!/bin/bash
# Resume remaining ablation: joint (skip completed) + structure (all 6)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

PHASE0="outputs/phase0"
RESULTS="outputs/ablation_results"
CONFIG="configs/santander/pipeline.yaml"
SEED=42
BATCH=4096

run_one() {
    local NAME="$1"
    local EXTRA_HPS="$2"
    local EPOCHS="$3"
    local LR="$4"
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
    SM_HPS="{\"config\":\"$CONFIG\",\"epochs\":$EPOCHS,\"batch_size\":$BATCH,\"learning_rate\":$LR,\"seed\":$SEED,\"amp\":true,\"early_stopping_patience\":$EPOCHS,\"ablation_scenario\":\"$NAME\"$EXTRA_HPS}" \
    python -u containers/training/train.py \
        > "$OUT/logs/stdout.log" 2> "$OUT/logs/stderr.log"

    local END=$(date +%s)
    local MINS=$(( (END - START) / 60 ))

    if [ -f "$OUT/eval_metrics.json" ]; then
        local AUC=$(python -c "import json; m=json.load(open('$OUT/eval_metrics.json')); fm=m.get('final_metrics',m); print(f\"{fm.get('auc','N/A'):.4f}\")" 2>/dev/null)
        echo "[OK  ] $NAME: AUC=$AUC (${MINS}m)"
    else
        echo "[FAIL] $NAME (${MINS}m)"
    fi
}

echo "============================================================"
echo "JOINT ABLATION (resume): $(date)"
echo "============================================================"

# Joint: 17 scenarios x 5ep (completed ones will be SKIPped)
JE=5; JL=0.008
run_one "joint_deepfm_base" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\"" $JE $JL
run_one "joint_deepfm_all_features" ",\"shared_experts\":\"[\\\"deepfm\\\"]\"" $JE $JL
run_one "joint_full" "" $JE $JL
run_one "joint_deepfm+tda" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"perslay\\\"]\",\"removed_feature_groups\":\"[\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\"]\"" $JE $JL
run_one "joint_deepfm+temporal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\"]\"" $JE $JL
run_one "joint_deepfm+hgcn" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"hgcn\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"graph_collaborative\\\"]\"" $JE $JL
run_one "joint_deepfm+lightgcn" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"lightgcn\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\"]\"" $JE $JL
run_one "joint_deepfm+causal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"causal\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"graph_collaborative\\\"]\"" $JE $JL
run_one "joint_deepfm+ot" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\"]\"" $JE $JL
run_one "joint_deepfm+gmm" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"model_derived\\\"]\"" $JE $JL
run_one "joint_deepfm+model_derived" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\"]\"" $JE $JL
run_one "joint_full-tda_perslay" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\"]\"" $JE $JL
run_one "joint_full-temporal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"hmm_states\\\",\\\"mamba_temporal\\\"]\"" $JE $JL
run_one "joint_full-hgcn_hierarchy" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\"" $JE $JL
run_one "joint_full-lightgcn_graph" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"graph_collaborative\\\"]\"" $JE $JL
run_one "joint_full-causal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\"" $JE $JL
run_one "joint_full-ot" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\"]\"" $JE $JL

echo ""
echo "============================================================"
echo "STRUCTURE ABLATION (re-run with fixed ranges): $(date)"
echo "============================================================"

# Clean incomplete structure results
rm -rf "$RESULTS"/struct_18_*

SE=20; SL=0.008
run_one "struct_18_shared_bottom" ",\"use_ple\":\"false\",\"use_adatt\":\"false\"" $SE $SL
run_one "struct_18_ple_softmax" ",\"use_ple\":\"true\",\"use_adatt\":\"false\",\"gate_type\":\"softmax\"" $SE $SL
run_one "struct_18_ple_sigmoid" ",\"use_ple\":\"true\",\"use_adatt\":\"false\",\"gate_type\":\"sigmoid\"" $SE $SL
run_one "struct_18_adatt_only" ",\"use_ple\":\"false\",\"use_adatt\":\"true\"" $SE $SL
run_one "struct_18_ple_softmax_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"softmax\"" $SE $SL
run_one "struct_18_ple_sigmoid_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"sigmoid\"" $SE $SL

echo ""
echo "============================================================"
echo "ALL ABLATION COMPLETE. $(date)"
echo "============================================================"

echo ""
echo "=== Structure Summary ==="
for d in "$RESULTS"/struct_18_*/; do
    name=$(basename "$d")
    if [ -f "$d/eval_metrics.json" ]; then
        auc=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(f\"{fm.get('auc','N/A'):.4f}\")" 2>/dev/null)
        f1=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(f\"{fm.get('f1_macro_avg',0):.4f}\")" 2>/dev/null)
        vloss=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(f\"{fm.get('loss',0):.4f}\")" 2>/dev/null)
        echo "  $name: AUC=$auc F1m=$f1 loss=$vloss"
    else
        echo "  $name: INCOMPLETE"
    fi
done

echo ""
echo "=== Joint Summary ==="
for d in "$RESULTS"/joint_*/; do
    name=$(basename "$d")
    if [ -f "$d/eval_metrics.json" ]; then
        auc=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(f\"{fm.get('auc','N/A'):.4f}\")" 2>/dev/null)
        echo "  $name: AUC=$auc"
    else
        echo "  $name: INCOMPLETE"
    fi
done
