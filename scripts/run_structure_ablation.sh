#!/bin/bash
# Structure ablation runner — 6 scenarios x 20 epochs (tasks_18 only)
# Usage: nohup bash scripts/run_structure_ablation.sh > outputs/structure_ablation.log 2>&1 &
#
# Runs tasks_18 (all tasks) with 4 structure variants:
#   shared_bottom, ple_softmax, ple_sigmoid, adatt_only, ple_softmax+adatt, ple_sigmoid+adatt
#
# LR override for shared_bottom is read from config by train.py automatically.
# GradScaler parameters are read from config by trainer.py automatically.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

PHASE0="outputs/phase0"
RESULTS="outputs/ablation_results"
CONFIG="configs/santander/pipeline.yaml"

EPOCHS=20
BATCH=4096
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

    # Clean incomplete previous run
    rm -rf "$OUT"
    mkdir -p "$OUT/model" "$OUT/logs"

    echo "[RUN ] $NAME ($(date '+%H:%M:%S')) ..."
    local START=$(date +%s)

    SM_CHANNEL_TRAIN="$PHASE0" \
    SM_OUTPUT_DATA_DIR="$OUT" \
    SM_MODEL_DIR="$OUT/model" \
    SM_HPS="{\"config\":\"$CONFIG\",\"epochs\":$EPOCHS,\"batch_size\":$BATCH,\"learning_rate\":$LR,\"seed\":$SEED,\"amp\":true,\"early_stopping_patience\":20,\"ablation_scenario\":\"$NAME\"$EXTRA_HPS}" \
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
echo "STRUCTURE ABLATION: tasks_18 x 6 variants ($(date))"
echo "Epochs=$EPOCHS  Batch=$BATCH  LR=$LR (shared_bottom: config override)"
echo "============================================================"

# 1. Shared bottom (no PLE, no adaTT) — LR auto-lowered by train.py from config
run_one "struct_18_shared_bottom" ",\"use_ple\":\"false\",\"use_adatt\":\"false\""

# 2. PLE softmax only (default gate)
run_one "struct_18_ple_softmax" ",\"use_ple\":\"true\",\"use_adatt\":\"false\",\"gate_type\":\"softmax\""

# 3. PLE sigmoid only (NeurIPS 2024 gate)
run_one "struct_18_ple_sigmoid" ",\"use_ple\":\"true\",\"use_adatt\":\"false\",\"gate_type\":\"sigmoid\""

# 4. adaTT only (no PLE)
run_one "struct_18_adatt_only" ",\"use_ple\":\"false\",\"use_adatt\":\"true\""

# 5. PLE softmax + adaTT
run_one "struct_18_ple_softmax_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"softmax\""

# 6. PLE sigmoid + adaTT (full system)
run_one "struct_18_ple_sigmoid_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"sigmoid\""

echo ""
echo "============================================================"
echo "STRUCTURE ABLATION DONE. $(date)"
echo "============================================================"

# Structure summary
echo ""
echo "Structure Summary:"
for d in "$RESULTS"/struct_18_*/; do
    name=$(basename "$d")
    if [ -f "$d/eval_metrics.json" ]; then
        auc=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(fm.get('auc',m.get('aggregate_score','N/A')))" 2>/dev/null)
        vloss=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(f\"{fm.get('loss',0):.4f}\")" 2>/dev/null)
        echo "  $name: AUC=$auc val_loss=$vloss"
    else
        echo "  $name: INCOMPLETE"
    fi
done

# ============================================================
# PHASE 2: Joint Ablation Re-run (unified activations)
# ============================================================
echo ""
echo "============================================================"
echo "JOINT ABLATION RE-RUN: 17 scenarios x 5 epochs ($(date))"
echo "============================================================"

# Delete old joint results (ran with pre-unification activations)
echo "Cleaning old joint results..."
rm -rf "$RESULTS"/joint_*

JOINT_EPOCHS=5
JOINT_LR=0.008

run_joint() {
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
    SM_HPS="{\"config\":\"$CONFIG\",\"epochs\":$JOINT_EPOCHS,\"batch_size\":$BATCH,\"learning_rate\":$JOINT_LR,\"seed\":$SEED,\"amp\":false,\"early_stopping_patience\":$JOINT_EPOCHS,\"ablation_scenario\":\"$NAME\"$EXTRA_HPS}" \
    python -u containers/training/train.py \
        > "$OUT/logs/stdout.log" 2> "$OUT/logs/stderr.log"

    local END=$(date +%s)
    local ELAPSED=$((END - START))
    local MINS=$((ELAPSED / 60))

    if [ -f "$OUT/eval_metrics.json" ]; then
        echo "[OK  ] $NAME (${MINS}m)"
    else
        echo "[FAIL] $NAME (${MINS}m) — check $OUT/logs/"
    fi
}

# Baselines
run_joint "joint_deepfm_base" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_joint "joint_deepfm_all_features" ",\"shared_experts\":\"[\\\"deepfm\\\"]\""
run_joint "joint_full" ""

# DeepFM + one expert (with matching features)
run_joint "joint_deepfm+tda" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"perslay\\\"]\",\"removed_feature_groups\":\"[\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_joint "joint_deepfm+temporal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_joint "joint_deepfm+hgcn" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"hgcn\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_joint "joint_deepfm+lightgcn" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"lightgcn\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_joint "joint_deepfm+causal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"causal\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_joint "joint_deepfm+ot" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\",\\\"model_derived\\\"]\""
run_joint "joint_deepfm+gmm" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"model_derived\\\"]\""
run_joint "joint_deepfm+model_derived" ",\"shared_experts\":\"[\\\"deepfm\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\",\\\"hmm_states\\\",\\\"mamba_temporal\\\",\\\"product_hierarchy\\\",\\\"graph_collaborative\\\",\\\"gmm_clustering\\\"]\""

# Top-down: full minus one
run_joint "joint_full-tda_perslay" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"tda_global\\\",\\\"tda_local\\\"]\""
run_joint "joint_full-temporal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"hmm_states\\\",\\\"mamba_temporal\\\"]\""
run_joint "joint_full-hgcn_hierarchy" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"product_hierarchy\\\"]\""
run_joint "joint_full-lightgcn_graph" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"optimal_transport\\\"]\",\"removed_feature_groups\":\"[\\\"graph_collaborative\\\"]\""
run_joint "joint_full-causal" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"lightgcn\\\",\\\"optimal_transport\\\"]\""
run_joint "joint_full-ot" ",\"shared_experts\":\"[\\\"deepfm\\\",\\\"temporal_ensemble\\\",\\\"hgcn\\\",\\\"perslay\\\",\\\"causal\\\",\\\"lightgcn\\\"]\""

echo ""
echo "============================================================"
echo "ALL ABLATION DONE. $(date)"
echo "============================================================"

# Joint summary
echo ""
echo "Joint Summary:"
for d in "$RESULTS"/joint_*/; do
    name=$(basename "$d")
    if [ -f "$d/eval_metrics.json" ]; then
        echo "  $name: OK"
    else
        echo "  $name: INCOMPLETE"
    fi
done
