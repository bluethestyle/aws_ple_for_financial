#!/bin/bash
# Joint ablation: 15 scenarios, PLE softmax (default), 10 epochs
# Run directly without orchestrator to avoid orphan process issues

set -e
export PYTHONIOENCODING=utf-8
export PYTHONPATH=.

BASE_HP='{"config":"configs/santander/pipeline.yaml","epochs":10,"batch_size":5632,"learning_rate":0.0005,"seed":42,"amp":true,"early_stopping_patience":10,"warmup_epochs":3,"num_workers":0}'
PHASE0=outputs/phase0_v12
OUTDIR=outputs/ablation_v12

run_scenario() {
    local name=$1
    local extra_hp=$2
    local dir="$OUTDIR/$name"

    # Skip if already completed
    if [ -f "$dir/model/model.pth" ]; then
        echo "[SKIP] $name: already completed"
        return
    fi

    mkdir -p "$dir/model" "$dir/logs"

    # Merge base HP with extra HP
    local hp=$(python -c "
import json, sys
base = json.loads('$BASE_HP')
extra = json.loads('${extra_hp:-{}}')
base.update(extra)
print(json.dumps(base))
")

    echo "[START] $name ($(date +%H:%M:%S))"

    SM_CHANNEL_TRAIN=$PHASE0 \
    SM_OUTPUT_DATA_DIR=$dir \
    SM_MODEL_DIR=$dir/model \
    SM_HPS="$hp" \
    python -u containers/training/train.py \
        > "$dir/logs/stdout.log" \
        2> "$dir/logs/stderr.log"

    echo "[DONE]  $name ($(date +%H:%M:%S))"
    echo ""
}

echo "============================================================"
echo "Joint Ablation: 15 scenarios x 10 epochs (PLE softmax)"
echo "Started: $(date)"
echo "============================================================"
echo ""

# Baselines
run_scenario "joint_full" '{}'
run_scenario "joint_base_only" '{"removed_feature_groups":"[\"tda_global\",\"tda_local\",\"hmm_states\",\"mamba_temporal\",\"product_hierarchy\",\"merchant_hierarchy\",\"graph_collaborative\",\"gmm_clustering\",\"model_derived\",\"txn_behavior\",\"derived_temporal\"]"}'

# Bottom-up: DeepFM + one expert
run_scenario "joint_deepfm_base" '{"shared_experts":"[\"deepfm\"]"}'
run_scenario "joint_deepfm+temporal" '{"shared_experts":"[\"deepfm\",\"temporal_ensemble\"]"}'
run_scenario "joint_deepfm+hgcn" '{"shared_experts":"[\"deepfm\",\"hgcn\"]"}'
run_scenario "joint_deepfm+tda" '{"shared_experts":"[\"deepfm\",\"perslay\"]"}'
run_scenario "joint_deepfm+lightgcn" '{"shared_experts":"[\"deepfm\",\"lightgcn\"]"}'
run_scenario "joint_deepfm+causal" '{"shared_experts":"[\"deepfm\",\"causal\"]"}'
run_scenario "joint_deepfm+ot" '{"shared_experts":"[\"deepfm\",\"optimal_transport\"]"}'

# Top-down: Full minus one expert
run_scenario "joint_full-temporal" '{"removed_experts":"[\"temporal_ensemble\"]"}'
run_scenario "joint_full-hgcn" '{"removed_experts":"[\"hgcn\"]"}'
run_scenario "joint_full-tda" '{"removed_experts":"[\"perslay\"]"}'
run_scenario "joint_full-lightgcn" '{"removed_experts":"[\"lightgcn\"]"}'
run_scenario "joint_full-causal" '{"removed_experts":"[\"causal\"]"}'
run_scenario "joint_full-ot" '{"removed_experts":"[\"optimal_transport\"]"}'

echo "============================================================"
echo "Joint Ablation COMPLETE: $(date)"
echo "============================================================"
