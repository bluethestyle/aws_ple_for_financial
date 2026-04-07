#!/bin/bash
# Structure ablation ONLY (no joint) — 6 scenarios x 20 epochs
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
        echo "[FAIL] $NAME (${MINS}m)"
    fi
}

echo "============================================================"
echo "STRUCTURE ABLATION (re-run with fixed ranges): $(date)"
echo "============================================================"

run_one "struct_18_shared_bottom" ",\"use_ple\":\"false\",\"use_adatt\":\"false\""
run_one "struct_18_ple_softmax" ",\"use_ple\":\"true\",\"use_adatt\":\"false\",\"gate_type\":\"softmax\""
run_one "struct_18_ple_sigmoid" ",\"use_ple\":\"true\",\"use_adatt\":\"false\",\"gate_type\":\"sigmoid\""
run_one "struct_18_adatt_only" ",\"use_ple\":\"false\",\"use_adatt\":\"true\""
run_one "struct_18_ple_softmax_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"softmax\""
run_one "struct_18_ple_sigmoid_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"sigmoid\""

echo ""
echo "Structure Summary:"
for d in "$RESULTS"/struct_18_*/; do
    name=$(basename "$d")
    if [ -f "$d/eval_metrics.json" ]; then
        auc=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(fm.get('auc',m.get('aggregate_score','N/A')))" 2>/dev/null)
        vloss=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(f\"{fm.get('loss',0):.4f}\")" 2>/dev/null)
        f1=$(python -c "import json; m=json.load(open('$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(f\"{fm.get('f1_macro_avg',0):.4f}\")" 2>/dev/null)
        echo "  $name: AUC=$auc F1m=$f1 loss=$vloss"
    else
        echo "  $name: INCOMPLETE"
    fi
done
echo "DONE. $(date)"
