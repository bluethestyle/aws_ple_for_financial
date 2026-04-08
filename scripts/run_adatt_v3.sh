#!/bin/bash
# adaTT v3: uncertainty weighting + adaTT sequential (2 scenarios)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

PHASE0="outputs/phase0"
RESULTS="outputs/ablation_results"
CONFIG="configs/santander/pipeline.yaml"
EPOCHS=10; BATCH=4096; LR=0.008; SEED=42

run_one() {
    local NAME="$1"; local EXTRA_HPS="$2"; local OUT="$RESULTS/$NAME"
    rm -rf "$OUT"; mkdir -p "$OUT/model" "$OUT/logs"
    echo "[RUN ] $NAME ($(date '+%H:%M:%S')) ..."
    local START=$(date +%s)
    SM_CHANNEL_TRAIN="$PHASE0" SM_OUTPUT_DATA_DIR="$OUT" SM_MODEL_DIR="$OUT/model" \
    SM_HPS="{\"config\":\"$CONFIG\",\"epochs\":$EPOCHS,\"batch_size\":$BATCH,\"learning_rate\":$LR,\"seed\":$SEED,\"amp\":false,\"early_stopping_patience\":$EPOCHS,\"ablation_scenario\":\"$NAME\"$EXTRA_HPS}" \
    python -u containers/training/train.py > "$OUT/logs/stdout.log" 2> "$OUT/logs/stderr.log"
    local MINS=$(( ($(date +%s) - START) / 60 ))
    if [ -f "$OUT/eval_metrics.json" ]; then
        local AUC=$(python -c "import json; m=json.load(open('$OUT/eval_metrics.json')); fm=m.get('final_metrics',m); print(f\"{fm.get('auc','N/A'):.4f}\")" 2>/dev/null)
        echo "[OK  ] $NAME: AUC=$AUC (${MINS}m)"
    else echo "[FAIL] $NAME (${MINS}m)"; fi
}

echo "============================================================"
echo "adaTT v3: uncertainty + adaTT sequential: $(date)"
echo "============================================================"

run_one "struct_18_ple_softmax_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"softmax\""
run_one "struct_18_ple_sigmoid_adatt" ",\"use_ple\":\"true\",\"use_adatt\":\"true\",\"gate_type\":\"sigmoid\""

echo ""
echo "Baselines: softmax=0.5684  sigmoid=0.5771  shared_bottom=0.5726"
echo "v2 results: softmax_adatt=0.5666  sigmoid_adatt=0.5605"
echo ""
echo "v3 results:"
for d in struct_18_ple_softmax_adatt struct_18_ple_sigmoid_adatt; do
    if [ -f "$RESULTS/$d/eval_metrics.json" ]; then
        auc=$(python -c "import json; m=json.load(open('$RESULTS/$d/eval_metrics.json')); fm=m.get('final_metrics',m); print(f\"{fm.get('auc','N/A'):.4f}\")" 2>/dev/null)
        echo "  $d: AUC=$auc"
    fi
done
echo "DONE. $(date)"
