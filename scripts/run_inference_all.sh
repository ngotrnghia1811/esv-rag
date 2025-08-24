#!/bin/bash
# Evaluate ESV-RAG on all 12 FlashRAG benchmark datasets

CONFIG=${1:-configs/esv_mcts.yaml}
MAX_EXAMPLES=${2:-}  # leave empty to run full test sets

CMD="python evaluate.py --config $CONFIG --all"
if [ -n "$MAX_EXAMPLES" ]; then
    CMD="$CMD --max-examples $MAX_EXAMPLES"
fi
CMD="$CMD --output results/all_datasets_results.json"

echo "=== ESV-RAG Full Benchmark Evaluation ==="
echo "Config: $CONFIG"
echo "Running: $CMD"
echo ""

mkdir -p results
$CMD
