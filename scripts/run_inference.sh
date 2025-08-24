#!/bin/bash
# Run ESV-RAG evaluation on FlashRAG benchmarks

CONFIG=${1:-configs/esv_mcts.yaml}
DATASET=${2:-nq}
MAX_EXAMPLES=${3:-500}

echo "=== ESV-RAG Evaluation ==="
echo "Config:      $CONFIG"
echo "Dataset:     $DATASET"
echo "Max samples: $MAX_EXAMPLES"
echo ""

python evaluate.py \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --max-examples "$MAX_EXAMPLES" \
    --output "results/${DATASET}_results.json"
