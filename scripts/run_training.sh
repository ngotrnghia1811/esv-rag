#!/bin/bash
# Full two-stage ESV-RAG training pipeline

CONFIG=${1:-configs/training.yaml}
BREAK_PATH=${2:-data/break/train.jsonl}
MONACO_PATH=${3:-data/monaco/train.json}

echo "=== ESV-RAG Training Pipeline ==="
echo "Config:      $CONFIG"
echo "BREAK data:  $BREAK_PATH"
echo "Monaco data: $MONACO_PATH"
echo ""

# Launch with Ray for distributed PPO
python -c "
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
from esv_rag.config import ESVConfig
from esv_rag.train import ESVRAGTrainer

config = ESVConfig.from_yaml('$CONFIG')
trainer = ESVRAGTrainer(config)
trainer.run(break_path='$BREAK_PATH', monaco_path='$MONACO_PATH')
"
