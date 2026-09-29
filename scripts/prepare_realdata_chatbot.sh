#!/usr/bin/env bash
# Prepare single-turn Arena pairs: GPT-4 as target, Koala-13B as behavior; ties count as losses.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATASET_NAME="lmsys/chatbot_arena_conversations"

DATASET_SHORT=${DATASET_NAME##*/}

OUTPUT_JSON="real_data/chatbot_arena_conversations/${DATASET_SHORT}.json"

python -m data_preparation.prepare_realdata_chatbot \
  --dataset-name "$DATASET_NAME" \
  --split train \
  --model-a-col model_a \
  --model-b-col model_b \
  --conversation-a-col conversation_a \
  --conversation-b-col conversation_b \
  --winner-col winner \
  --label-for-model-a model_a \
  --label-for-model-b model_b \
  --tie-mode zero \
  --target-model gpt-4 \
  --reference-model koala-13b \
  --mixed-models vicuna-13b alpaca-13b RWKV-4-Raven-14B stablelm-tuned-alpha-7b \
  --output-json "$OUTPUT_JSON"
