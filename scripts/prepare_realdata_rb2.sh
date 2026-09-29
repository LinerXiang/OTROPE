#!/usr/bin/env bash
# Prepare RewardBench-2 pairs for Claude 3.5 Sonnet against the configured opponent pool.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATASET_NAME="allenai/reward-bench-2"

DATASET_SHORT=${DATASET_NAME##*/}

OUTPUT_JSON="real_data/reward-bench-2/${DATASET_SHORT}.json"

python -m data_preparation.prepare_realdata_rb2 \
  --dataset-name "$DATASET_NAME" \
  --split test \
  --prompt-col prompt \
  --chosen-col chosen \
  --rejected-col rejected \
  --models-col models \
  --subset-col subset \
  --question-id-col id \
  --target-models claude-3-5-sonnet-20241022 \
  --reference-models Qwen2.5-72B-Instruct Llama-3.1-70B-Instruct Qwen2.5-7B-Instruct \
  --mixed-models Llama-3.1-Tulu-3-8B Llama-3.1-8B-Instruct Mistral-7B-Instruct-v0.3 Llama-3.1-70B-Instruct Qwen2.5-72B-Instruct human gpt-4o-2024-08-06 Qwen2.5-7B-Instruct \
  --output-json "$OUTPUT_JSON"
