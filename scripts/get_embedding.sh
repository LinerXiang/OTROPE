#!/usr/bin/env bash
# Embed candidate/opponent prompt-response pairs and save train/test embeddings with ID sidecars.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# DATA must match the dataset used in the other preparation stages.
DATA=${DATA:-reward-bench-2}
MODEL=${EMBEDDING_MODEL:-BAAI/bge-m3}
ID_NAME=id
if [[ "$DATA" == chatbot_arena_conversations ]]; then ID_NAME=question_id; fi
python -m data_preparation.get_embedding \
  --path "real_data/${DATA}" --input_file "${DATA}.json" \
  --embedding_model "$MODEL" --embed_type two_prompt_answer \
  --id_name "$ID_NAME" --candidate_name candidate_response \
  --opponent_name opponent_response --whether_train_col whether_train \
  --ground_truth_col label \
  --train_out_name "embed_${MODEL##*/}_train.safetensors" \
  --test_out_name "embed_${MODEL##*/}_test.safetensors"
