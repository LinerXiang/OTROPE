#!/usr/bin/env bash
# Align merged judge scores with labels and write combine_train.csv and combine_test.csv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# DATA must match the dataset used in the other preparation stages.
DATA=${DATA:-reward-bench-2}
ID_NAME=id
if [[ "$DATA" == chatbot_arena_conversations ]]; then ID_NAME=question_id; fi
python -m data_preparation.training_data_ready --target_filename judge_ --target_variable check \
  --base_path "real_data/${DATA}" --ground_truth "$DATA" \
  --id_name "$ID_NAME" --whether_train_col whether_train
