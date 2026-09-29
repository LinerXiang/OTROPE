#!/usr/bin/env bash
# Generate one judge verdict per pair, then merge the GPU shards for each model.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# DATA must match the dataset used in the other preparation stages.
DATA=${DATA:-reward-bench-2}
read -r -a GPUS <<< "${GPU_IDS:-0}"
OUTPUT_DIR="real_data/${DATA}"
MODELS=(microsoft/Phi-4-mini-instruct google/gemma-7b-it deepseek-ai/deepseek-llm-7b-chat meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-14B-Instruct)
mkdir -p "$OUTPUT_DIR"
for MODEL in "${MODELS[@]}"; do
  PIDS=()
  for INDEX in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$INDEX]}" python -m data_preparation.gen_eval \
      --model_path "$MODEL" --input_json "${OUTPUT_DIR}/${DATA}.json" \
      --local_index "$INDEX" --num_shards "${#GPUS[@]}" \
      --output_dir "$OUTPUT_DIR" --num_gen 1 --gpu_usage "${GPU_USAGE:-0.95}" &
    PIDS+=("$!")
  done
  FAILED=0
  for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=1
  done
  if (( FAILED )); then
    echo "Judge shard failed for ${MODEL}; inspect output before rerunning." >&2
    exit 1
  fi
  python -m data_preparation.merge --base_path "${OUTPUT_DIR}/judge_${MODEL##*/}_" \
    --output_dir "${OUTPUT_DIR}/judge_${MODEL##*/}.json" --num_datasets "${#GPUS[@]}"
  # Archive shards so score aggregation reads only merged judge files.
  mkdir -p "${OUTPUT_DIR}/shards"
  for INDEX in "${!GPUS[@]}"; do
    mv "${OUTPUT_DIR}/judge_${MODEL##*/}_${INDEX}.json" "${OUTPUT_DIR}/shards/"
  done
done
