#!/usr/bin/env bash
# Train the MoE router and compute OTROPE/baseline estimates for RewardBench-2.
# Requires prepared score CSVs and embeddings under real_data/<dataset>/.
# Override GPU_IDS, SIZES, N_REP, and JOBS_PER_GPU via environment variables.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA="reward-bench-2"
BASE_DIR="real_data/reward-bench-2"

TRAIN_SPLIT_PATH="${BASE_DIR}/combine_train.csv"
TEST_SPLIT_PATH="${BASE_DIR}/combine_test.csv"

TRAIN_BASE="${BASE_DIR}"
TEST_BASE="${BASE_DIR}"
OUTPUT_DIR="${BASE_DIR}/replication"

EMBEDDING_MODELS=(
  "bge-m3"
)

PREDICTION_COLS=(
  "p_weighted"
)

# Paper Appendix D.2: Claude 3.5 Sonnet uses n=1400 (Mixed uses n=1600).
read -r -a SAMPLE_SIZES <<< "${SIZES:-1400}"
N_REP=${N_REP:-10}
BASE_SEED=4128
read -r -a GPU_LIST <<< "${GPU_IDS:-0}"
JOBS_PER_GPU=${JOBS_PER_GPU:-1}      # Concurrent MoE/evaluation jobs per GPU.

if ! [[ "$N_REP" =~ ^[1-9][0-9]*$ && "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_REP and JOBS_PER_GPU must be positive integers" >&2
  exit 1
fi
for GPU_ID in "${GPU_LIST[@]}"; do
  if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS must contain space-separated numeric device indices" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}/logs"
RUN_LOG_DIR=$(mktemp -d "${OUTPUT_DIR}/logs/run.XXXXXX")
FAILED_JOBS=0
LOG_FILE="${RUN_LOG_DIR}/pipeline.log"
: > "$LOG_FILE"

_get_error_reason() {
  local job_log="$1"
  local exit_code="$2"
  if [[ ! -f "$job_log" ]]; then
    echo "exit_code=${exit_code} (no job log)"
    return
  fi
  if grep -qiE "OutOfMemoryError|out of memory" "$job_log" 2>/dev/null; then
    echo "CUDA OOM"
  elif grep -qiE "CUDA error" "$job_log" 2>/dev/null; then
    grep -iE "CUDA error" "$job_log" | tail -1 | cut -c1-120
  elif grep -qE "^[A-Za-z]+Error:" "$job_log" 2>/dev/null; then
    grep -E "^[A-Za-z]+Error:" "$job_log" | tail -1 | cut -c1-120
  else
    echo "exit_code=${exit_code} (unknown reason)"
  fi
}

run_job() {
  local GPU_ID="$1"
  local MODEL="$2"
  local SAMPLE_SIZE="$3"
  local REP="$4"
  local SEED="$5"

  local JOB_LOG="${RUN_LOG_DIR}/moe_job_${MODEL}_${SAMPLE_SIZE}_${REP}.log"
  : > "$JOB_LOG"

  # Inputs and MoE outputs for this sample size and replication.
  local TRAIN_EMB_PATH="${TRAIN_BASE}/embed_${MODEL}_train.safetensors"
  local TEST_EMB_PATH="${TEST_BASE}/embed_${MODEL}_test.safetensors"

  local IDS_PATH="${OUTPUT_DIR}/ids_${SAMPLE_SIZE}_${REP}.csv"
  local TRAIN_MOE_CSV="${OUTPUT_DIR}/train_MOE_${MODEL}_${SAMPLE_SIZE}_${REP}.csv"
  local TEST_MOE_CSV="${OUTPUT_DIR}/test_MOE_${MODEL}_${SAMPLE_SIZE}_${REP}.csv"
  local MOE_LOG="router_MOE_${MODEL}_${SAMPLE_SIZE}_${REP}.txt"

  # PCA and Sinkhorn/Adam settings passed to OTROPE.estimate_policy_value.
  # Appendix D.4: Claude 3.5 Sonnet uses 0.10; Mixed uses 0.08.
  local REG=0.10
  local MAX_ITER=1000
  local SINKHORN=2000
  local LR=0.01
  local PCA_DIM=128
  local PCA_MODE="pool"
  local START_TIME
  START_TIME=$(date +%s)

  echo "[$(date '+%F %T')] START GPU=${GPU_ID} MODEL=${MODEL} SAMPLE=${SAMPLE_SIZE} REP=${REP} SEED=${SEED}" \
    | tee -a "$LOG_FILE"

  # Train the router; save sampled IDs and source/target predictions.

  echo "[$(date '+%F %T')] MOE START GPU=${GPU_ID} MODEL=${MODEL} SAMPLE=${SAMPLE_SIZE} REP=${REP}" \
    | tee -a "$LOG_FILE"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m data_preparation.MOE \
    --embedding_model "${MODEL}" \
    --train_split_path "${TRAIN_SPLIT_PATH}" \
    --test_split_path "${TEST_SPLIT_PATH}" \
    --train_embedding_path "${TRAIN_EMB_PATH}" \
    --test_embedding_path "${TEST_EMB_PATH}" \
    --alpha_list 0.0 0.001 0.01 0.05 0.1 \
    --temperature_list 0.3 0.5 0.7 1.0 \
    --hidden_dim 64 32 \
    --dropout_p 0 \
    --lr 3e-4 \
    --weight_decay 0 \
    --batch_size 256 \
    --epoch_times 50 \
    --loss_type bce \
    --split_ratio 0.8 \
    --sample_size "${SAMPLE_SIZE}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_path "${MOE_LOG}" \
    --log_flag 0 \
    --replication "${REP}" \
    >> "$JOB_LOG" 2>&1

  echo "[$(date '+%F %T')] MOE END GPU=${GPU_ID} MODEL=${MODEL} SAMPLE=${SAMPLE_SIZE} REP=${REP}" \
    | tee -a "$LOG_FILE"

  if [[ ! -f "$IDS_PATH" ]]; then
    echo "[$(date '+%F %T')] ERROR missing ids file: ${IDS_PATH}" | tee -a "$LOG_FILE"
    return 1
  fi

  # Compute OT weights and OTROPE, DM, IS, DR, PPI, PPI++, and raw OT estimates.
  for PRED_COL in "${PREDICTION_COLS[@]}"; do
    echo "[$(date '+%F %T')] OT START GPU=${GPU_ID} MODEL=${MODEL} PRED=${PRED_COL} SAMPLE=${SAMPLE_SIZE} REP=${REP}" \
      | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m OTROPE.estimate_policy_value \
      --train_path "${BASE_DIR}" \
      --test_path "${BASE_DIR}" \
      --sampled_ids_path "${IDS_PATH}" \
      --train_embedding_file "embed_${MODEL}_train.safetensors" \
      --test_embedding_file "embed_${MODEL}_test.safetensors" \
      --train_pred_file "combine_train.csv" \
      --test_pred_file "combine_test.csv" \
      --prediction_col "${PRED_COL}" \
      --train_eval_csv "${TRAIN_MOE_CSV}" \
      --test_eval_csv "${TEST_MOE_CSV}" \
      --reg "${REG}" \
      --max_iter "${MAX_ITER}" \
      --sinkhorn_iters "${SINKHORN}" \
      --lr "${LR}" \
      --device cuda \
      --pca_mode "${PCA_MODE}" \
      --pca_dim "${PCA_DIM}" \
      --output_dir "${OUTPUT_DIR}" \
      --save_name "ot_${MODEL}_${PRED_COL}_${SAMPLE_SIZE}_${REP}" \
      --sample_size "${SAMPLE_SIZE}" \
      --seed "${SEED}" \
      --replication "${REP}" \
      >> "$JOB_LOG" 2>&1

    echo "[$(date '+%F %T')] OT END GPU=${GPU_ID} MODEL=${MODEL} PRED=${PRED_COL} SAMPLE=${SAMPLE_SIZE} REP=${REP}" \
      | tee -a "$LOG_FILE"
  done

  local END_TIME
  END_TIME=$(date +%s)
  local DURATION=$((END_TIME - START_TIME))

  echo "[$(date '+%F %T')] END GPU=${GPU_ID} MODEL=${MODEL} SAMPLE=${SAMPLE_SIZE} REP=${REP} duration=${DURATION}s" \
    | tee -a "$LOG_FILE"

  # Append this job's output to the run log; retain its individual log and predictions.
  cat "$JOB_LOG" >> "$LOG_FILE"
}


_NEXT_GPU=""
_NEXT_SLOT=""

# Record a completed job's status and release its GPU slot.
_handle_job_exit() {
  local GPU_ID="$1" SLOT="$2" PID="$3"
  local EXIT_CODE=0
  wait "$PID" 2>/dev/null || EXIT_CODE=$?
  if [[ $EXIT_CODE -ne 0 ]]; then
    FAILED_JOBS=$((FAILED_JOBS + 1))
    local JLOG_VAR="JLOG_${GPU_ID}_${SLOT}"
    local JOB_LOG="${!JLOG_VAR:-}"
    local REASON
    REASON=$(_get_error_reason "$JOB_LOG" "$EXIT_CODE")
    echo "[$(date '+%F %T')] ERROR: job on GPU=${GPU_ID} SLOT=${SLOT} failed [${REASON}]" \
      | tee -a "$LOG_FILE" >&2
    if [[ -f "$JOB_LOG" ]]; then
      echo "--- last 20 lines of job log ---" >> "$LOG_FILE"
      tail -20 "$JOB_LOG" >> "$LOG_FILE"
      echo "--- end job log ---" >> "$LOG_FILE"
    fi
  fi
  eval "PID_${GPU_ID}_${SLOT}=''"
}

# Reuse an empty or completed GPU slot, recording failures before reuse.
wait_for_any_slot() {
  while true; do
    for GPU_ID in "${GPU_LIST[@]}"; do
      for SLOT in $(seq 0 $((JOBS_PER_GPU - 1))); do
        local PID_VAR="PID_${GPU_ID}_${SLOT}"
        local PID="${!PID_VAR:-}"

        if [[ -z "$PID" ]]; then
          _NEXT_GPU="$GPU_ID"
          _NEXT_SLOT="$SLOT"
          return
        fi

        if ! kill -0 "$PID" 2>/dev/null; then
          _handle_job_exit "$GPU_ID" "$SLOT" "$PID"
          _NEXT_GPU="$GPU_ID"
          _NEXT_SLOT="$SLOT"
          return
        fi
      done
    done
    sleep 2
  done
}

# Track the process ID and log path for each GPU slot.
for GPU_ID in "${GPU_LIST[@]}"; do
  for SLOT in $(seq 0 $((JOBS_PER_GPU - 1))); do
    eval "PID_${GPU_ID}_${SLOT}=''"
    eval "JLOG_${GPU_ID}_${SLOT}=''"
  done
done

# Schedule one MoE/evaluation job per embedding model, sample size, and replication.
for MODEL in "${EMBEDDING_MODELS[@]}"; do
  for SAMPLE_SIZE in "${SAMPLE_SIZES[@]}"; do
    for REP in $(seq 1 ${N_REP}); do
      SEED=$((BASE_SEED + REP))

      wait_for_any_slot
      FREE_GPU="$_NEXT_GPU"
      FREE_SLOT="$_NEXT_SLOT"

      run_job "$FREE_GPU" "$MODEL" "$SAMPLE_SIZE" "$REP" "$SEED" &
      NEW_PID=$!
      eval "PID_${FREE_GPU}_${FREE_SLOT}=$NEW_PID"
      eval "JLOG_${FREE_GPU}_${FREE_SLOT}='${RUN_LOG_DIR}/moe_job_${MODEL}_${SAMPLE_SIZE}_${REP}.log'"
    done
  done
done

# Wait for remaining jobs and propagate failures to the caller.
for GPU_ID in "${GPU_LIST[@]}"; do
  for SLOT in $(seq 0 $((JOBS_PER_GPU - 1))); do
    PID_VAR="PID_${GPU_ID}_${SLOT}"
    PID="${!PID_VAR:-}"
    if [[ -n "$PID" ]]; then
      _handle_job_exit "$GPU_ID" "$SLOT" "$PID"
    fi
  done
done
if (( FAILED_JOBS > 0 )); then
  echo "${FAILED_JOBS} jobs failed. See ${LOG_FILE}" >&2
  exit 1
fi
echo "All jobs finished. Logs: ${LOG_FILE}"
