#!/bin/bash

set -euo pipefail

# Usage:
#   bash run_evalscope.sh --model MODEL_PATH [--port PORT] [--temp TEMPERATURE] [--tasks TASK1,TASK2]
#
# This script starts vLLM serve and then runs evalscope automatically.

PORT=8001
MODEL=~/models/minimax-m2.7-mxfp
TEMPERATURE=1.0
TOP_P=0.95
MAX_TOKENS=64000
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=102400
MAX_NUM_SEQS=1024
MAX_NUM_BATCHED_TOKENS=32768
SERVED_MODEL_NAME="minimax-m2.7"
TASKS=""
SKIP_SERVE="${SKIP_SERVE:-false}"
VLLM_PID=""
LOG_TAIL_PID=""

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

cleanup() {
  if [[ -n "${LOG_TAIL_PID}" ]] && kill -0 "${LOG_TAIL_PID}" 2>/dev/null; then
    kill "${LOG_TAIL_PID}" 2>/dev/null || true
  fi

  if [[ "${SKIP_SERVE}" == "true" ]]; then
    return
  fi

  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    CHILDREN=$(pgrep -P "${VLLM_PID}" || true)
    if [[ -n "${CHILDREN}" ]]; then
      kill -9 ${CHILDREN} 2>/dev/null || true
    fi
    kill -9 "${VLLM_PID}" 2>/dev/null || true
    return
  fi

  VLLM_PIDS=$(ps aux | grep -- "vllm serve" | grep -- "--port[ =]${PORT}" | grep -v grep | awk '{print $2}')
  if [[ -n "${VLLM_PIDS}" ]]; then
    for PID in ${VLLM_PIDS}; do
      CHILDREN=$(pgrep -P "${PID}" || true)
      if [[ -n "${CHILDREN}" ]]; then
        kill -9 ${CHILDREN} 2>/dev/null || true
      fi
      kill -9 "${PID}" 2>/dev/null || true
    done
  fi
}

trap cleanup EXIT

stop_log_tail() {
  if [[ -n "${LOG_TAIL_PID}" ]] && kill -0 "${LOG_TAIL_PID}" 2>/dev/null; then
    kill "${LOG_TAIL_PID}" 2>/dev/null || true
    LOG_TAIL_PID=""
  fi
}

trim_task_name() {
  local task_name="$1"
  task_name="${task_name#${task_name%%[![:space:]]*}}"
  task_name="${task_name%${task_name##*[![:space:]]}}"
  echo "${task_name}"
}

task_in_list() {
  local target_task="$1"
  shift
  local task_name
  for task_name in "$@"; do
    if [[ "${task_name}" == "${target_task}" ]]; then
      return 0
    fi
  done
  return 1
}

print_section_header() {
  echo "=== [${STEP_INDEX}/${TOTAL_STEPS}] $1 ===" | tee -a "$OUTPUT_FILE"
  STEP_INDEX=$((STEP_INDEX + 1))
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"; shift 2 ;;
    --model)
      MODEL="$2"; shift 2 ;;
    --temp)
      TEMPERATURE="$2"; shift 2 ;;
    --tasks)
      TASKS="$2"; shift 2 ;;
    --skip_serve|--skip-serve)
      SKIP_SERVE="true"; shift 1 ;;
    --tp)
      TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"; shift 2 ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

SKIP_SERVE="$(echo "${SKIP_SERVE}" | tr '[:upper:]' '[:lower:]')"

API_URL="http://127.0.0.1:${PORT}/v1"

if [[ "${SKIP_SERVE}" != "true" ]]; then
  echo "Starting vLLM serve on port ${PORT} ..."

  VLLM_CMD=(
    vllm serve "${MODEL}"
    --trust-remote-code
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --tool-call-parser minimax_m2
    --enable-auto-tool-choice
    --reasoning-parser minimax_m2
    --served-model-name "${SERVED_MODEL_NAME}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --enable-chunked-prefill
    --port "${PORT}"
  )

  "${VLLM_CMD[@]}" >/tmp/vllm_${PORT}.log 2>&1 &
  VLLM_PID=$!
  echo "vLLM launched. Log: /tmp/vllm_${PORT}.log"
  echo "vLLM PID: ${VLLM_PID}"
  echo "=== vLLM startup log (will stop after API wait ends) ==="
  tail -n +1 -f "/tmp/vllm_${PORT}.log" &
  LOG_TAIL_PID=$!
fi

# Wait until the API is ready
echo "Waiting for API at ${API_URL} ..."
for _ in $(seq 1 90); do
  if curl -sf "${API_URL}/models" -o /dev/null; then
    break
  fi
  if [[ "${SKIP_SERVE}" != "true" ]] && [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    stop_log_tail
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM exited before API became ready."
    echo "----- Last 80 lines of /tmp/vllm_${PORT}.log -----"
    tail -n 80 "/tmp/vllm_${PORT}.log" || true
    exit 1
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Port ${PORT} not ready, retrying in 20s..."
  sleep 20
done

stop_log_tail

if ! curl -sf "${API_URL}/models" -o /dev/null; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Timeout waiting for API at ${API_URL}."
  echo "----- Last 80 lines of /tmp/vllm_${PORT}.log -----"
  tail -n 80 "/tmp/vllm_${PORT}.log" || true
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] API is ready, starting evaluation."
MODEL_NORMALIZED="${MODEL%/}"
MODEL_NAME="${MODEL_NORMALIZED##*/}"
LOG_DIR="logs/${MODEL_NAME}"
mkdir -p "$LOG_DIR"
OUTPUT_FILE="${LOG_DIR}/eval_results_$(date +%Y%m%d_%H%M%S)_port${PORT}_temp${TEMPERATURE}.log"

# Generation config with thinking enabled (reasoning_effort=max)
GEN_CONFIG="{\"temperature\": ${TEMPERATURE}, \"top_p\": ${TOP_P}, \"n\": 1, \"extra_body\": {\"chat_template_kwargs\": {\"enable_thinking\": true, \"reasoning_effort\": \"max\"}}, \"max_tokens\": ${MAX_TOKENS}}"

SUPPORTED_TASKS=(gpqa_diamond aime25 gsm8k piqa)
SELECTED_TASKS=()
RUN_ALL="true"

if [[ -n "${TASKS}" ]]; then
  RUN_ALL="false"
  IFS=',' read -r -a REQUESTED_TASKS <<< "${TASKS}"
  for raw_task in "${REQUESTED_TASKS[@]}"; do
    task_name="$(trim_task_name "${raw_task}")"
    [[ -z "${task_name}" ]] && continue
    if ! task_in_list "${task_name}" "${SUPPORTED_TASKS[@]}"; then
      echo "Unsupported task: ${task_name}"
      echo "Supported tasks: ${SUPPORTED_TASKS[*]}"
      exit 1
    fi
    if ! task_in_list "${task_name}" "${SELECTED_TASKS[@]}"; then
      SELECTED_TASKS+=("${task_name}")
    fi
  done
else
  SELECTED_TASKS=("${SUPPORTED_TASKS[@]}")
fi

TOTAL_STEPS=${#SELECTED_TASKS[@]}
STEP_INDEX=1

echo "=== Evaluation started at $(date) ===" | tee "$OUTPUT_FILE"
echo "Model: $MODEL" | tee -a "$OUTPUT_FILE"
echo "Served model name: ${SERVED_MODEL_NAME}" | tee -a "$OUTPUT_FILE"
echo "API URL: $API_URL" | tee -a "$OUTPUT_FILE"
echo "Temperature: $TEMPERATURE / top_p: ${TOP_P}" | tee -a "$OUTPUT_FILE"
echo "Tasks: ${SELECTED_TASKS[*]}" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

for task in "${SELECTED_TASKS[@]}"; do
  echo "" | tee -a "$OUTPUT_FILE"
  print_section_header "${task}"

  EXTRA_DATASET_ARGS=""
  EXTRA_BATCH_SIZE="--eval-batch-size 256"

  if [[ "${task}" == "live_code_bench" ]]; then
    EXTRA_DATASET_ARGS="--dataset-args '{\"live_code_bench\": {\"subset_list\": [\"v6\"]}}'"
  fi

  evalscope eval \
    --model "${SERVED_MODEL_NAME}" \
    --eval-type openai_api \
    --api-key EMPTY \
    --timeout 36000 \
    --datasets "${task}" \
    ${EXTRA_DATASET_ARGS} \
    --generation-config "${GEN_CONFIG}" \
    --eval-batch-size 256 \
    --api-url "${API_URL}" 2>&1 | tee -a "$OUTPUT_FILE"
done

echo "" | tee -a "$OUTPUT_FILE"
echo "=== Evaluation finished at $(date) ===" | tee -a "$OUTPUT_FILE"
echo "Results saved to: $OUTPUT_FILE"

# Kill vLLM to free GPU
echo "Stopping vLLM on port ${PORT} to free GPU..." | tee -a "$OUTPUT_FILE"
VLLM_PIDS=$(ps aux | grep -- "vllm serve" | grep -- "--port[ =]${PORT}" | grep -v grep | awk '{print $2}')
if [[ -n "$VLLM_PIDS" ]]; then
  for PID in $VLLM_PIDS; do
    CHILDREN=$(pgrep -P $PID || true)
    if [[ -n "$CHILDREN" ]]; then
      kill -9 $CHILDREN 2>/dev/null || true
    fi
    kill -9 $PID 2>/dev/null || true
    echo "Killed vllm serve process: $PID" | tee -a "$OUTPUT_FILE"
  done
else
  echo "No vllm serve process found with --port ${PORT}." | tee -a "$OUTPUT_FILE"
fi
