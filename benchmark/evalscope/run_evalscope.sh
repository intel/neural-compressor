#!/bin/bash

set -euo pipefail

# Usage:
#  bash run_evalscope.sh --model MODEL_PATH [--port PORT] [--tasks TASK1,TASK2]


PORT=8009
MODEL=/workspace/models/deepseek-ai/DeepSeek-V4-Flash
SCHEME="mxfp4"
TEMPERATURE=1.0
KV_CACHE_DTYPE="fp8"
BLOCK_SIZE=256
TENSOR_PARALLEL_SIZE=2
OUTPUT_FILE=""
TASKS=""
SKIP_SERVE="${SKIP_SERVE:-false}"
STOP_SERVE="${STOP_SERVE:-true}"
VLLM_PID=""
LOG_TAIL_PID=""

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

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
    --scheme)
      SCHEME="$2"; shift 2 ;;
    --temp)
      TEMPERATURE="$2"; shift 2 ;;
    --tasks)
      TASKS="$2"; shift 2 ;;
    --skip-serve)
      SKIP_SERVE="$2"; shift 2 ;;
    --stop-serve)
      STOP_SERVE="$2"; shift 2 ;;
    --tp)
      TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --kv-cache-dtype)
      KV_CACHE_DTYPE="$2"; shift 2 ;;
    --block-size)
      BLOCK_SIZE="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

SKIP_SERVE="$(echo "${SKIP_SERVE}" | tr '[:upper:]' '[:lower:]')"
STOP_SERVE="$(echo "${STOP_SERVE}" | tr '[:upper:]' '[:lower:]')"

API_URL="http://127.0.0.1:${PORT}/v1"

# All logs are stored under the local logs/ directory.
mkdir -p logs
VLLM_LOG="logs/vllm_${PORT}.log"
OUTPUT_FILE="logs/eval_results_$(date +%Y%m%d_%H%M%S)_port${PORT}_temp${TEMPERATURE}.log"

start_vllm_serve() {
  if [[ "${SKIP_SERVE}" == "true" ]]; then
    return
  fi
  echo "Starting vLLM serve on port ${PORT} ..."
  bash "${SCRIPT_DIR}/start_vllm_serve.sh" \
    --model "${MODEL}" \
    --port "${PORT}" \
    --scheme "${SCHEME}" \
    --tp "${TENSOR_PARALLEL_SIZE}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --block-size "${BLOCK_SIZE}" \
    >"${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  echo "vLLM launched. Log: ${VLLM_LOG}"
  echo "vLLM PID: ${VLLM_PID}"
  echo "=== vLLM startup log (will stop after API wait ends) ==="
  tail -n +1 -f "${VLLM_LOG}" &
  LOG_TAIL_PID=$!
}

check_vllm_status() {
  # Wait until the API is ready
  echo "Waiting for API at ${API_URL} ..."
  for _ in $(seq 1 90); do
    if curl -sf "${API_URL}/models" -o /dev/null; then
      stop_log_tail
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] API is ready, starting evaluation."
      return 0
    fi
    if [[ "${SKIP_SERVE}" != "true" ]] && [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM exited before API became ready."
      break
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Port ${PORT} not ready, retrying in 20s..."
    sleep 20
  done

  stop_log_tail
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to reach API at ${API_URL}."
  echo "----- Last 80 lines of ${VLLM_LOG} -----"
  tail -n 80 "${VLLM_LOG}" || true
  exit 1
}

run_evalscope() {
  # Determine which tasks to evaluate based on user input or default tasks
  local DEFAULT_TASKS=(mmlu_pro gpqa_diamond aime26 math_500)
  SELECTED_TASKS=()
  if [[ -n "${TASKS}" ]]; then
    IFS=',' read -r -a SELECTED_TASKS <<< "${TASKS}"
  else
    SELECTED_TASKS=("${DEFAULT_TASKS[@]}")
  fi
  TOTAL_STEPS=${#SELECTED_TASKS[@]}
  STEP_INDEX=1

  echo "=== Evaluation started at $(date) ===" | tee "$OUTPUT_FILE"
  echo "Model: $MODEL" | tee -a "$OUTPUT_FILE"
  echo "API URL: $API_URL" | tee -a "$OUTPUT_FILE"
  echo "Temperature: $TEMPERATURE" | tee -a "$OUTPUT_FILE"
  echo "Selected tasks: ${SELECTED_TASKS[*]}" | tee -a "$OUTPUT_FILE"

  local task
  for task in "${SELECTED_TASKS[@]}"; do
    echo "" | tee -a "$OUTPUT_FILE"
    print_section_header "${task}"
    evalscope eval \
      --model "$MODEL" \
      --eval-type openai_api \
      --api-key EMPTY \
      --timeout 36000 \
      --datasets "${task}" \
      --generation-config '{"temperature":'"$TEMPERATURE"', "top_p":0.95, "n":1, "extra_body": {"chat_template_kwargs": { "enable_thinking": true, "reasoning_effort": "max"}},"max_tokens":64000}' \
      --eval-batch-size 512 --api-url "$API_URL" 2>&1 | tee -a "$OUTPUT_FILE"
  done

  echo -e "\n" | tee -a "$OUTPUT_FILE"
  echo "=== Evaluation finished at $(date) ===" | tee -a "$OUTPUT_FILE"
  echo "Results saved to: $OUTPUT_FILE"
}

stop_vllm_serve() {
  # Kill the vllm serve process (and its child workers) bound to this port to free GPU.
  echo "Stopping process on port ${PORT} to free GPU..." | tee -a "$OUTPUT_FILE"
  local pids
  pids=$(pgrep -f "vllm serve.*--port[ =]${PORT}" || true)
  if [[ -z "$pids" ]]; then
    echo "No vllm serve process found with --port ${PORT}." | tee -a "$OUTPUT_FILE"
    return
  fi
  echo "Found vllm serve process(es) with --port ${PORT}: $pids" | tee -a "$OUTPUT_FILE"
  local pid
  for pid in $pids; do
    pkill -9 -P "$pid" 2>/dev/null || true
    kill -9 "$pid" 2>/dev/null || true
  done
}

# Ensure the log tail and vLLM server are stopped even if the script exits early.
cleanup() {
  stop_log_tail
  if [[ "${STOP_SERVE}" == "true" ]]; then
    stop_vllm_serve
  fi
}
trap cleanup EXIT

start_vllm_serve
check_vllm_status
run_evalscope
