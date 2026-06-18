#!/bin/bash

set -euo pipefail

# Usage:
#   bash run_evalscope.sh --model MODEL_PATH [--port PORT] [--temp TEMPERATURE] [--tasks TASK1,TASK2]
#
# This script can start vLLM serve and then run evalscope automatically.

PORT=8009
MODEL=/workspace/models/deepseek-ai/DeepSeek-V4-Flash
TEMPERATURE=0
KV_CACHE_DTYPE="fp8"
BLOCK_SIZE=256
TENSOR_PARALLEL_SIZE=2
SAFETENSORS_FAST_GPU="1"
TRUST_REMOTE_CODE="true"
NO_ENABLE_FLASHINFER_AUTOTUNE="true"
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

  # Kill the process listening on the specified port to free GPU.
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
    --skip_serve)
      SKIP_SERVE="true"; shift 1 ;;
    --skip-serve)
      SKIP_SERVE="true"; shift 1 ;;
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

API_URL="http://127.0.0.1:${PORT}/v1"

if [[ "${SKIP_SERVE}" != "true" ]]; then
  echo "Starting vLLM serve on port ${PORT} ..."
  MODEL_NORMALIZED="${MODEL%/}"
  MODEL_NAME="${MODEL_NORMALIZED##*/}"
  EXTRA_ARGS=()
  # Only for base DeepSeek-V4-Flash/Pro model names without quantized suffixes.
  if [[ "${MODEL_NAME}" == "DeepSeek-V4-Flash" || "${MODEL_NAME}" == "DeepSeek-V4-Pro" ]]; then
    EXTRA_ARGS+=(--enable-expert-parallel)
    EXTRA_ARGS+=(--moe-backend deep_gemm_mega_moe)
  fi

  VLLM_CMD=(
    vllm serve "${MODEL}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --block-size "${BLOCK_SIZE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --attention_config.use_fp4_indexer_cache=True
    --port "${PORT}"
  )
  if [[ "${MODEL_NAME}" == *"DeepSeek-V4-Pro"* ]]; then
    VLLM_CMD+=(--max-model-len 1048576)
  fi
  if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
    VLLM_CMD+=(--trust-remote-code)
  fi
  if [[ "${NO_ENABLE_FLASHINFER_AUTOTUNE}" == "true" ]]; then
    VLLM_CMD+=(--no-enable-flashinfer-autotune)
  fi
  VLLM_CMD+=("${EXTRA_ARGS[@]}")

  SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU}" "${VLLM_CMD[@]}" >/tmp/vllm_${PORT}.log 2>&1 &
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

DEFAULT_STANDARD_TASKS=(piqa hellaswag gsm8k mmlu_pro math_500 mmlu)
SUPPORTED_TASKS=(aime26 gpqa_diamond ruler_qa_squad "${DEFAULT_STANDARD_TASKS[@]}")
SELECTED_STANDARD_TASKS=()
RUN_AIME26="true"
RUN_GPQA_DIAMOND="true"
RUN_STANDARD_TASKS="true"
RUN_RULER_QA_SQUAD="true"

if [[ -n "${TASKS}" ]]; then
  RUN_AIME26="false"
  RUN_GPQA_DIAMOND="false"
  RUN_STANDARD_TASKS="false"
  RUN_RULER_QA_SQUAD="false"

  IFS=',' read -r -a REQUESTED_TASKS <<< "${TASKS}"
  for raw_task in "${REQUESTED_TASKS[@]}"; do
    task_name="$(trim_task_name "${raw_task}")"
    if [[ -z "${task_name}" ]]; then
      continue
    fi
    if ! task_in_list "${task_name}" "${SUPPORTED_TASKS[@]}"; then
      echo "Unsupported task: ${task_name}"
      echo "Supported tasks: ${SUPPORTED_TASKS[*]}"
      exit 1
    fi

    case "${task_name}" in
      aime26)
        RUN_AIME26="true"
        ;;
      gpqa_diamond)
        RUN_GPQA_DIAMOND="true"
        ;;
      ruler_qa_squad)
        RUN_RULER_QA_SQUAD="true"
        ;;
      *)
        if ! task_in_list "${task_name}" "${SELECTED_STANDARD_TASKS[@]}"; then
          SELECTED_STANDARD_TASKS+=("${task_name}")
          RUN_STANDARD_TASKS="true"
        fi
        ;;
    esac
  done

  if [[ "${RUN_AIME26}" != "true" ]] && [[ "${RUN_GPQA_DIAMOND}" != "true" ]] \
    && [[ "${RUN_STANDARD_TASKS}" != "true" ]] && [[ "${RUN_RULER_QA_SQUAD}" != "true" ]]; then
    echo "No valid tasks selected from --tasks '${TASKS}'."
    exit 1
  fi
else
  SELECTED_STANDARD_TASKS=("${DEFAULT_STANDARD_TASKS[@]}")
fi

TOTAL_STEPS=0
if [[ "${RUN_AIME26}" == "true" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "${RUN_GPQA_DIAMOND}" == "true" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "${RUN_STANDARD_TASKS}" == "true" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "${RUN_RULER_QA_SQUAD}" == "true" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
STEP_INDEX=1

echo "=== Evaluation started at $(date) ===" | tee "$OUTPUT_FILE"
echo "Model: $MODEL" | tee -a "$OUTPUT_FILE"
echo "API URL: $API_URL" | tee -a "$OUTPUT_FILE"
echo "Temperature: $TEMPERATURE" | tee -a "$OUTPUT_FILE"
if [[ -n "${TASKS}" ]]; then
  echo "Tasks: ${TASKS}" | tee -a "$OUTPUT_FILE"
else
  echo "Tasks: all default tasks" | tee -a "$OUTPUT_FILE"
fi
echo "" | tee -a "$OUTPUT_FILE"
  

if [[ "${RUN_AIME26}" == "true" ]]; then
  echo "" | tee -a "$OUTPUT_FILE"
  print_section_header "aime26 (n=10)"
  evalscope eval \
    --model "$MODEL" \
    --eval-type openai_api \
    --api-key EMPTY \
    --datasets aime26 \
    --generation-config "{\"temperature\": ${TEMPERATURE}, \"n\": 10}" \
    --eval-batch-size 10 --timeout 3000 \
    --api-url "$API_URL" 2>&1 | tee -a "$OUTPUT_FILE"
fi

if [[ "${RUN_GPQA_DIAMOND}" == "true" ]]; then
  echo "" | tee -a "$OUTPUT_FILE"
  print_section_header "gpqa_diamond (n=5)"
  evalscope eval \
    --model "$MODEL" \
    --eval-type openai_api \
    --api-key EMPTY \
    --datasets gpqa_diamond \
    --generation-config "{\"temperature\": ${TEMPERATURE}, \"n\": 5}" \
    --eval-batch-size 10 --timeout 3000 \
    --api-url "$API_URL" 2>&1 | tee -a "$OUTPUT_FILE"
fi

if [[ "${RUN_STANDARD_TASKS}" == "true" ]]; then
  echo "" | tee -a "$OUTPUT_FILE"
  print_section_header "${SELECTED_STANDARD_TASKS[*]}"
  evalscope eval \
    --model "$MODEL" \
    --eval-type openai_api \
    --api-key EMPTY \
    --datasets "${SELECTED_STANDARD_TASKS[@]}" \
    --eval-batch-size 10 --timeout 3000 \
    --api-url "$API_URL" 2>&1 | tee -a "$OUTPUT_FILE"
fi

if [[ "${RUN_RULER_QA_SQUAD}" == "true" ]]; then
  echo "" | tee -a "$OUTPUT_FILE"
  print_section_header "ruler_qa_squad (lm_eval, 1M)"
  if [[ "${MODEL_NAME}" == *"DeepSeek-V4-Pro"* ]]; then
    LMEVAL_OUTPUT_DIR="${LOG_DIR}/lm_eval_ruler_1M_qa"
    mkdir -p "${LMEVAL_OUTPUT_DIR}"
    LMEVAL_METADATA=$(printf '{"max_seq_lengths":[1000000],"pretrained":"%s/","use_fast":false}' "${MODEL_NORMALIZED}")
    lm_eval \
      --model local-completions \
      --tasks ruler_qa_squad \
      --model_args "model=${MODEL_NORMALIZED},base_url=${API_URL}/completions,num_concurrent=1,max_retries=3,max_length=1048576" \
      --gen_kwargs "temperature=${TEMPERATURE},do_sample=False,max_tokens=128" \
      --metadata "${LMEVAL_METADATA}" \
      --batch_size 1 \
      --log_samples \
      --output_path "${LMEVAL_OUTPUT_DIR}" 2>&1 | tee -a "$OUTPUT_FILE"
  else
    echo "Skip ruler_qa_squad: only DeepSeek-V4-Pro is supported for this test." | tee -a "$OUTPUT_FILE"
  fi
fi


echo "" | tee -a "$OUTPUT_FILE"
echo "=== Evaluation finished at $(date) ===" | tee -a "$OUTPUT_FILE"
echo "Results saved to: $OUTPUT_FILE"

# Kill the process listening on the specified port to free GPU
echo "Stopping process on port ${PORT} to free GPU..." | tee -a "$OUTPUT_FILE"
VLLM_PIDS=$(ps aux | grep -- "vllm serve" | grep -- "--port[ =]${PORT}" | grep -v grep | awk '{print $2}')
if [[ -n "$VLLM_PIDS" ]]; then
  echo "Found vllm serve process(es) with --port ${PORT}: $VLLM_PIDS" | tee -a "$OUTPUT_FILE"
  for PID in $VLLM_PIDS; do
    # Kill all child processes (including GPU processes)
    CHILDREN=$(pgrep -P $PID)
    if [[ -n "$CHILDREN" ]]; then
      echo "Killing child processes of $PID: $CHILDREN" | tee -a "$OUTPUT_FILE"
      kill -9 $CHILDREN 2>/dev/null
    fi
    kill -9 $PID 2>/dev/null
    echo "Killed vllm serve process and its children: $PID $CHILDREN" | tee -a "$OUTPUT_FILE"
  done
else
  echo "No vllm serve process found with --port ${PORT}." | tee -a "$OUTPUT_FILE"
fi
