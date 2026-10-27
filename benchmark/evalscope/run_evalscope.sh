#!/bin/bash
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Shared evalscope driver for quantized LLM examples.
#
# Starts `vllm serve`, evaluates every task --num_runs times through the OpenAI-compatible API
# with evalscope, and summarizes the scores with eval_summary.py. With 3 or more runs the
# highest and lowest score of each task are dropped before averaging.
#
# Model-family specific settings are provided by the caller (a thin run_benchmark.sh in each
# example directory) through these variables. Built-in defaults follow the MiniMax example.
#   DEFAULT_TASKS              default value for --tasks (default: gpqa_diamond,aime25,gsm8k,piqa)
#   DEFAULT_NUM_RUNS           default value for --num_runs (default: 5)
#   DEFAULT_TEMPERATURE        default value for --temperature (default: 1.0)
#   DEFAULT_TOP_P              default value for --top_p (default: 0.95, empty: not sent)
#   DEFAULT_MAX_TOKENS         default value for --max_tokens (default: 64000, empty: not sent)
#   DEFAULT_MAX_MODEL_LEN      default value for --max_model_len (default: 102400, empty: vLLM default)
#   DEFAULT_SERVED_MODEL_NAME  default value for --served_model_name (default: minimax-m2.7, empty: the model path)
#   DEFAULT_EVAL_BATCH_SIZE    default value for --eval_batch_size (default: 256)
#   EVAL_TIMEOUT               evalscope request timeout in seconds (default: 3600)
#   EXTRA_GENERATION_CONFIG    JSON object merged into the evalscope --generation-config
#   SINGLE_RUN_TASKS           tasks evaluated only once regardless of --num_runs (default: hellaswag,piqa,mmlu,gsm8k)
#   EXTRA_SERVE_ARGS           extra flags appended to `vllm serve`
#   ATTENTION_BACKEND_FP8KV    vLLM attention backend when static kv is fp8 (default: FLASHINFER)
#   ATTENTION_BACKEND_FP8ATTN  vLLM attention backend when static attention is fp8 (default: FLASHINFER;
#                              forced to TRITON_ATTN on GPUs other than SM 10.x)
#   FLASHINFER_WORKSPACE_SIZE  VLLM_AR_FLASHINFER_WORKSPACE_BUFFER_SIZE value

set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODEL_PATH=""
TASKS="${DEFAULT_TASKS:-gpqa_diamond,aime25,gsm8k,piqa}"
NUM_RUNS="${DEFAULT_NUM_RUNS:-5}"
TEMPERATURE="${DEFAULT_TEMPERATURE:-1.0}"
# An explicitly empty value disables the following options.
TOP_P="${DEFAULT_TOP_P-0.95}"
MAX_TOKENS="${DEFAULT_MAX_TOKENS-64000}"
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN-102400}"
SERVED_MODEL_NAME="${DEFAULT_SERVED_MODEL_NAME-minimax-m2.7}"
EVAL_BATCH_SIZE="${DEFAULT_EVAL_BATCH_SIZE:-256}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-3600}"
SINGLE_RUN_TASKS="${SINGLE_RUN_TASKS-hellaswag,piqa,mmlu,gsm8k}"
STATIC_KV_DTYPE="auto"
STATIC_ATTENTION_DTYPE="auto"
SKIP_SERVE="${SKIP_SERVE:-false}"

USER_SERVER_PORT="${SERVER_PORT:+1}"
SERVER_PORT="${SERVER_PORT:-8001}"

usage() {
    echo "Usage: CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>]"
    echo "  --model_path               Path to the quantized model (required)"
    echo "  --tasks                    Comma-separated evalscope dataset names (default: ${TASKS})"
    echo "  --num_runs                 Times each task is evaluated; with >= 3 runs max/min are dropped (default: ${NUM_RUNS})."
    echo "                             Tasks in SINGLE_RUN_TASKS (${SINGLE_RUN_TASKS:-none}) always run once."
    echo "  --temperature              Generation temperature (default: ${TEMPERATURE})"
    echo "  --top_p                    Generation top_p (default: ${TOP_P:-not set})"
    echo "  --max_tokens               Max generated tokens (default: ${MAX_TOKENS:-not set})"
    echo "  --max_model_len            vLLM max model length (default: ${MAX_MODEL_LEN:-vLLM default})"
    echo "  --served_model_name        vLLM served model name (default: ${SERVED_MODEL_NAME:-model path})"
    echo "  --eval_batch_size          evalscope concurrent requests (default: ${EVAL_BATCH_SIZE})"
    echo "  --port                     vLLM API port (default: \$SERVER_PORT or 8001; the next free port is used if unset and busy)"
    echo "  --static_kv_dtype          Data type for static kv cache (default: auto)"
    echo "  --static_attention_dtype   Data type for static attention cache (default: auto)"
    echo "  --skip_serve               true to reuse a vLLM server already listening on --port (default: false)"
    echo ""
    echo "Tensor parallel size is inferred from CUDA_VISIBLE_DEVICES."
    exit 0
}

for arg in "$@"; do
    case $arg in
        --model_path=*)
            MODEL_PATH="${arg#*=}"
            ;;
        --tasks=*)
            TASKS="${arg#*=}"
            ;;
        --num_runs=*)
            NUM_RUNS="${arg#*=}"
            ;;
        --temperature=*)
            TEMPERATURE="${arg#*=}"
            ;;
        --top_p=*)
            TOP_P="${arg#*=}"
            ;;
        --max_tokens=*)
            MAX_TOKENS="${arg#*=}"
            ;;
        --max_model_len=*)
            MAX_MODEL_LEN="${arg#*=}"
            ;;
        --served_model_name=*)
            SERVED_MODEL_NAME="${arg#*=}"
            ;;
        --eval_batch_size=*)
            EVAL_BATCH_SIZE="${arg#*=}"
            ;;
        --port=*)
            SERVER_PORT="${arg#*=}"
            USER_SERVER_PORT=1
            ;;
        --static_kv_dtype=*)
            STATIC_KV_DTYPE="${arg#*=}"
            ;;
        --static_attention_dtype=*)
            STATIC_ATTENTION_DTYPE="${arg#*=}"
            ;;
        --skip_serve=*)
            SKIP_SERVE="${arg#*=}"
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter: $arg" >&2
            exit 1
            ;;
    esac
done

# `--model_path=~/...` is not tilde-expanded by the shell.
MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
SKIP_SERVE="$(echo "${SKIP_SERVE}" | tr '[:upper:]' '[:lower:]')"

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required." >&2
    exit 1
fi
if [[ "$SKIP_SERVE" != "true" && ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model path '$MODEL_PATH' does not exist!" >&2
    exit 1
fi
if ! [[ "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Invalid --num_runs '${NUM_RUNS}' (must be a positive integer)." >&2
    exit 1
fi
if ! [[ "$SERVER_PORT" =~ ^[0-9]+$ ]] || (( SERVER_PORT < 1 || SERVER_PORT > 65535 )); then
    echo "Error: Invalid server port '${SERVER_PORT}'. Expected a value from 1 to 65535." >&2
    exit 1
fi

SELECTED_TASKS=()
IFS=',' read -ra task_array <<< "$TASKS"
for task in "${task_array[@]}"; do
    task="${task//[[:space:]]/}"
    [[ -z "$task" ]] && continue
    if [[ " ${SELECTED_TASKS[*]} " != *" ${task} "* ]]; then
        SELECTED_TASKS+=("$task")
    fi
done
if [[ ${#SELECTED_TASKS[@]} -eq 0 ]]; then
    echo "Error: No tasks selected from --tasks '${TASKS}'." >&2
    exit 1
fi

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    TP_SIZE=${#GPU_ARRAY[@]}
else
    TP_SIZE=1
fi

# For CT format eval on GPUs other than SM 10.x, use the QDQ plugin.
# The GPU is only queried when this script starts the server.
IS_SM10=0
if [[ "$SKIP_SERVE" != "true" ]]; then
    GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
    CUDA_SM=$(nvidia-smi --id="${GPU_ID:-0}" --query-gpu=compute_cap --format=csv,noheader,nounits)
    if [[ "$CUDA_SM" == "10."* ]]; then
        IS_SM10=1
    else
        export VLLM_QDQ=1
        export VLLM_MXFP4_USE_MARLIN=1
    fi
fi

# Resolve the attention backend. FP8 attention implies an FP8 kv cache and takes precedence.
KV_CACHE_DTYPE="$STATIC_KV_DTYPE"
ATTN_BACKEND=""
ATTN_SERVE_ARGS=()
if [[ "$STATIC_ATTENTION_DTYPE" == "fp8" ]]; then
    KV_CACHE_DTYPE="fp8"
    ATTN_BACKEND="${ATTENTION_BACKEND_FP8ATTN:-FLASHINFER}"
    # FP8 attention only works with TRITON_ATTN on GPUs other than SM 10.x.
    [[ "$IS_SM10" == 1 ]] || ATTN_BACKEND="TRITON_ATTN"
    # Quantize Q only when attention itself runs in FP8.
    export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=0
    echo "Using FP8 Attention with ${ATTN_BACKEND} backend"
elif [[ "$STATIC_KV_DTYPE" == "fp8" ]]; then
    ATTN_BACKEND="${ATTENTION_BACKEND_FP8KV:-FLASHINFER}"
    export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=1
    echo "Using FP8 for KV cache with ${ATTN_BACKEND} backend"
fi

# Apply the backend: TRITON_ATTN goes through CLI args, FlashInfer variants through env vars.
if [[ "$ATTN_BACKEND" == "TRITON_ATTN" ]]; then
    ATTN_SERVE_ARGS=(--attention-backend TRITON_ATTN)
elif [[ -n "$ATTN_BACKEND" ]]; then
    export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
    if [[ -n "${FLASHINFER_WORKSPACE_SIZE:-}" ]]; then
        export VLLM_AR_FLASHINFER_WORKSPACE_BUFFER_SIZE="${FLASHINFER_WORKSPACE_SIZE}"
    fi
fi

# Repeated evaluation is done with --num_runs; evalscope only scores the first choice, so n stays 1.
GEN_CONFIG=$(python3 -c '
import json, sys

temperature, top_p, max_tokens, extra = sys.argv[1:]
config = {"temperature": float(temperature), "n": 1}
if top_p:
    config["top_p"] = float(top_p)
if max_tokens:
    config["max_tokens"] = int(max_tokens)
if extra:
    config.update(json.loads(extra))
print(json.dumps(config))
' "$TEMPERATURE" "$TOP_P" "$MAX_TOKENS" "${EXTRA_GENERATION_CONFIG:-}")

port_in_use() {
    (: < "/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

if [[ "$SKIP_SERVE" != "true" ]]; then
    if [[ -n "$USER_SERVER_PORT" ]]; then
        if port_in_use "$SERVER_PORT"; then
            echo "Error: Port ${SERVER_PORT} is already in use. Pass --skip_serve=true to reuse that server." >&2
            exit 1
        fi
    else
        while port_in_use "$SERVER_PORT"; do
            echo "Port ${SERVER_PORT} is occupied; trying port $((SERVER_PORT + 1))."
            SERVER_PORT=$((SERVER_PORT + 1))
            if (( SERVER_PORT > 65535 )); then
                echo "Error: No available server port up to 65535." >&2
                exit 1
            fi
        done
    fi
fi

MODEL_PATH="${MODEL_PATH%/}"
MODEL_NAME=$(basename "$MODEL_PATH")
EVAL_MODEL="${SERVED_MODEL_NAME:-$MODEL_PATH}"
API_URL="http://127.0.0.1:${SERVER_PORT}/v1"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${MODEL_NAME}-tp${TP_SIZE}-evalscope"
VLLM_LOG="${OUTPUT_DIR}/vllm_server_${RUN_TIMESTAMP}_port${SERVER_PORT}.log"
OUTPUT_FILE="${OUTPUT_DIR}/eval_results_${RUN_TIMESTAMP}_port${SERVER_PORT}_temp${TEMPERATURE}.log"
# evalscope outputs of each run: ${EVAL_OUTPUT_ROOT}/<task>/run<i>/
EVAL_OUTPUT_ROOT="${OUTPUT_DIR}/outputs_${RUN_TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

VLLM_PID=""
LOG_TAIL_PID=""

stop_log_tail() {
    if [[ -n "$LOG_TAIL_PID" ]] && kill -0 "$LOG_TAIL_PID" 2>/dev/null; then
        kill "$LOG_TAIL_PID" 2>/dev/null || true
    fi
    LOG_TAIL_PID=""
}

cleanup_server() {
    stop_log_tail
    if [[ -z "$VLLM_PID" ]]; then
        return
    fi
    echo "Shutting down vLLM server (PID ${VLLM_PID})..."
    # vLLM runs in its own process group (see setsid below), so its workers are stopped as well.
    kill -TERM -- "-${VLLM_PID}" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$VLLM_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-${VLLM_PID}" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    echo "Server stopped"
}

trap cleanup_server EXIT
trap 'exit 130' INT TERM

# Keep the readiness check below identical to benchmark/lm_eval/run_lm_eval.sh.
SERVER_WAIT_TIMEOUT=1800
SERVER_WAIT_INTERVAL=10

print_server_log_tail() {
    if [[ -n "$VLLM_LOG" && -f "$VLLM_LOG" ]]; then
        echo "----- Last 80 lines of ${VLLM_LOG} -----" >&2
        tail -n 80 "$VLLM_LOG" >&2 || true
    fi
}

# Poll the OpenAI API until it responds; fail early if the vLLM process started by this script exits.
wait_for_server() {
    local url="http://127.0.0.1:${SERVER_PORT}/v1/models"
    local deadline=$((SECONDS + SERVER_WAIT_TIMEOUT))
    echo "Waiting for vLLM server at ${url} (timeout ${SERVER_WAIT_TIMEOUT}s)..."
    while (( SECONDS < deadline )); do
        if curl -sf "$url" -o /dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM server is ready."
            return 0
        fi
        if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: vLLM server exited before becoming ready." >&2
            return 1
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM server on port ${SERVER_PORT} not ready, retrying in ${SERVER_WAIT_INTERVAL}s..."
        sleep "$SERVER_WAIT_INTERVAL"
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Timed out after ${SERVER_WAIT_TIMEOUT}s waiting for vLLM server at ${url}." >&2
    return 1
}

if [[ "$SKIP_SERVE" != "true" ]]; then
    VLLM_CMD=(
        vllm serve "$MODEL_PATH"
        --port "$SERVER_PORT"
        --tensor-parallel-size "$TP_SIZE"
    )
    if [[ -n "$SERVED_MODEL_NAME" ]]; then
        VLLM_CMD+=(--served-model-name "$SERVED_MODEL_NAME")
    fi
    if [[ -n "$MAX_MODEL_LEN" ]]; then
        VLLM_CMD+=(--max-model-len "$MAX_MODEL_LEN")
    fi
    if [[ "$KV_CACHE_DTYPE" != "auto" ]]; then
        VLLM_CMD+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    fi
    VLLM_CMD+=("${ATTN_SERVE_ARGS[@]}")
    read -ra SERVE_EXTRA_ARGS <<< "${EXTRA_SERVE_ARGS:-}"
    VLLM_CMD+=("${SERVE_EXTRA_ARGS[@]}")

    echo "Starting vLLM server on port ${SERVER_PORT}..."
    echo "  ${VLLM_CMD[*]}"
    setsid "${VLLM_CMD[@]}" > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "vLLM server started with PID: ${VLLM_PID}, log: ${VLLM_LOG}"
    echo "=== vLLM startup log (stops once the API is ready) ==="
    tail -n +1 -f "$VLLM_LOG" &
    LOG_TAIL_PID=$!
fi

if ! wait_for_server; then
    stop_log_tail
    print_server_log_tail
    exit 1
fi
stop_log_tail

task_num_runs() {
    if [[ ",${SINGLE_RUN_TASKS//[[:space:]]/}," == *",$1,"* ]]; then
        echo 1
    else
        echo "$NUM_RUNS"
    fi
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] API is ready, starting evaluation."
{
    echo "=== Evaluation started at $(date) ==="
    echo "Model: ${MODEL_PATH}"
    echo "Served model name: ${EVAL_MODEL}"
    echo "API URL: ${API_URL}"
    echo "Tensor parallel size: ${TP_SIZE}"
    echo "KV cache dtype: ${KV_CACHE_DTYPE}"
    echo "Tasks: ${SELECTED_TASKS[*]}"
    echo "Runs per task: ${NUM_RUNS} (single run: ${SINGLE_RUN_TASKS:-none})"
    echo "Generation config: ${GEN_CONFIG}"
    echo "Evalscope outputs: ${EVAL_OUTPUT_ROOT}"
} | tee "$OUTPUT_FILE"

STEP_INDEX=1
SUMMARY_FAILED=0
for task in "${SELECTED_TASKS[@]}"; do
    DATASET_ARGS=()
    if [[ "$task" == "live_code_bench" ]]; then
        DATASET_ARGS=(--dataset-args '{"live_code_bench": {"subset_list": ["v6"]}}')
    fi

    {
        echo ""
        echo "=== [${STEP_INDEX}/${#SELECTED_TASKS[@]}] ${task} ==="
    } | tee -a "$OUTPUT_FILE"
    STEP_INDEX=$((STEP_INDEX + 1))

    TASK_RUNS=$(task_num_runs "$task")
    for run in $(seq 1 "$TASK_RUNS"); do
        echo "--- ${task}: run ${run}/${TASK_RUNS} ---" | tee -a "$OUTPUT_FILE"
        evalscope eval \
            --model "$EVAL_MODEL" \
            --eval-type openai_api \
            --api-url "$API_URL" \
            --api-key EMPTY \
            --timeout "$EVAL_TIMEOUT" \
            --datasets "$task" \
            "${DATASET_ARGS[@]}" \
            --generation-config "$GEN_CONFIG" \
            --eval-batch-size "$EVAL_BATCH_SIZE" \
            --work-dir "${EVAL_OUTPUT_ROOT}/${task}/run${run}" \
            --no-timestamp 2>&1 | tee -a "$OUTPUT_FILE"
    done

    echo "" | tee -a "$OUTPUT_FILE"
    if ! python3 "${SCRIPT_DIR}/eval_summary.py" --root "$EVAL_OUTPUT_ROOT" --tasks "$task" 2>&1 | tee -a "$OUTPUT_FILE"; then
        SUMMARY_FAILED=1
    fi
done

SELECTED_TASKS_CSV="$(IFS=','; echo "${SELECTED_TASKS[*]}")"
echo "" | tee -a "$OUTPUT_FILE"
if ! python3 "${SCRIPT_DIR}/eval_summary.py" --root "$EVAL_OUTPUT_ROOT" --tasks "$SELECTED_TASKS_CSV" --summary 2>&1 \
    | tee -a "$OUTPUT_FILE"; then
    SUMMARY_FAILED=1
fi

{
    echo ""
    echo "=== Evaluation finished at $(date) ==="
} | tee -a "$OUTPUT_FILE"
if [[ "$SUMMARY_FAILED" == 1 ]]; then
    echo "Error: Failed to summarize evalscope results, see errors above. Logs: ${OUTPUT_FILE}" >&2
    exit 1
fi
echo "Benchmark completed successfully! Results saved to ${OUTPUT_FILE}"
