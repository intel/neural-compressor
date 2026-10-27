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

# Shared lm-eval driver for quantized LLM examples.
#
# Model-family specific settings are provided by the caller (a thin
# run_benchmark.sh in each example directory) through these variables:
#   ATTENTION_BACKEND_FP8KV    vLLM attention backend when static kv is fp8
#   ATTENTION_BACKEND_FP8ATTN  vLLM attention backend when static attention is fp8
#   FLASHINFER_WORKSPACE_SIZE  VLLM_AR_FLASHINFER_WORKSPACE_BUFFER_SIZE value
#   EXTRA_SERVE_ARGS           extra flags appended to `vllm serve`
#   EXTRA_LM_EVAL_MODEL_ARGS   extra `,key=value` pairs appended to --model_args

set -eo pipefail

MODEL_PATH=""
TASKS="hellaswag,piqa,mmlu,gsm8k"
BATCH_SIZE="auto"
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=8192
STATIC_KV_DTYPE="auto"
STATIC_ATTENTION_DTYPE="auto"
RULER_MAX_POS="${RULER_MAX_POS:-131072}"

USER_SERVER_PORT="${SERVER_PORT:+1}"
SERVER_PORT="${SERVER_PORT:-8000}"

usage() {
    echo "Usage: CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>]"
    echo "  --model_path               Path to the quantized model (required)"
    echo "  --tasks                    Task name(s) to evaluate (default: ${TASKS})"
    echo "  --batch_size               Batch size (default: auto)"
    echo "  --gpu_memory_utilization   GPU memory utilization (default: 0.90)"
    echo "  --max_model_len            Max model length for standard tasks (default: 8192)"
    echo "  --static_kv_dtype          Data type for static kv cache (default: auto)"
    echo "  --static_attention_dtype   Data type for static attention cache (default: auto)"
    echo "  --ruler_max_pos            Max position length for RULER eval (default: ${RULER_MAX_POS:-131072})"
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
        --batch_size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        --gpu_memory_utilization=*)
            GPU_MEMORY_UTILIZATION="${arg#*=}"
            ;;
        --max_model_len=*)
            MAX_MODEL_LEN="${arg#*=}"
            ;;
        --static_kv_dtype=*)
            STATIC_KV_DTYPE="${arg#*=}"
            ;;
        --static_attention_dtype=*)
            STATIC_ATTENTION_DTYPE="${arg#*=}"
            ;;
        --ruler_max_pos=*)
            RULER_MAX_POS="${arg#*=}"
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

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model path '$MODEL_PATH' does not exist!"
    exit 1
fi

# Count available GPUs and set tensor_parallel_size
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    TP_SIZE=${#GPU_ARRAY[@]}
else
    TP_SIZE=1
fi

MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="${MODEL_NAME}-tp${TP_SIZE}-eval"
mkdir -p "$OUTPUT_DIR"

KV_CACHE_DTYPE="$STATIC_KV_DTYPE"

LONGBENCH_TASKS=""
RULER_TASKS=""
CHAT_TEMPLATE_TASKS=""
STANDARD_TASKS=""
IFS=',' read -ra task_array <<< "$TASKS"
for task in "${task_array[@]}"; do
    [[ -z "$task" ]] && continue
    if [[ "$task" == *longbench* ]]; then
        LONGBENCH_TASKS="${LONGBENCH_TASKS:+$LONGBENCH_TASKS,}$task"
    elif [[ "$task" == *ruler* || "$task" == niah_* ]]; then
        RULER_TASKS="${RULER_TASKS:+$RULER_TASKS,}$task"
    # These tasks need a chat template applied.
    elif [[ "$task" == "gsm8k_llama" || "$task" == "mmlu_llama" ]]; then
        CHAT_TEMPLATE_TASKS="${CHAT_TEMPLATE_TASKS:+$CHAT_TEMPLATE_TASKS,}$task"
    else
        STANDARD_TASKS="${STANDARD_TASKS:+$STANDARD_TASKS,}$task"
    fi
done

LONGBENCH_MAX_LEN=131072
LONGBENCH_MAX_CTX_LEN=$((LONGBENCH_MAX_LEN - 2048))

RULER_GEN_TOKS=128
# Leave room for generated tokens so prompt + output stays within the server context limit.
RULER_SEQ_LEN=$((RULER_MAX_POS - RULER_GEN_TOKS))
RULER_BATCH_SIZE=32

LM_EVAL_EXTRA_ARGS="${EXTRA_LM_EVAL_MODEL_ARGS:-}"
SERVE_EXTRA_ARGS="${EXTRA_SERVE_ARGS:-}"

# For CT format eval on GPUs other than SM 10.x, use the QDQ plugin.
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
CUDA_SM=$(nvidia-smi --id="${GPU_ID:-0}" --query-gpu=compute_cap --format=csv,noheader,nounits)
IS_SM10=0
if [[ "$CUDA_SM" == "10."* ]]; then
    IS_SM10=1
else
    export VLLM_QDQ=1
    export VLLM_MXFP4_USE_MARLIN=1
fi

# Resolve the attention backend. FP8 attention implies an FP8 kv cache and takes precedence.
ATTN_BACKEND=""
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
    SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS} --attention-backend TRITON_ATTN"
    LM_EVAL_EXTRA_ARGS="${LM_EVAL_EXTRA_ARGS},attention_backend=TRITON_ATTN"
elif [[ -n "$ATTN_BACKEND" ]]; then
    export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
    if [[ -n "${FLASHINFER_WORKSPACE_SIZE:-}" ]]; then
        export VLLM_AR_FLASHINFER_WORKSPACE_BUFFER_SIZE="${FLASHINFER_WORKSPACE_SIZE}"
    fi
fi


echo "Running benchmark with parameters:"
echo "  Model Path: ${MODEL_PATH}"
echo "  Tasks: ${TASKS}"
echo "    standard: ${STANDARD_TASKS:-none}; chat-template: ${CHAT_TEMPLATE_TASKS:-none}; longbench: ${LONGBENCH_TASKS:-none}; ruler: ${RULER_TASKS:-none}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Max Model Length: ${MAX_MODEL_LEN}"
echo "  KV Cache Dtype: ${KV_CACHE_DTYPE}"
echo "  GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  Tensor Parallel Size: ${TP_SIZE}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  Output directory: ${OUTPUT_DIR}"


STANDARD_VLLM_MODEL_ARGS="pretrained=${MODEL_PATH},tensor_parallel_size=${TP_SIZE},max_model_len=${MAX_MODEL_LEN},max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,kv_cache_dtype=${KV_CACHE_DTYPE}${LM_EVAL_EXTRA_ARGS}"

run_standard_eval() {
    echo "Running evaluation for tasks: ${STANDARD_TASKS}"
    lm_eval --model vllm \
        --model_args "${STANDARD_VLLM_MODEL_ARGS}" \
        --tasks "${STANDARD_TASKS}" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --seed 42 \
        --output_path "$OUTPUT_DIR" \
        --show_config
}

run_chat_template_eval() {
    echo "Running chat-template evaluation for tasks: ${CHAT_TEMPLATE_TASKS}"
    lm_eval --model vllm \
        --model_args "${STANDARD_VLLM_MODEL_ARGS}" \
        --tasks "${CHAT_TEMPLATE_TASKS}" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --seed 42 \
        --output_path "$OUTPUT_DIR" \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --show_config
}

port_in_use() {
    (: < "/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

find_available_port() {
    if ! [[ "$SERVER_PORT" =~ ^[0-9]+$ ]] || (( SERVER_PORT < 1 || SERVER_PORT > 65535 )); then
        echo "Error: Invalid server port '${SERVER_PORT}'. Expected a value from 1 to 65535." >&2
        return 1
    fi

    if [[ -n "$USER_SERVER_PORT" ]]; then
        if port_in_use "$SERVER_PORT"; then
            echo "Error: SERVER_PORT ${SERVER_PORT} is already in use." >&2
            return 1
        fi
        return 0
    fi

    while port_in_use "$SERVER_PORT"; do
        echo "Port ${SERVER_PORT} is occupied; trying port $((SERVER_PORT + 1))."
        SERVER_PORT=$((SERVER_PORT + 1))
        if (( SERVER_PORT > 65535 )); then
            echo "Error: No available server port up to 65535." >&2
            return 1
        fi
    done
}

VLLM_PID=""
VLLM_LOG=""

start_vllm_server() {
    local max_len=$1
    local log_name=$2
    find_available_port || exit 1
    VLLM_LOG="${OUTPUT_DIR}/vllm_server_${log_name}.log"
    echo "Starting vLLM server on port ${SERVER_PORT}, log: ${VLLM_LOG}"

    local serve_extra_args=()
    read -ra serve_extra_args <<< "${SERVE_EXTRA_ARGS}"
    # setsid puts vLLM and its workers in their own process group so cleanup_server can stop all of them.
    setsid vllm serve "${MODEL_PATH}" \
        --port "${SERVER_PORT}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-model-len "${max_len}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --dtype bfloat16 \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        "${serve_extra_args[@]}" \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    echo "vLLM server started with PID: ${VLLM_PID}"
}

# Keep the readiness check below identical to benchmark/evalscope/run_evalscope.sh.
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

cleanup_server() {
    if [[ -z "$VLLM_PID" ]]; then
        return
    fi
    echo "Shutting down vLLM server (PID ${VLLM_PID})..."
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

start_server_or_exit() {
    start_vllm_server "$1" "$2"
    if ! wait_for_server; then
        print_server_log_tail
        exit 1
    fi
}

run_longbench_eval() {
    start_server_or_exit "${LONGBENCH_MAX_LEN}" longbench

    echo "Running LongBench evaluation against vLLM server..."
    python -m long_bench_eval.cli \
        --api-key dummy \
        --base-url "http://localhost:${SERVER_PORT}/v1" \
        --model "${MODEL_PATH}" \
        --max-context-length "${LONGBENCH_MAX_CTX_LEN}" \
        --num-threads 1 \
        --deterministic \
        --categories "Long In-context Learning"
    cleanup_server
}

run_ruler_eval() {
    start_server_or_exit "${RULER_MAX_POS}" ruler

    echo "Running RULER evaluation against vLLM server..."
    lm_eval \
        --model local-completions \
        --model_args "model=${MODEL_PATH},base_url=http://localhost:${SERVER_PORT}/v1/completions,num_concurrent=1,max_retries=50,timeout=500,tokenized_requests=False,max_gen_toks=${RULER_GEN_TOKS},max_length=${RULER_SEQ_LEN}" \
        --tasks "${RULER_TASKS}" \
        --metadata="{\"max_seq_lengths\":[${RULER_SEQ_LEN}],\"tokenizer\":\"${MODEL_PATH}\"}" \
        --gen_kwargs "max_gen_toks=${RULER_GEN_TOKS}" \
        --batch_size "${RULER_BATCH_SIZE}" \
        ${LIMIT:+--limit $LIMIT} \
        --log_samples \
        --output_path "${OUTPUT_DIR}/seq_${RULER_SEQ_LEN}" \
        --seed 42
    cleanup_server
}

# Standard evals load vLLM in-process, so run them before any server-based eval.
if [[ -n "$STANDARD_TASKS" ]]; then
    run_standard_eval
fi
if [[ -n "$CHAT_TEMPLATE_TASKS" ]]; then
    run_chat_template_eval
fi
if [[ -n "$LONGBENCH_TASKS" ]]; then
    echo "Running LongBench evaluation..."
    run_longbench_eval
fi
if [[ -n "$RULER_TASKS" ]]; then
    echo "Running RULER evaluation..."
    # Limit the number of samples per task for RULER evaluation to accelerate testing
    export LIMIT=100 
    run_ruler_eval
fi

echo "Benchmark completed successfully! Results saved to ${OUTPUT_DIR}"
