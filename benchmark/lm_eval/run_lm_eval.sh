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
#   ROPE_SCALING_JSON          enables rope scaling on `vllm serve` when set
#   SKIP_ROPE_SCALING_PATTERN  model name substring that disables rope scaling

set -eo pipefail

MODEL_PATH=""
TASKS="hellaswag,piqa,mmlu,gsm8k"
SCHEME="${DEFAULT_SCHEME:-}"
BATCH_SIZE="auto"
GPU_MEMORY_UTILIZATION=0.8
MAX_MODEL_LEN=8192
STATIC_KV_DTYPE="auto"
STATIC_ATTENTION_DTYPE="auto"
RULER_MAX_POS="${RULER_MAX_POS:-131072}"
LIMIT=""

SERVER_PORT="${SERVER_PORT:-8000}"
# Tasks that must run one at a time with a chat template applied.
CHAT_TEMPLATE_TASKS="gsm8k_llama mmlu_llama"

usage() {
    echo "Usage: CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>]"
    echo "  --model_path               Path to the quantized model (required)"
    echo "  --tasks                    Task name(s) to evaluate (default: ${TASKS})"
    echo "  --scheme                   Quantization scheme: mxfp4, mxfp8, nvfp4, bf16 (default: ${SCHEME:-unset})"
    echo "  --batch_size               Batch size (default: auto)"
    echo "  --gpu_memory_utilization   GPU memory utilization (default: 0.8)"
    echo "  --max_model_len            Max model length for standard tasks (default: 8192)"
    echo "  --static_kv_dtype          Data type for static kv cache (default: auto)"
    echo "  --static_attention_dtype   Data type for static attention cache (default: auto)"
    echo "  --ruler_max_pos            Max position length for RULER eval (default: ${RULER_MAX_POS:-131072})"
    echo "  --limit                    Limit the number of samples per task (RULER defaults to 100)"
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
        --scheme=*)
            SCHEME="${arg#*=}"
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
        --limit=*)
            LIMIT="${arg#*=}"
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

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required."
    usage
fi

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
max_length=$MAX_MODEL_LEN
max_gen_toks=2048
SEQ_LENGTHS=""

if [[ "$TASKS" == *"longbench"* ]]; then
    max_length=131072
    max_ctx_length=$((max_length - max_gen_toks))
fi

if [[ "$TASKS" == *"ruler"* ]] || [[ "$TASKS" == *"niah_multiquery"* ]]; then
    max_gen_toks=128
    MODEL_MAX_POS=${RULER_MAX_POS:-131072}
    if [[ "$TASKS" == *"ruler_qa_squad"* ]]; then
        TASKS="ruler_qa_squad"
    else
        TASKS="niah_multiquery"
    fi
    # Leave room for generated tokens so prompt + output stays within the server context limit.
    max_length=${MODEL_MAX_POS}
    SEQ_LENGTHS=$((MODEL_MAX_POS - max_gen_toks))
    BATCH_SIZE=32
    LIMIT="${LIMIT:-100}"
fi

# Set environment variables based on the quantization scheme
case "$SCHEME" in
    mxfp4)
        VLLM_AR_MXFP4_MODULAR_MOE=1
        VLLM_MXFP4_PRE_UNPACK_TO_FP8=1
        VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
        VLLM_ENABLE_STATIC_MOE=0
        VLLM_USE_DEEP_GEMM=0
        ;;
    mxfp8|nvfp4)
        VLLM_AR_MXFP4_MODULAR_MOE=0
        VLLM_MXFP4_PRE_UNPACK_TO_FP8=0
        VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
        VLLM_ENABLE_STATIC_MOE=0
        VLLM_USE_DEEP_GEMM=0
        ;;
    bf16|fp8)
        VLLM_USE_DEEP_GEMM=0
        ;;
    "")
        ;;
    *)
        echo "Error: Invalid quantization scheme (--scheme): '$SCHEME'."
        usage
        ;;
esac

LM_EVAL_EXTRA_ARGS="${EXTRA_LM_EVAL_MODEL_ARGS:-}"
SERVE_EXTRA_ARGS="${EXTRA_SERVE_ARGS:-}"

if [[ "$STATIC_KV_DTYPE" == "fp8" ]]; then
    export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=1
    export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND_FP8KV:-FLASHINFER}"
    if [[ -n "${FLASHINFER_WORKSPACE_SIZE:-}" ]]; then
        export VLLM_AR_FLASHINFER_WORKSPACE_BUFFER_SIZE="${FLASHINFER_WORKSPACE_SIZE}"
    fi
    echo "Using FP8 for KV cache"
fi

if [[ "$STATIC_ATTENTION_DTYPE" == "fp8" ]]; then
    KV_CACHE_DTYPE="fp8"
    backend="${ATTENTION_BACKEND_FP8ATTN:-FLASHINFER}"
    if [[ "$backend" == "TRITON_ATTN" ]]; then
        SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS} --attention-backend TRITON_ATTN"
        LM_EVAL_EXTRA_ARGS="${LM_EVAL_EXTRA_ARGS},attention_backend=TRITON_ATTN"
    else
        export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=0
        export VLLM_ATTENTION_BACKEND="$backend"
    fi
    echo "Using FP8 Attention with ${backend} backend"
fi

echo "Running benchmark with parameters:"
echo "  Model Path: ${MODEL_PATH}"
echo "  Tasks: ${TASKS}"
echo "  Quantization scheme: ${SCHEME:-unset}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Max Model Length: ${max_length}"
echo "  KV Cache Dtype: ${KV_CACHE_DTYPE}"
echo "  GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  Tensor Parallel Size: ${TP_SIZE}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  Output directory: ${OUTPUT_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_AR_MXFP4_MODULAR_MOE=${VLLM_AR_MXFP4_MODULAR_MOE:-}
export VLLM_MXFP4_PRE_UNPACK_TO_FP8=${VLLM_MXFP4_PRE_UNPACK_TO_FP8:-}
export VLLM_MXFP4_PRE_UNPACK_WEIGHTS=${VLLM_MXFP4_PRE_UNPACK_WEIGHTS:-}
export VLLM_ENABLE_STATIC_MOE=${VLLM_ENABLE_STATIC_MOE:-}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-}
# For https://github.com/yiliu30/vllm-qdq-plugin.git CT format eval on GPUs other than SM 10.0.
# nvidia-smi does not honor CUDA_VISIBLE_DEVICES; select its first GPU explicitly.
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
CUDA_SM=$(nvidia-smi --id="${GPU_ID:-0}" --query-gpu=compute_cap --format=csv,noheader,nounits)
if [[ "$CUDA_SM" != "10.0" ]]; then
    export VLLM_QDQ=1
    export VLLM_MXFP4_USE_MARLIN=1
fi

run_standard_eval() {
    local tasks=$1
    local extra_args=$2

    echo "Running evaluation for tasks: ${tasks}"
    lm_eval --model vllm \
        --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=${TP_SIZE},max_model_len=${MAX_MODEL_LEN},max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,kv_cache_dtype=${KV_CACHE_DTYPE}${LM_EVAL_EXTRA_ARGS}" \
        --tasks "$tasks" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --seed 42 \
        ${LIMIT:+--limit $LIMIT} \
        --output_path "$OUTPUT_DIR" \
        $extra_args \
        --show_config 2>&1 | tee -a "${OUTPUT_DIR}/log.txt"
}

start_vllm_server() {
    echo "Starting vLLM server on port ${SERVER_PORT}..."

    local rope_args=()
    if [[ -n "${ROPE_SCALING_JSON:-}" ]]; then
        # vLLM >= 0.19 removed --rope-scaling; use --hf-overrides instead
        local version major_minor
        version=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "0.0.0")
        major_minor=$(echo "$version" | awk -F. '{printf "%d%02d", $1, $2}')
        if [ "$major_minor" -ge 19 ] 2>/dev/null; then
            rope_args=("--hf-overrides" "{\"rope_scaling\":${ROPE_SCALING_JSON},\"max_position_embeddings\":131072}")
        else
            rope_args=("--rope-scaling" "${ROPE_SCALING_JSON}")
        fi
        if [[ -n "${SKIP_ROPE_SCALING_PATTERN:-}" && "$MODEL_NAME" == *"${SKIP_ROPE_SCALING_PATTERN}"* ]]; then
            rope_args=()
            echo "Skipping rope scaling for model: ${MODEL_NAME}"
        fi
    fi

    vllm serve "${MODEL_PATH}" \
        --port "${SERVER_PORT}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-model-len "${max_length}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --dtype bfloat16 \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        "${rope_args[@]}" \
        ${SERVE_EXTRA_ARGS} \
        > "${OUTPUT_DIR}/vllm_server.log" 2>&1 &
    VLLM_PID=$!
    echo "vLLM server started with PID: ${VLLM_PID}"
}

wait_for_server() {
    local max_retries=300
    local retry_count=0

    echo "Waiting for vLLM server to be ready..."
    while [ $retry_count -lt $max_retries ]; do
        if curl -s "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi
        retry_count=$((retry_count + 1))
        echo "Waiting for server... (${retry_count}/${max_retries})"
        sleep 5
    done

    echo "Error: vLLM server failed to start within expected time"
    echo "Check ${OUTPUT_DIR}/vllm_server.log for details"
    return 1
}

cleanup_server() {
    echo "Shutting down vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "Server stopped"
}

start_server_or_exit() {
    start_vllm_server
    if ! wait_for_server; then
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    trap cleanup_server EXIT INT TERM
}

run_longbench_eval() {
    start_server_or_exit

    echo "Running LongBench evaluation against vLLM server..."
    python -m long_bench_eval.cli \
        --api-key dummy \
        --base-url "http://localhost:${SERVER_PORT}/v1" \
        --model "${MODEL_PATH}" \
        --max-context-length "${max_ctx_length}" \
        --num-threads 1 \
        --deterministic \
        --categories "Long In-context Learning"
}

run_ruler_eval() {
    start_server_or_exit

    echo "Running RULER evaluation against vLLM server..."
    lm_eval \
        --model local-completions \
        --model_args "model=${MODEL_PATH},base_url=http://localhost:${SERVER_PORT}/v1/completions,num_concurrent=1,max_retries=50,timeout=500,tokenized_requests=False,max_gen_toks=${max_gen_toks},max_length=${SEQ_LENGTHS}" \
        --tasks "${TASKS}" \
        --metadata="{\"max_seq_lengths\":[${SEQ_LENGTHS}],\"tokenizer\":\"${MODEL_PATH}\"}" \
        --gen_kwargs "max_gen_toks=${max_gen_toks}" \
        --batch_size "${BATCH_SIZE}" \
        ${LIMIT:+--limit $LIMIT} \
        --log_samples \
        --output_path "${OUTPUT_DIR}/seq_${SEQ_LENGTHS}" \
        --seed 42
}

# Split out tasks that need a chat template so they run one at a time.
split_tasks() {
    local remaining=""
    local special=""
    IFS=',' read -ra task_array <<< "$TASKS"
    for task in "${task_array[@]}"; do
        if [[ " $CHAT_TEMPLATE_TASKS " == *" $task "* ]]; then
            special="${special:+$special,}$task"
        else
            remaining="${remaining:+$remaining,}$task"
        fi
    done
    OTHER_TASKS="$remaining"
    SPECIAL_TASKS="$special"
}

if [[ "$TASKS" == *"longbench"* ]]; then
    echo "Running LongBench evaluation..."
    run_longbench_eval
elif [[ "$TASKS" == "niah_multiquery" ]] || [[ "$TASKS" == "ruler_qa_squad" ]]; then
    echo "Running RULER evaluation..."
    run_ruler_eval
else
    split_tasks
    if [[ -n "$OTHER_TASKS" ]]; then
        run_standard_eval "$OTHER_TASKS" ""
    fi
    if [[ -n "$SPECIAL_TASKS" ]]; then
        IFS=',' read -ra special_array <<< "$SPECIAL_TASKS"
        for special_task in "${special_array[@]}"; do
            run_standard_eval "$special_task" "--apply_chat_template --fewshot_as_multiturn"
        done
    fi
fi

echo "Benchmark completed successfully! Results saved to ${OUTPUT_DIR}"
