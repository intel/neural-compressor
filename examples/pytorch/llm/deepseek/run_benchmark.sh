#!/bin/bash
set -e

# Usage: CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>] [--batch_size=<size>]
# Default values
MODEL_PATH=""
SCHEME="mxfp8"
TASK_NAME="piqa,hellaswag,mmlu"
BATCH_SIZE=512
GPU_MEMORY_UTILIZATION=0.8
KV_CACHE_DTYPE="auto"
ATTN_DTYPE="auto"
SEQ_LENGTHS=""
RULER_MAX_POS=""

# Function to display usage
usage() {
    echo "Usage: bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>] [--batch_size=<size>]"
    echo "  --model_path               Path to the quantized model (required)"
    echo "  --scheme                   Quantization scheme: mxfp4, mxfp8, nvfp4, fp8 (default: mxfp8)"
    echo "  --tasks                    Task name(s) to evaluate (default: piqa,hellaswag,mmlu)"
    echo "  --batch_size               Batch size (default: 512)"
    echo "  --gpu_memory_utilization   GPU memory utilization (default: 0.8)"
    echo "  --static_kv_dtype          Data type for static kv cache (default: auto)"
    echo "  --static_attention_dtype   Data type for static attention cache (default: auto)"
    echo "  --ruler_max_pos            Max position length for RULER eval (default: 131072)"
    echo ""
    echo "Examples:"
    echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --model_path=/path/to/model --scheme=mxfp4 --tasks=gsm8k --batch_size=256"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path=*)
            MODEL_PATH="${1#*=}"
            shift
            ;;
        --scheme=*)
            SCHEME="${1#*=}"
            shift
            ;;
        --tasks=*)
            TASK_NAME="${1#*=}"
            shift
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --gpu_memory_utilization=*)
            GPU_MEMORY_UTILIZATION="${1#*=}"
            shift
            ;;
        --static_kv_dtype=*)
            KV_CACHE_DTYPE="${1#*=}"
            shift
            ;;
        --static_attention_dtype=*)
            ATTN_DTYPE="${1#*=}"
            shift
            ;;
        --ruler_max_pos=*)
            RULER_MAX_POS="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: Model path (--model_path) is required."
    usage
    exit 1
fi

# Count available GPUs and set tensor_parallel_size
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    TP_SIZE=${#GPU_ARRAY[@]}
else
    TP_SIZE=1
fi

# Extract model name and set output directory
MODEL_NAME=$(basename ${MODEL_PATH})
OUTPUT_DIR="${MODEL_NAME}-tp${TP_SIZE}-eval"
# Create output directory
mkdir -p ${OUTPUT_DIR}


SERVER_PORT=8000
max_length=8192
max_gen_toks=2048

# update max_length based on the task
if [[ "$TASK_NAME" == *"longbench"* ]]; then
    max_length=131072 # 128k
    max_gen_toks=2048
fi

# update max_length based on the task
if [[ "$TASK_NAME" == *"ruler"* ]] || [[ "$TASK_NAME" == *"niah_multiquery"* ]]; then
    max_gen_toks=128
    MODEL_MAX_POS=${RULER_MAX_POS:-131072}
    if [[ "$TASK_NAME" == *"ruler_qa_squad"* ]]; then
        # if input task is ruler_qa_squad
        MODEL_MAX_POS=$((131072 - max_gen_toks))
        TASK_NAME="ruler_qa_squad"
    else
        # if input task is ruler or niah_multiquery
        TASK_NAME="niah_multiquery"
    fi
    max_length=${MODEL_MAX_POS}
    SEQ_LENGTHS=${MODEL_MAX_POS}
    BATCH_SIZE=32
fi

max_ctx_length=$((max_length - max_gen_toks))

echo "max_length: ${max_length}"
echo "max_gen_toks: ${max_gen_toks}"
echo "max_ctx_length: ${max_ctx_length}"


#FIXME: (yiliu30) remove these envs once we have fixed the pynccl issues
export NCCL_NVLS_ENABLE=0
# export VLLM_DISABLE_PYNCCL=1
# Set environment variables based on the quantization scheme
if [[ "$SCHEME" == "mxfp4" ]]; then
    VLLM_AR_MXFP4_MODULAR_MOE=1
    VLLM_MXFP4_PRE_UNPACK_TO_FP8=1
    VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
    VLLM_ENABLE_STATIC_MOE=0
    VLLM_USE_DEEP_GEMM=0
elif [[ "$SCHEME" == "nvfp4" ]]; then
    VLLM_AR_MXFP4_MODULAR_MOE=0
    VLLM_MXFP4_PRE_UNPACK_TO_FP8=0
    VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
    VLLM_ENABLE_STATIC_MOE=0
    VLLM_USE_DEEP_GEMM=0
elif [[ "$SCHEME" == "mxfp8" ]]; then
    VLLM_AR_MXFP4_MODULAR_MOE=0
    VLLM_MXFP4_PRE_UNPACK_TO_FP8=0
    VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
    VLLM_ENABLE_STATIC_MOE=0
    VLLM_USE_DEEP_GEMM=0
    export VLLM_DISABLE_PYNCCL=1
elif [[ "$SCHEME" == "fp8" ]]; then
    echo "Run original model."
    VLLM_USE_DEEP_GEMM=0
else
    echo "Error: Invalid quantization scheme (--scheme). Must be 'mxfp4', 'nvfp4' or 'mxfp8'."
    usage
    exit 1
fi

# for fp8 kv cache
if [[ "$KV_CACHE_DTYPE" == "fp8" ]]; then
    export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=1
    export VLLM_ATTENTION_BACKEND="FLASHINFER_MLA"
    # 512 * 1024 * 1024
    export VLLM_AR_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
    echo "Using FP8 for KV cache"
fi

# for fp8 attention cache
if [[ "$ATTN_DTYPE" == "fp8" ]]; then
    export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=0
    export VLLM_ATTENTION_BACKEND="FLASHINFER_MLA"
    KV_CACHE_DTYPE="fp8"
    echo "Using FP8 Attention"
fi

# Run evaluation
echo "Evaluating model: ${MODEL_PATH}"
echo "Quantization scheme: ${SCHEME}"
echo "Tasks: ${TASK_NAME}"
echo "Tensor parallelism size: ${TP_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Output directory: ${OUTPUT_DIR}"



# Export vLLM environment variables
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_AR_MXFP4_MODULAR_MOE=$VLLM_AR_MXFP4_MODULAR_MOE
export VLLM_MXFP4_PRE_UNPACK_TO_FP8=$VLLM_MXFP4_PRE_UNPACK_TO_FP8
export VLLM_MXFP4_PRE_UNPACK_WEIGHTS=$VLLM_MXFP4_PRE_UNPACK_WEIGHTS
export VLLM_ENABLE_STATIC_MOE=$VLLM_ENABLE_STATIC_MOE
export VLLM_USE_DEEP_GEMM=$VLLM_USE_DEEP_GEMM
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# For https://github.com/yiliu30/vllm-qdq-plugin.git CT format eval
export VLLM_QDQ=1
# A100 need to close torch compile
# export TORCH_COMPILE_DISABLE=1

# Function to run standard lm-eval tasks
run_standard_eval() {
    lm_eval --model vllm \
        --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=${TP_SIZE},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,kv_cache_dtype=${KV_CACHE_DTYPE}" \
        --tasks $TASK_NAME \
        --batch_size $BATCH_SIZE \
        --log_samples \
        --seed 42 \
        --output_path ${OUTPUT_DIR} \
        --show_config 2>&1 | tee ${OUTPUT_DIR}/log.txt
}

# Function to start vLLM server
start_vllm_server() {
    echo "Starting vLLM server on port ${SERVER_PORT}..."

    # Detect vLLM version for backward-compatible rope scaling
    # vLLM >= 0.19 removed --rope-scaling; use --hf-overrides instead
    VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "0.0.0")
    VLLM_MAJOR_MINOR=$(echo "$VLLM_VERSION" | awk -F. '{printf "%d%02d", $1, $2}')
    ROPE_SCALING_JSON='{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
    if [ "$VLLM_MAJOR_MINOR" -ge 19 ] 2>/dev/null; then
        ROPE_FLAG="--hf-overrides"
        ROPE_VALUE="{\"rope_scaling\":${ROPE_SCALING_JSON},\"max_position_embeddings\":131072}"
    else
        ROPE_FLAG="--rope-scaling"
        ROPE_VALUE="${ROPE_SCALING_JSON}"
    fi
    echo "vLLM version: ${VLLM_VERSION}, using: ${ROPE_FLAG} '${ROPE_VALUE}'"

    vllm serve ${MODEL_PATH} \
        --port ${SERVER_PORT} \
        --tensor-parallel-size ${TP_SIZE} \
        --max-model-len ${max_length} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --dtype bfloat16 \
        --kv-cache-dtype ${KV_CACHE_DTYPE} \
        > ${OUTPUT_DIR}/vllm_server.log 2>&1 &
    
    VLLM_PID=$!
    echo "vLLM server started with PID: ${VLLM_PID}"
}

# Function to wait for vLLM server to be ready
wait_for_server() {
    local max_retries=300
    local retry_count=0
    
    echo "Waiting for vLLM server to be ready..."
    while [ $retry_count -lt $max_retries ]; do
        if curl -s http://localhost:${SERVER_PORT}/health > /dev/null 2>&1; then
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

# Function to cleanup vLLM server on exit
cleanup_server() {
    echo "Shutting down vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "Server stopped"
}



# Function to run longbench evaluation via API
run_longbench_eval() {
    start_vllm_server
    
    if ! wait_for_server; then
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    
    # Setup cleanup trap
    trap cleanup_server EXIT INT TERM
    
    # Run LongBench evaluation
    echo "Running LongBench evaluation against vLLM server..."
    python -m long_bench_eval.cli \
        --api-key dummy \
        --base-url http://localhost:${SERVER_PORT}/v1 \
        --model ${MODEL_PATH} \
        --max-context-length ${max_ctx_length} \
        --num-threads 1 \
        --deterministic \
        --categories "Long In-context Learning"
    
    echo "Evaluation completed! Results saved to ${OUTPUT_DIR}"
}

# Function to run ruler evaluation via API
run_ruler_eval() {
    start_vllm_server

    if ! wait_for_server; then
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi

    # Setup cleanup trap
    trap cleanup_server EXIT INT TERM

    # Run Ruler evaluation
    echo "Running Ruler evaluation against vLLM server..."
    lm_eval \
        --model local-completions \
        --model_args "model=$MODEL_PATH,base_url=http://localhost:${SERVER_PORT}/v1/completions,num_concurrent=1,max_retries=50,timeout=500,tokenized_requests=False,max_gen_toks=${max_gen_toks},max_length=${max_length}" \
        --tasks $TASK_NAME \
        --metadata="{\"max_seq_lengths\":[${SEQ_LENGTHS}],\"tokenizer\":\"${MODEL_PATH}\"}" \
        --gen_kwargs "max_gen_toks=${max_gen_toks}" \
        --batch_size ${BATCH_SIZE} \
        --output_path "${OUTPUT_DIR}/seq_${SEQ_LENGTHS}" \
        --seed 42

    echo "Evaluation completed! Results saved to ${OUTPUT_DIR}"
}


# Main evaluation logic
if [[ "$TASK_NAME" == *"longbench"* ]]; then
    echo "Running LongBench v2 evaluation..."
    run_longbench_eval
elif [[ "$TASK_NAME" == *"ruler"* ]] || [[ "$TASK_NAME" == *"niah_multiquery"* ]]; then
    echo "Running RULER evaluation..."
    run_ruler_eval
else
    echo "Running standard lm-eval tasks..."
    run_standard_eval
fi
