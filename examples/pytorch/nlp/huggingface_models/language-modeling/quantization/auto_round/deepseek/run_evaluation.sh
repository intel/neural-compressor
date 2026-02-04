#!/bin/bash
set -e

# Usage: ./run_evaluation.sh -m [model_path] -s [mxfp4|mxfp8] -t [task_name] -tp [tensor_parallel_size] -b [batch_size]
# Default values
MODEL_PATH=""
SCHEME="mxfp8"
TASK_NAME="piqa,hellaswag,mmlu"
TP_SIZE=8
BATCH_SIZE=512
KV_CACHE_DTYPE="auto"
ATTN_DTYPE="None"

# Function to display usage
usage() {
    echo "Usage: $0 -m [model_path] -s [mxfp4|mxfp8|nvfp4] -t [task_name] -tp [tensor_parallel_size] -b [batch_size]"
    echo "  -m: Path to the quantized model (required)"
    echo "  -s: Quantization scheme (mxfp4 or mxfp8, default: mxfp8)"
    echo "  -t: Task name(s) to evaluate (default: piqa,hellaswag,mmlu)"
    echo "  -tp: Tensor parallelism size (default: 8)"
    echo "  -b: Batch size (default: 512)"
    echo ""
    echo "Examples:"
    echo "  $0 -m /path/to/model -s mxfp4 -t gsm8k -tp 4 -b 256"
    echo "  $0 -m /path/to/model -s mxfp8 -t piqa,hellaswag -tp 8 -b 512"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m)
            MODEL_PATH="$2"
            shift 2
            ;;
        -s)
            SCHEME="$2"
            shift 2
            ;;
        -t)
            TASK_NAME="$2"
            shift 2
            ;;
        -tp)
            TP_SIZE="$2"
            shift 2
            ;;
        -kv)
            KV_CACHE_DTYPE="$2"
            shift 2
            ;;
        -attn)
            ATTN_DTYPE="$2"
            shift 2
            ;;
        -b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Invalid option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: Model path (-m) is required."
    usage
    exit 1
fi

# Extract model name and set output directory
MODEL_NAME=$(basename ${MODEL_PATH})
OUTPUT_DIR="${MODEL_NAME}-tp${TP_SIZE}-eval"

# Create output directory
mkdir -p ${OUTPUT_DIR}
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
    VLLM_ENABLE_AR_EXT=1
elif [[ "$SCHEME" == "nvfp4" ]]; then
    VLLM_AR_MXFP4_MODULAR_MOE=0
    VLLM_MXFP4_PRE_UNPACK_TO_FP8=0
    VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
    VLLM_ENABLE_STATIC_MOE=0
    VLLM_USE_DEEP_GEMM=0
    VLLM_ENABLE_AR_EXT=0
elif [[ "$SCHEME" == "mxfp8" ]]; then
    VLLM_AR_MXFP4_MODULAR_MOE=0
    VLLM_MXFP4_PRE_UNPACK_TO_FP8=0
    VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
    VLLM_ENABLE_STATIC_MOE=0
    VLLM_USE_DEEP_GEMM=0
    VLLM_ENABLE_AR_EXT=1
elif [[ "$SCHEME" == "fp8" ]]; then
    echo "Run original model."
    VLLM_USE_DEEP_GEMM=0
else
    echo "Error: Invalid quantization scheme (-s). Must be 'mxfp4', 'nvfp4' or 'mxfp8'."
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
echo "Output directory: ${OUTPUT_DIR}"

VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ENABLE_AR_EXT=$VLLM_ENABLE_AR_EXT \
VLLM_AR_MXFP4_MODULAR_MOE=$VLLM_AR_MXFP4_MODULAR_MOE \
VLLM_MXFP4_PRE_UNPACK_TO_FP8=$VLLM_MXFP4_PRE_UNPACK_TO_FP8 \
VLLM_MXFP4_PRE_UNPACK_WEIGHTS=$VLLM_MXFP4_PRE_UNPACK_WEIGHTS \
VLLM_ENABLE_STATIC_MOE=$VLLM_ENABLE_STATIC_MOE \
VLLM_USE_DEEP_GEMM=$VLLM_USE_DEEP_GEMM \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
lm_eval --model vllm \
  --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=${TP_SIZE},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,kv_cache_dtype=${KV_CACHE_DTYPE}" \
  --tasks $TASK_NAME \
  --batch_size $BATCH_SIZE \
  --log_samples \
  --seed 42 \
  --output_path ${OUTPUT_DIR} \
  --show_config 2>&1 | tee ${OUTPUT_DIR}/log.txt
