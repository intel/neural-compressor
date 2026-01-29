#!/bin/bash
set -e

# Model Testing Script
# Usage: ./run_generate.sh -s [mxfp4|mxfp8] -m [model_path] -tp [tensor_parallel_size]

# Default values
QUANT_TYPE="mxfp8"
MODEL_PATH="/path/to/quantized_model"
TP_SIZE=8
KV_CACHE_DTYPE="auto"
ATTN_DTYPE="None"


# Function to display usage
usage() {
    echo "Usage: $0 -s [mxfp4|mxfp8] -m [model_path] -tp [tensor_parallel_size]"
    echo "  -s: Quantization scheme (mxfp4 or mxfp8, default: mxfp8)"
    echo "  -m: Path to quantized model (required)"
    echo "  -tp: Tensor parallelism size (default: 8)"
    echo "  -kv: Data type for KV cache (default: auto)"
    echo "  -attn: Data type for Attention cache (default: None)"
    echo ""
    echo "Examples:"
    echo "  $0 -s mxfp4 -m /path/to/my/model -tp 4"
    echo "  $0 -m /path/to/my/model"
    echo "  $0 -s mxfp8 -m /path/to/my/model"
    echo "  $0 -kv fp8 -m /path/to/my/model"
    echo "  $0 -m /path/to/my/model -attn fp8"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s)
            QUANT_TYPE="$2"
            shift 2
            ;;
        -m)
            MODEL_PATH="$2"
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
        -h)
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


# Validate quantization type
QUANT_TYPE_UPPER=$(echo "$QUANT_TYPE" | tr '[:lower:]' '[:upper:]')
if [[ "$QUANT_TYPE_UPPER" != "MXFP4" && "$QUANT_TYPE_UPPER" != "MXFP8" && "$QUANT_TYPE_UPPER" != "NVFP4" ]]; then
    echo "Error: Quantization type must be mxfp4, mxfp8 or nvfp4"
    usage
    exit 1
fi

# Validate model path
if [[ "$MODEL_PATH" == "/path/to/quantized_model" ]]; then
    echo "Error: Model path is required (-m)"
    usage
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model path '$MODEL_PATH' does not exist or is not a directory"
    exit 1
fi

# Validate TP_SIZE is a number
if ! [[ "$TP_SIZE" =~ ^[0-9]+$ ]] || [ "$TP_SIZE" -lt 1 ]; then
    echo "Error: Tensor parallelism size must be a positive integer"
    exit 1
fi

echo "Running $QUANT_TYPE_UPPER test with:"
echo "  Model: $MODEL_PATH"
echo "  Tensor Parallelism: $TP_SIZE"
echo "  KV Cache Dtype: $KV_CACHE_DTYPE"
echo "  Attention Dtype: $ATTN_DTYPE"
echo ""

#FIXME: (yiliu30) remove these envs once we have fixed the pynccl issues
export NCCL_NVLS_ENABLE=0
export VLLM_DISABLE_PYNCCL=1

# Set environment variables based on quantization type
# Set environment variables based on quantization type
if [[ "$QUANT_TYPE_UPPER" == "MXFP4" ]]; then
    export VLLM_ENABLE_AR_EXT=1
    export VLLM_AR_MXFP4_MODULAR_MOE=1
    export VLLM_MXFP4_PRE_UNPACK_TO_FP8=1
    echo "Using MXFP4 configuration"
elif [[ "$QUANT_TYPE_UPPER" == "NVFP4" ]]; then
    export VLLM_ENABLE_AR_EXT=0
    export VLLM_AR_MXFP4_MODULAR_MOE=0
    export VLLM_MXFP4_PRE_UNPACK_TO_FP8=0
    echo "Using NVFP4 configuration"
else
    export VLLM_ENABLE_AR_EXT=1
    export VLLM_AR_MXFP4_MODULAR_MOE=0
    export VLLM_MXFP4_PRE_UNPACK_TO_FP8=0
    echo "Using MXFP8 configuration"
fi

# for fp8 kv cache
if [[ "$KV_CACHE_DTYPE" == "fp8" ]]; then
    export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=1
    export VLLM_ATTENTION_BACKEND="FLASHINFER_MLA"
    # 1024 * 1024 * 1024
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

# Common environment variables
export VLLM_ENABLE_STATIC_MOE=0
export VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "Environment variables set:"
echo "  VLLM_AR_MXFP4_MODULAR_MOE=$VLLM_AR_MXFP4_MODULAR_MOE"
echo "  VLLM_MXFP4_PRE_UNPACK_TO_FP8=$VLLM_MXFP4_PRE_UNPACK_TO_FP8"
echo "  VLLM_ENABLE_AR_EXT=$VLLM_ENABLE_AR_EXT"
echo ""

# Run the model
echo "Starting model generation..."
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python generate.py \
    --model "${MODEL_PATH}" \
    --tensor_parallel_size $TP_SIZE \
    --max-tokens 16 \
    --max-num-seqs 4 \
    --max-model-len 2048 \
    --gpu_memory_utilization 0.75 \
    --no-enable-prefix-caching  \
    --kv-cache-dtype $KV_CACHE_DTYPE
