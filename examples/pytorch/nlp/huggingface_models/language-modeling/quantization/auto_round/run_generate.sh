
# Model Testing Script
# Usage: ./run_generate.sh -s [mxfp4|mxfp8] -m [model_path] -tp [tensor_parallel_size]

# Default values
QUANT_TYPE="mxfp8"
MODEL_PATH="/path/to/quantized_model"
TP_SIZE=8

# Function to display usage
usage() {
    echo "Usage: $0 -s [mxfp4|mxfp8] -m [model_path] -tp [tensor_parallel_size]"
    echo "  -s: Quantization scheme (mxfp4 or mxfp8, default: mxfp8)"
    echo "  -m: Path to quantized model (required)"
    echo "  -tp: Tensor parallelism size (default: 8)"
    echo ""
    echo "Examples:"
    echo "  $0 -s mxfp4 -m /path/to/my/model -tp 4"
    echo "  $0 -m /path/to/my/model"
    echo "  $0 -s mxfp8 -m /path/to/my/model"
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
if [[ "$QUANT_TYPE_UPPER" != "MXFP4" && "$QUANT_TYPE_UPPER" != "MXFP8" ]]; then
    echo "Error: Quantization type must be mxfp4 or mxfp8"
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
echo ""

# Set environment variables based on quantization type
if [[ "$QUANT_TYPE_UPPER" == "MXFP4" ]]; then
    export VLLM_AR_MXFP4_MODULAR_MOE=1
    export VLLM_MXFP4_PRE_UNPACK_TO_FP8=1
    echo "Using MXFP4 configuration"
else
    export VLLM_AR_MXFP4_MODULAR_MOE=0
    export VLLM_MXFP4_PRE_UNPACK_TO_FP8=0
    echo "Using MXFP8 configuration"
fi

# Common environment variables
export VLLM_ENABLE_AR_EXT=1
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
python generate.py \
    --model "${MODEL_PATH}" \
    --tensor_parallel_size $TP_SIZE \
    --max-tokens 16 \
    --max-num-seqs 4 \
    --gpu_memory_utilization 0.75 \
    --no-enable-prefix-caching \
    --enable_expert_parallel