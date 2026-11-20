#!/bin/bash

# Usage: CUDA_VISIBLE_DEVICES=0 bash run_quant.sh --dtype=mxfp8 --input_model=/models/Meta-Llama-3.1-8B-Instruct --output_model=Llama-3.1-8B-MXFP8

# Parse command line arguments
STATIC_KV_DTYPE="auto"
STATIC_ATTENTION_DTYPE="auto"
EXPORT_FORMAT="llm_compressor"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dtype=*)
            DTYPE="${1#*=}"
            shift
            ;;
        --input_model=*)
            INPUT_MODEL="${1#*=}"
            shift
            ;;
        --output_model=*)
            OUTPUT_MODEL="${1#*=}"
            shift
            ;;
        --static_kv_dtype=*)
            STATIC_KV_DTYPE="${1#*=}"
            shift
            ;;
        --static_attention_dtype=*)
            STATIC_ATTENTION_DTYPE="${1#*=}"
            shift
            ;;
        --export_format=*)
            EXPORT_FORMAT="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$DTYPE" || -z "$INPUT_MODEL" || -z "$OUTPUT_MODEL" ]]; then
    echo "Usage: bash run_quant.sh --dtype=<dtype> --input_model=<input_model> --output_model=<output_model>"
    echo "Supported dtypes: mxfp8, mxfp4, mxfp4_mixed, nvfp4, unvfp4"
    exit 1
fi

# Detect the model topology from the input model name
MODEL_LOWER=$(echo "$INPUT_MODEL" | tr '[:upper:]' '[:lower:]')
if [[ "$MODEL_LOWER" == *llama-3.3* && "$MODEL_LOWER" == *70b* ]]; then
    TOPOLOGY="Llama-3.3-70B"
elif [[ "$MODEL_LOWER" == *llama-3.1* && "$MODEL_LOWER" == *70b* ]]; then
    TOPOLOGY="Llama-3.1-70B"
elif [[ "$MODEL_LOWER" == *llama-3.1* && "$MODEL_LOWER" == *8b* ]]; then
    TOPOLOGY="Llama-3.1-8B"
else
    echo "Error: Cannot detect topology from '$INPUT_MODEL'."
    echo "Supported topologies: Llama-3.1-8B, Llama-3.3-70B, Llama-3.1-70B"
    exit 1
fi

echo "Starting quantization with parameters:"
echo "  Topology: $TOPOLOGY"
echo "  Data Type: $DTYPE"
echo "  Input Model: $INPUT_MODEL"
echo "  Output Model: $OUTPUT_MODEL"

# Set common parameters
COMMON_ARGS=(--quantize --enable_torch_compile --low_gpu_mem_usage --export_format "$EXPORT_FORMAT")
if [ "$STATIC_KV_DTYPE" != "auto" ]; then
    COMMON_ARGS+=(--static_kv_dtype "$STATIC_KV_DTYPE")
fi
if [ "$STATIC_ATTENTION_DTYPE" != "auto" ]; then
    COMMON_ARGS+=(--static_attention_dtype "$STATIC_ATTENTION_DTYPE")
fi

case "$TOPOLOGY" in
    "Llama-3.1-8B")
        case "$DTYPE" in
            "mxfp8")
                echo "Running Llama 3.1 8B MXFP8 quantization..."
                CMD="python quantize.py --model_name_or_path \"$INPUT_MODEL\" ${COMMON_ARGS[*]} --dtype MXFP8 --iters 0 --export_path \"$OUTPUT_MODEL\""
                echo "Executing command: $CMD"
                python quantize.py \
                    --model_name_or_path "$INPUT_MODEL" \
                    "${COMMON_ARGS[@]}" \
                    --dtype MXFP8 \
                    --iters 0 \
                    --export_path "$OUTPUT_MODEL"
                ;;
            "mxfp4")
                echo "Running Llama 3.1 8B MXFP4 quantization..."
                CMD="python quantize.py --model_name_or_path \"$INPUT_MODEL\" ${COMMON_ARGS[*]} --dtype MXFP4  --iters 0 --export_path \"$OUTPUT_MODEL\""
                echo "Executing command: $CMD"
                python quantize.py \
                    --model_name_or_path "$INPUT_MODEL" \
                    "${COMMON_ARGS[@]}" \
                    --dtype MXFP4 \
                    --iters 0 \
                    --export_path "$OUTPUT_MODEL"
                ;;
            "mxfp4_mixed")
                echo "Running Llama 3.1 8B MXFP4 (Mixed with MXFP8) quantization..."
                CMD="python quantize.py --model_name_or_path \"$INPUT_MODEL\" ${COMMON_ARGS[*]} --target_bits 7.8 --iters 0 --options \"MXFP4\" \"MXFP8\" --shared_layers \"k_proj\" \"v_proj\" \"q_proj\" --shared_layers \"gate_proj\" \"up_proj\" --export_path \"$OUTPUT_MODEL\""
                echo "Executing command: $CMD"
                python quantize.py \
                    --model_name_or_path "$INPUT_MODEL" \
                    "${COMMON_ARGS[@]}" \
                    --target_bits 7.8 \
                    --options "MXFP4" "MXFP8" \
                    --shared_layers "k_proj" "v_proj" "q_proj" \
                    --iters 0 \
                    --shared_layers "gate_proj" "up_proj" \
                    --export_path "$OUTPUT_MODEL"
                ;;
            *)
                echo "Error: Unsupported dtype '$DTYPE' for topology '$TOPOLOGY'"
                echo "Supported dtypes for Llama-3.1-8B: mxfp8, mxfp4_mixed"
                exit 1
                ;;
        esac
        ;;
    "Llama-3.3-70B")
        case "$DTYPE" in
            "mxfp8")
                echo "Running Llama 3.3 70B MXFP8 quantization..."
                CMD="python quantize.py --model_name_or_path \"$INPUT_MODEL\" ${COMMON_ARGS[*]} --dtype MXFP8 --iters 0 --export_path \"$OUTPUT_MODEL\""
                echo "Executing command: $CMD"
                python quantize.py \
                    --model_name_or_path "$INPUT_MODEL" \
                    "${COMMON_ARGS[@]}" \
                    --dtype MXFP8 \
                    --iters 0 \
                    --export_path "$OUTPUT_MODEL"
                ;;
            "mxfp4")
                echo "Running Llama 3.3 70B MXFP4 quantization..."
                CMD="python quantize.py --model_name_or_path \"$INPUT_MODEL\" ${COMMON_ARGS[*]} --dtype MXFP4  --iters 0 --export_path \"$OUTPUT_MODEL\""
                echo "Executing command: $CMD"
                python quantize.py \
                    --model_name_or_path "$INPUT_MODEL" \
                    "${COMMON_ARGS[@]}" \
                    --dtype MXFP4 \
                    --iters 0 \
                    --export_path "$OUTPUT_MODEL"
                ;;
            "mxfp4_mixed")
                echo "Running Llama 3.3 70B MXFP4 (Mixed with MXFP8) quantization..."
                CMD="python quantize.py --model_name_or_path \"$INPUT_MODEL\" ${COMMON_ARGS[*]} --target_bits 5.8 --options \"MXFP4\" \"MXFP8\" --shared_layers \"k_proj\" \"v_proj\" \"q_proj\" --shared_layers \"gate_proj\" \"up_proj\" --export_path \"$OUTPUT_MODEL\""
                echo "Executing command: $CMD"
                python quantize.py \
                    --model_name_or_path "$INPUT_MODEL" \
                    "${COMMON_ARGS[@]}" \
                    --target_bits 5.8 \
                    --options "MXFP4" "MXFP8" \
                    --shared_layers "k_proj" "v_proj" "q_proj" \
                    --shared_layers "gate_proj" "up_proj" \
                    --export_path "$OUTPUT_MODEL"
                ;;
            *)
                echo "Error: Unsupported dtype '$DTYPE' for topology '$TOPOLOGY'"
                echo "Supported dtypes for Llama-3.3-70B: mxfp8, mxfp4_mixed"
                exit 1
                ;;
        esac
        ;;
    "Llama-3.1-70B")
        case "$DTYPE" in
            "mxfp8")
                echo "Running Llama 3.1 70B MXFP8 quantization..."
                CMD="python quantize.py --model_name_or_path \"$INPUT_MODEL\" ${COMMON_ARGS[*]} --dtype MXFP8 --iters 0 --export_path \"$OUTPUT_MODEL\""
                echo "Executing command: $CMD"
                python quantize.py \
                    --model_name_or_path "$INPUT_MODEL" \
                    "${COMMON_ARGS[@]}" \
                    --dtype MXFP8 \
                    --iters 0 \
                    --export_path "$OUTPUT_MODEL"
                ;;
            "nvfp4")
                echo "Running Llama 3.1 70B NVFP4 quantization..."
                CMD="python quantize.py --model_name_or_path \"$INPUT_MODEL\" --quantize --low_gpu_mem_usage --dtype NVFP4 --export_format llm_compressor --export_path \"$OUTPUT_MODEL\""
                echo "Executing command: $CMD"
                python quantize.py \
                    --model_name_or_path "$INPUT_MODEL" \
                    --quantize \
                    --low_gpu_mem_usage \
                    --dtype NVFP4 \
                    --export_format llm_compressor \
                    --export_path "$OUTPUT_MODEL"
                ;;
            "unvfp4")
                echo "Running Llama 3.1 70B uNVFP4 quantization..."
                CMD="python quantize.py --model_name_or_path \"$INPUT_MODEL\" --quantize --dtype uNVFP4 --quant_lm_head --iters 0 --enable_torch_compile --low_gpu_mem_usage --export_format fake --export_path \"$OUTPUT_MODEL\" --accuracy"
                echo "Executing command: $CMD"
                python quantize.py \
                    --model_name_or_path "$INPUT_MODEL" \
                    --quantize \
                    --dtype uNVFP4 \
                    --quant_lm_head \
                    --iters 0 \
                    --enable_torch_compile \
                    --low_gpu_mem_usage \
                    --export_format fake \
                    --export_path "$OUTPUT_MODEL" \
                    --accuracy
                ;;
            *)
                echo "Error: Unsupported dtype '$DTYPE' for topology '$TOPOLOGY'"
                echo "Supported dtypes for Llama-3.3-70B: mxfp8, mxfp4_mixed"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Error: Unsupported topology '$TOPOLOGY'"
        exit 1
        ;;
esac

echo "Quantization completed successfully!"
echo "Output model saved to: $OUTPUT_MODEL"
