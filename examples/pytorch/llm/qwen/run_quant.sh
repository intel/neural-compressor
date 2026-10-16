#!/bin/bash
set -e

# Usage: CUDA_VISIBLE_DEVICES=0 bash run_quant.sh --dtype=mxfp4 --input_model=/models/Qwen3-8B --output_model=Qwen3-8B-MXFP4

DTYPE=""
INPUT_MODEL=""
OUTPUT_MODEL=""
EXPORT_FORMAT="llm_compressor"
STATIC_KV_DTYPE="auto"
STATIC_ATTENTION_DTYPE="auto"

usage() {
  echo "Usage: bash run_quant.sh --dtype=<dtype> --input_model=<input_model> --output_model=<output_model>"
  echo "  --dtype                    quantization data type, e.g. mxfp8, mxfp4, nvfp4"
  echo "  --input_model              Hugging Face model ID or local path"
  echo "  --output_model             output directory for the quantized model"
  echo "  --export_format            export format (default: llm_compressor)"
  echo "  --static_kv_dtype          data type for static kv cache (default: auto)"
  echo "  --static_attention_dtype   data type for static attention cache (default: auto)"
  echo ""
  echo "Examples:"
  echo "  bash run_quant.sh --dtype=mxfp4 --input_model=/path/to/model --output_model=/path/to/output"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --export_format=*)
      EXPORT_FORMAT="${1#*=}"
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
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown parameter: $1"
      usage
      ;;
  esac
done

if [[ -z "$DTYPE" || -z "$INPUT_MODEL" || -z "$OUTPUT_MODEL" ]]; then
  usage
fi

echo "Starting quantization with parameters:"
echo "  Data Type: $DTYPE"
echo "  Input Model: $INPUT_MODEL"
echo "  Output Model: $OUTPUT_MODEL"

EXTRA_ARGS=()
if [ "$STATIC_KV_DTYPE" != "auto" ]; then
  EXTRA_ARGS+=(--static_kv_dtype "$STATIC_KV_DTYPE")
fi
if [ "$STATIC_ATTENTION_DTYPE" != "auto" ]; then
  EXTRA_ARGS+=(--static_attention_dtype "$STATIC_ATTENTION_DTYPE")
fi

python quantize.py \
  --model_name_or_path "$INPUT_MODEL" \
  --dtype "$DTYPE" \
  --export_format "$EXPORT_FORMAT" \
  --export_path "$OUTPUT_MODEL" \
  "${EXTRA_ARGS[@]}"

echo "Quantization completed successfully!"
echo "Output model saved to: $OUTPUT_MODEL"
