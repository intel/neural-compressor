#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DTYPE=""
INPUT_MODEL=""
OUTPUT_MODEL=""
EXPORT_FORMAT="llm_compressor"
IGNORE_LAYERS="compressor,indexer.weights_proj"

usage() {
  echo "Usage: bash run_quant.sh --dtype=<dtype> --input_model=<input_model> --output_model=<output_model>"
  echo "  --dtype            quantization data type: mxfp4, mxfp4_mixed, mxfp8, w4a16"
  echo "  --input_model      Hugging Face model ID or local path"
  echo "  --output_model     output directory for the quantized model"
  echo "  --export_format    export format (default: llm_compressor)"
  echo "  --ignore_layers    comma-separated layer name patterns to skip"
  echo ""
  echo "Examples:"
  echo "  bash run_quant.sh --dtype=mxfp4 --input_model=/path/to/model --output_model=/path/to/output"
  exit 1
}

for arg in "$@"; do
  case $arg in
    --dtype=*)
      DTYPE="${arg#*=}"
      ;;
    --input_model=*)
      INPUT_MODEL="${arg#*=}"
      ;;
    --output_model=*)
      OUTPUT_MODEL="${arg#*=}"
      ;;
    --export_format=*)
      EXPORT_FORMAT="${arg#*=}"
      ;;
    --ignore_layers=*)
      IGNORE_LAYERS="${arg#*=}"
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown parameter: $arg"
      usage
      ;;
  esac
done

[[ -z "$DTYPE" ]] && echo "Error: --dtype is required" && usage
[[ -z "$INPUT_MODEL" ]] && echo "Error: --input_model is required" && usage
[[ -z "$OUTPUT_MODEL" ]] && echo "Error: --output_model is required" && usage

echo "Starting quantization with parameters:"
echo "  Data Type: $DTYPE"
echo "  Input Model: $INPUT_MODEL"
echo "  Output Model: $OUTPUT_MODEL"

cd "$SCRIPT_DIR"
python quantize.py \
  --dtype "$DTYPE" \
  --model_name_or_path "$INPUT_MODEL" \
  --export_path "$OUTPUT_MODEL" \
  --export_format "$EXPORT_FORMAT" \
  --ignore_layers "$IGNORE_LAYERS"

echo "Quantization completed successfully!"
echo "Output model saved to: $OUTPUT_MODEL"
