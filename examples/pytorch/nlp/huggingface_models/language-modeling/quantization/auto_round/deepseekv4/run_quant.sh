#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DTYPE=""
INPUT_MODEL=""
OUTPUT_MODEL=""
FORMAT="llm_compressor"
IGNORE_LAYERS="compressor,indexer.weights_proj"

usage() {
  echo "Usage: bash run_quant.sh --dtype=<mxfp4|mxfp4_mixed|mxfp8|w4a16> --input_model=<path_or_name> --output_model=<output_dir>"
  echo "Optional: --format=<auto_round|llm_compressor> --ignore_layers=<comma_separated_patterns>"
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
    --format=*)
      FORMAT="${arg#*=}"
      ;;
    --ignore_layers=*)
      IGNORE_LAYERS="${arg#*=}"
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $arg"
      usage
      ;;
  esac
done

[[ -z "$DTYPE" ]] && echo "Error: --dtype is required" && usage
[[ -z "$INPUT_MODEL" ]] && echo "Error: --input_model is required" && usage
[[ -z "$OUTPUT_MODEL" ]] && echo "Error: --output_model is required" && usage

cd "$SCRIPT_DIR"
python quantize.py \
  --dtype "$DTYPE" \
  --input_model "$INPUT_MODEL" \
  --output_model "$OUTPUT_MODEL" \
  --format "$FORMAT" \
  --ignore_layers "$IGNORE_LAYERS"
