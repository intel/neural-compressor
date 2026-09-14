#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DTYPE=""
INPUT_MODEL=""
OUTPUT_MODEL=""
FORMAT="llm_compressor"
STATIC_KV_DTYPE=""
STATIC_ATTENTION_DTYPE=""
# requires transformers==4.57.6 for fp8 kv static quant

usage() {
  echo "Usage: bash run_quant.sh --dtype=<mxfp8_moe_fp4|mxfp8|mxfp4> --input_model=<path_or_name> --output_model=<output_dir>"
  echo "Optional: --format=<auto_round|llm_compressor> --static_kv_dtype=<fp8> --static_attention_dtype=<fp8>"
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
    --static_kv_dtype=*)
      STATIC_KV_DTYPE="${arg#*=}"
      ;;
    --static_attention_dtype=*)
      STATIC_ATTENTION_DTYPE="${arg#*=}"
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
QUANTIZE_ARGS=(
  --dtype "$DTYPE"
  --input_model "$INPUT_MODEL"
  --output_model "$OUTPUT_MODEL"
  --format "$FORMAT"
)

[[ -n "$STATIC_KV_DTYPE" ]] && QUANTIZE_ARGS+=(--static_kv_dtype "$STATIC_KV_DTYPE")
[[ -n "$STATIC_ATTENTION_DTYPE" ]] && QUANTIZE_ARGS+=(--static_attention_dtype "$STATIC_ATTENTION_DTYPE")

python quantize.py "${QUANTIZE_ARGS[@]}"
