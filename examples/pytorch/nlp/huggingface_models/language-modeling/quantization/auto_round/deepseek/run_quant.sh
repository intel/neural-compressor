#!/bin/bash
set -e

MODEL=""
TARGET=""
OUTPUT_DIR=""
KV_CACHE_DTYPE="auto"

usage() {
  echo "Usage: $0 --model MODEL -t [mxfp4|mxfp8] --output_dir DIR"
  echo "  --model      Hugging Face model ID or local path"
  echo "  -t           quantization target (e.g. mxfp8, mxfp4)"
  echo "  --kv_cache_dtype datatype for kv cache (auto, fp8)"
  echo "  --output_dir output directory for quantized model"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    -t)
      TARGET="$2"
      shift 2
      ;;
    --kv_cache_dtype)
      KV_CACHE_DTYPE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

[ -z "$MODEL" ] && echo "Error: --model is required" && usage
[ -z "$TARGET" ] && echo "Error: -t is required" && usage
[ -z "$OUTPUT_DIR" ] && echo "Error: --output_dir is required" && usage

AR_LOG_LEVEL=TRACE \
python quantize.py \
  --model "$MODEL" \
  -t "$TARGET" \
  --use_autoround_format \
  --kv_cache_dtype "$KV_CACHE_DTYPE" \
  --output_dir "$OUTPUT_DIR"