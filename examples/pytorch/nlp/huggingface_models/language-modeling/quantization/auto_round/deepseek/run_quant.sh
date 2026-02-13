#!/bin/bash
set -e

MODEL=""
TARGET=""
OUTPUT_DIR=""
STATIC_KV_DTYPE="None"
STATIC_ATTENTION_DTYPE="None"

usage() {
  echo "Usage: $0 --model MODEL -t [mxfp4|mxfp8] --output_dir DIR"
  echo "  --model      Hugging Face model ID or local path"
  echo "  -t           quantization target (e.g. mxfp8, mxfp4)"
  echo "  -kv datatype for kv cache (auto, fp8)"
  echo "  -attn        Data type for static attention cache (default: None)"
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
    -kv)
      KV_CACHE_DTYPE="$2"
      shift 2
      ;;
    -attn)
      STATIC_ATTENTION_DTYPE="$2"
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
  --output_dir "$OUTPUT_DIR" \
  $( [ "$STATIC_KV_DTYPE" != "None" ] && echo "--static_kv_dtype $STATIC_KV_DTYPE" ) \
  $( [ "$STATIC_ATTENTION_DTYPE" != "None" ] && echo "--static_attention_dtype $STATIC_ATTENTION_DTYPE" )