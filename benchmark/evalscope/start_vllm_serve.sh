#!/bin/bash

set -euo pipefail

# Usage:
#   bash start_vllm_serve.sh --model MODEL_PATH [--port PORT] [--tp N] \
#       [--kv-cache-dtype DTYPE] [--block-size N]
#
# Builds and launches (via exec) a vLLM OpenAI-compatible server.
# This script is normally invoked by run_evalscope.sh, but can also be run standalone.

PORT=8009
MODEL=/workspace/models/deepseek-ai/DeepSeek-V4-Flash
SCHEME="mxfp4"
KV_CACHE_DTYPE="fp8"
BLOCK_SIZE=256
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=102400
MAX_NUM_SEQS=1024
MAX_NUM_BATCHED_TOKENS=32768
SAFETENSORS_FAST_GPU="1"
TRUST_REMOTE_CODE="true"
NO_ENABLE_FLASHINFER_AUTOTUNE="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"; shift 2 ;;
    --model)
      MODEL="$2"; shift 2 ;;
    --scheme)
      SCHEME="$2"; shift 2 ;;
    --tp)
      TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --kv-cache-dtype)
      KV_CACHE_DTYPE="$2"; shift 2 ;;
    --block-size)
      BLOCK_SIZE="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

EXTRA_ARGS=()
# Only for base DeepSeek-V4-Flash/Pro model names without quantized suffixes.
if [[ "${MODEL}" == *"DeepSeek-V4-"* ]] && [[ "${SCHEME}" == "bf16" ]]; then
  EXTRA_ARGS+=(--enable-expert-parallel)
  EXTRA_ARGS+=(--moe-backend deep_gemm_mega_moe)
fi

VLLM_CMD=(
  vllm serve "${MODEL}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --block-size "${BLOCK_SIZE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --attention_config.use_fp4_indexer_cache=True
  --port "${PORT}"
  --served-model-name "local_test"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --enable-chunked-prefill
)

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  VLLM_CMD+=(--trust-remote-code)
fi
if [[ "${NO_ENABLE_FLASHINFER_AUTOTUNE}" == "true" ]]; then
  VLLM_CMD+=(--no-enable-flashinfer-autotune)
fi
VLLM_CMD+=("${EXTRA_ARGS[@]}")

export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU}"

# Detect NVIDIA GPU compute capability (SM). Non-Blackwell (SM != 10.0) needs extra env vars.
GPU_SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
echo "Detected GPU compute capability (SM): ${GPU_SM:-unknown}"
if [[ "${GPU_SM}" != "10.0" ]]; then
    echo "SM is not 10.0, setting extra environment variables for non-Blackwell GPUs."
    # For https://github.com/yiliu30/vllm-qdq-plugin.git CT format eval
    export VLLM_QDQ=1
    export VLLM_MXFP4_USE_MARLIN=1
fi
exec "${VLLM_CMD[@]}"
