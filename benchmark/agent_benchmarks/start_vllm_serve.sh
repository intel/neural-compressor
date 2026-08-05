#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash start_vllm_serve.sh MODEL [VLLM_OPTIONS...]

Start a foreground vLLM server. Extra arguments are passed directly to
`vllm serve` and can override the defaults below.

Generic defaults:
  --port 8888
  --served-model-name gpt-3.5-turbo
  --max-model-len 262144 --trust-remote-code --enable-auto-tool-choice
  --tool-call-parser hermes (unless supplied through VLLM_OPTIONS)

DeepSeek-V4 models additionally use:
  max model length 1048576, tensor parallel 2, FP8 KV cache, block size 256,
  deepseek_v4 tool parser, FP4 indexer cache, and disabled FlashInfer
  autotuning. Names containing "mxfp4" use the CUTLASS MoE backend; other
  DeepSeek-V4 names use deep_gemm_mega_moe and expert parallelism.

Examples:
  CUDA_VISIBLE_DEVICES=0 bash start_vllm_serve.sh /path/to/Qwen3-8B \
      --tensor-parallel-size 1 --tool-call-parser hermes \
      --enable-auto-tool-choice
  CUDA_VISIBLE_DEVICES=0,1 bash start_vllm_serve.sh /path/to/DeepSeek-V4-Flash
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

MODEL="$1"
shift

has_option() {
  local NAME="$1"
  local OPTION
  shift
  for OPTION in "$@"; do
    if [[ "${OPTION}" == "${NAME}" || "${OPTION}" == "${NAME}="* ]]; then
      return 0
    fi
  done
  return 1
}

ARGS=(
  serve "${MODEL}"
  --served-model-name gpt-3.5-turbo
  --trust-remote-code
  --enable-auto-tool-choice
)

if ! has_option --port "$@"; then
  ARGS+=(--port 8888)
fi

MODEL_LOWER="${MODEL,,}"

if [[ "${MODEL_LOWER}" == *deepseek-v4* || "${MODEL_LOWER}" == *deepseek_v4* ]]; then
  export SAFETENSORS_FAST_GPU=1
  ARGS+=(
    --kv-cache-dtype fp8
    --block-size 256
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}"
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}"
    --tool-call-parser deepseek_v4
    --attention_config.use_fp4_indexer_cache=True
    --no-enable-flashinfer-autotune
  )

  if ! has_option --max-model-len "$@"; then
    ARGS+=(--max-model-len 1048576)
  fi

  if [[ "${MODEL_LOWER}" == *mxfp4* ]]; then
    ARGS+=(--moe-backend cutlass)
  else
    ARGS+=(--moe-backend deep_gemm_mega_moe --enable-expert-parallel)
  fi
else
  if ! has_option --max-model-len "$@"; then
    ARGS+=(--max-model-len "${MAX_MODEL_LEN:-262144}")
  fi

  if ! has_option --tool-call-parser "$@"; then
    ARGS+=(--tool-call-parser hermes)
  fi
fi

ARGS+=("$@")

printf '[vLLM] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<not set>}"
printf '[vLLM] Command:'
printf ' %q' vllm "${ARGS[@]}"
printf '\n'

VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}" \
  exec vllm "${ARGS[@]}"
