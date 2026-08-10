#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash start_vllm_serve.sh MODEL [VLLM_OPTIONS...]
  bash start_vllm_serve.sh --stop [--port PORT]

Start vLLM in the background. Extra options are passed to `vllm serve`.
Use --stop to stop the server for a port (default: 8888).

Defaults:
  port=8888, served-name=gpt-3.5-turbo, max-model-len=262144
  tensor parallel size=CUDA_VISIBLE_DEVICES count, tool parser=hermes
  Qwen3.6: tool=qwen3_coder, reasoning=qwen3
  DeepSeek-V4: model-specific FP8/MoE defaults

Logs and PID files are written under ./logs. Override with VLLM_LOG_DIR,
VLLM_LOG_FILE, or VLLM_PID_FILE. See README.md for model-specific details.
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

if [[ "$1" == "--stop" ]]; then
  shift
  STOP_PORT=8888
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        STOP_PORT="$2"
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        die "Unknown stop argument: $1"
        ;;
    esac
  done

  validate_port "${STOP_PORT}"
  init_benchmark_paths
  LOG_DIR="${VLLM_LOG_DIR:-${LOG_DIR}}"
  PID_FILE="${VLLM_PID_FILE:-${LOG_DIR}/vllm_${STOP_PORT}.pid}"
  stop_vllm_from_pid_file "${PID_FILE}"
  exit 0
fi

MODEL="$1"
shift

has_option() {
  local name="$1" option
  shift
  for option in "$@"; do
    [[ "${option}" == "${name}" || "${option}" == "${name}="* ]] && return 0
  done
  return 1
}

add_default_option() {
  local name="$1" value="$2"
  shift 2
  has_option "${name}" "$@" || ARGS+=("${name}" "${value}")
}

require_command vllm
require_command setsid
init_benchmark_paths

AVAILABLE_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | awk -F',' '{print NF}')

ARGS=(
  serve "${MODEL}"
  --served-model-name gpt-3.5-turbo
  --trust-remote-code
  --enable-auto-tool-choice
)

add_default_option --port 8888 "$@"
add_default_option --tensor-parallel-size "${AVAILABLE_GPUS}" "$@"

MODEL_LOWER="${MODEL,,}"
MAX_MODEL_LEN_DEFAULT="${MAX_MODEL_LEN:-262144}"

if [[ "${MODEL_LOWER}" == *qwen3.6* ]]; then
  add_default_option --tool-call-parser qwen3_coder "$@"
  add_default_option --reasoning-parser qwen3 "$@"
elif is_deepseek_v4_model "${MODEL}"; then
  export SAFETENSORS_FAST_GPU=1
  MAX_MODEL_LEN_DEFAULT=1048576
  ARGS+=(
    --kv-cache-dtype fp8
    --block-size 256
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}"
    --attention_config.use_fp4_indexer_cache=True
    --no-enable-flashinfer-autotune
  )
  add_default_option --tool-call-parser deepseek_v4 "$@"

  if [[ "${MODEL_LOWER}" == *mxfp4* ]]; then
    ARGS+=(--moe-backend cutlass)
  else
    ARGS+=(--moe-backend deep_gemm_mega_moe --enable-expert-parallel)
  fi
else
  add_default_option --tool-call-parser hermes "$@"
fi

add_default_option --max-model-len "${MAX_MODEL_LEN_DEFAULT}" "$@"
ARGS+=("$@")

printf '[vLLM] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<not set>}"
printf '[vLLM] Command:'
printf ' %q' vllm "${ARGS[@]}"
printf '\n'

LOG_DIR="${VLLM_LOG_DIR:-${LOG_DIR}}"
mkdir -p -- "${LOG_DIR}"

MODEL_NAME="${MODEL%/}"
MODEL_NAME="${MODEL_NAME##*/}"
MODEL_NAME="${MODEL_NAME//[^[:alnum:]._-]/_}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${VLLM_LOG_FILE:-${LOG_DIR}/vllm_${MODEL_NAME}_${TIMESTAMP}.log}"
LOG_PID_FILE="${LOG_FILE}.pid"
mkdir -p -- "$(dirname -- "${LOG_FILE}")"

EFFECTIVE_PORT=8888
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[i]}" == --port && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    EFFECTIVE_PORT="${ARGS[i + 1]}"
  elif [[ "${ARGS[i]}" == --port=* ]]; then
    EFFECTIVE_PORT="${ARGS[i]#--port=}"
  fi
done
PID_FILE="${VLLM_PID_FILE:-${LOG_DIR}/vllm_${EFFECTIVE_PORT}.pid}"
mkdir -p -- "$(dirname -- "${PID_FILE}")"

if [[ -f "${PID_FILE}" ]]; then
  read -r OLD_PID <"${PID_FILE}" || true
  if [[ "${OLD_PID}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    die "vLLM is already running for port ${EFFECTIVE_PORT} (PID=${OLD_PID})"
  fi
  rm -f -- "${PID_FILE}"
fi

nohup setsid env VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}" \
  vllm "${ARGS[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
VLLM_PID=$!
printf '%s\n' "${VLLM_PID}" >"${PID_FILE}"
printf '%s\n' "${VLLM_PID}" >"${LOG_PID_FILE}"

printf '[vLLM] Started in background (PID=%s)\n' "${VLLM_PID}"
printf '[vLLM] Log: %s\n' "${LOG_FILE}"
printf '[vLLM] PID file: %s\n' "${PID_FILE}"
