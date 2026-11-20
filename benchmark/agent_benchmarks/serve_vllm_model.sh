#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash serve_vllm_model.sh --model PATH --served-name NAME --port PORT --gpus 0,1 [OPTIONS] [-- VLLM_ARGS...]
  bash serve_vllm_model.sh --stop --port PORT [--pid-file FILE]

Generic vLLM deployment wrapper for benchmark servers. It starts an
OpenAI-compatible vLLM server from a local model/checkpoint directory and
records a PID file and log. Model-specific venv/patch/overlay setup is
intentionally out of scope: put the desired `vllm` on PATH, pass `--vllm`, or
export the environment before calling this script.

Options:
  --model, --ckpt PATH       Model or checkpoint directory (required)
  --served-name, --name ID   Served model id (default: basename of --model)
  --port PORT                API port (default: 8000)
  --gpus LIST                CUDA_VISIBLE_DEVICES list (default: existing value or 0)
  --tp N                     Tensor parallel size (default: number of --gpus entries)
  --vllm PATH                vllm executable (default: $VLLM_BIN or vllm)
  --work-dir DIR             Directory for logs and PID files (default: current directory)
  --log FILE                 Log file (default: WORK_DIR/logs/vllm_<name>_<UTC>.log)
  --pid-file FILE            PID file (default: WORK_DIR/logs/vllm_<port>.pid)
  --max-model-len, --max-len N
                             Max context length (default: 262144)
  --gpu-mem-util, --gpu-memory-utilization F
                             GPU memory utilization (default: 0.90)
  --max-num-seqs N           vLLM max concurrent sequences
  --max-num-batched-tokens N vLLM max batched tokens
  --dtype DTYPE              vLLM dtype (for example bfloat16)
  --kv-cache-dtype DTYPE     vLLM KV cache dtype (for example fp8)
  --reasoning-parser NAME    vLLM reasoning parser (for example qwen3)
  --tool-call-parser NAME    vLLM tool-call parser (for example qwen3_coder)
  --enable-auto-tool-choice  Enable vLLM automatic tool choice
  --trust-remote-code        Pass --trust-remote-code
  --language-model-only      Skip the vision tower for text benchmarks
  --enforce-eager            Disable CUDA graphs
  --enable-prefix-caching    Enable prefix caching
  --no-enable-prefix-caching Disable prefix caching
  --speculative-config JSON  vLLM speculative decoding config
  --safetensors-load-strategy NAME
                             vLLM safetensors load strategy (for example prefetch)
  --block-size N             vLLM KV cache block size
  --env KEY=VALUE            Export an environment variable for the server (repeatable)
  --min-free-gib N           Fail if a selected GPU has less than N GiB free (default: 0 = only print)
  --wait-timeout N           Poll /health for up to N seconds after launch (default: 0 = no wait)
  --replace                  Stop the existing PID file before starting
  --stop                     Stop the server from --pid-file and exit
  --foreground               exec vllm in the foreground instead of nohup
  --dry-run                  Print the resolved command and exit
  --                         Pass remaining arguments to `vllm serve`
  -h, --help                 Show this help

Examples:
  bash serve_vllm_model.sh \
    --model /models/Qwen3.8-27B --served-name Qwen3.8-27B-mxfp4 \
    --port 8227 --gpus 7 --max-model-len 262144 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice --env NCCL_NVLS_ENABLE=0

  bash serve_vllm_model.sh \
    --model /models/DeepSeek-V4-Flash --served-name dsv4flash0731-mxfp4 \
    --port 8230 --gpus 0,1 --tp 2 --max-model-len 8192 \
    --enforce-eager --no-enable-prefix-caching
EOF
}

is_secret_name() {
  local name="${1,,}"
  case "${name}" in
    *token | *key | *secret | *password | *credential | *auth) return 0 ;;
  esac
  return 1
}

redact_env_pair() {
  local pair="$1"
  local key="${pair%%=*}"
  if is_secret_name "${key}"; then
    printf '%s=REDACTED' "${key}"
  else
    printf '%s' "${pair}"
  fi
}

MODEL=""
SERVED_NAME=""
PORT=8000
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
TP=""
VLLM_BIN="${VLLM_BIN:-vllm}"
WORK_DIR="${PWD}"
LOG_FILE=""
PID_FILE=""
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS=""
MAX_NUM_BATCHED_TOKENS=""
DTYPE=""
KV_CACHE_DTYPE=""
REASONING_PARSER=""
TOOL_CALL_PARSER=""
ENABLE_AUTO_TOOL_CHOICE=0
TRUST_REMOTE_CODE=0
LANGUAGE_MODEL_ONLY=0
ENFORCE_EAGER=0
PREFIX_CACHING=""
SPECULATIVE_CONFIG=""
SAFETENSORS_LOAD_STRATEGY=""
BLOCK_SIZE=""
MIN_FREE_GIB=0
WAIT_TIMEOUT="${VLLM_WAIT_TIMEOUT:-0}"
DRY_RUN=0
FOREGROUND=0
REPLACE=0
STOP=0
ENV_PAIRS=()
EXTRA_VLLM_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model | --ckpt) MODEL="$2"; shift 2 ;;
    --served-name | --name) SERVED_NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --vllm) VLLM_BIN="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --log) LOG_FILE="$2"; shift 2 ;;
    --pid-file) PID_FILE="$2"; shift 2 ;;
    --max-model-len | --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util | --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
    --reasoning-parser) REASONING_PARSER="$2"; shift 2 ;;
    --tool-call-parser) TOOL_CALL_PARSER="$2"; shift 2 ;;
    --enable-auto-tool-choice) ENABLE_AUTO_TOOL_CHOICE=1; shift ;;
    --trust-remote-code) TRUST_REMOTE_CODE=1; shift ;;
    --language-model-only) LANGUAGE_MODEL_ONLY=1; shift ;;
    --enforce-eager) ENFORCE_EAGER=1; shift ;;
    --enable-prefix-caching) PREFIX_CACHING=enable; shift ;;
    --no-enable-prefix-caching) PREFIX_CACHING=disable; shift ;;
    --speculative-config) SPECULATIVE_CONFIG="$2"; shift 2 ;;
    --safetensors-load-strategy) SAFETENSORS_LOAD_STRATEGY="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --env) ENV_PAIRS+=("$2"); shift 2 ;;
    --min-free-gib) MIN_FREE_GIB="$2"; shift 2 ;;
    --wait-timeout) WAIT_TIMEOUT="$2"; shift 2 ;;
    --replace) REPLACE=1; shift ;;
    --stop) STOP=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --)
      shift
      EXTRA_VLLM_ARGS=("$@")
      break
      ;;
    -h | --help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use -- before raw vLLM options)" ;;
  esac
done

validate_port "${PORT}"
[[ "${MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ ]] || die "--max-model-len must be a positive integer"
[[ "${GPU_MEM_UTIL}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--gpu-mem-util must be a number"
[[ -z "${MAX_NUM_SEQS}" || "${MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]] || die "--max-num-seqs must be a positive integer"
[[ -z "${MAX_NUM_BATCHED_TOKENS}" || "${MAX_NUM_BATCHED_TOKENS}" =~ ^[1-9][0-9]*$ ]] || \
  die "--max-num-batched-tokens must be a positive integer"
[[ -z "${BLOCK_SIZE}" || "${BLOCK_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "--block-size must be a positive integer"
[[ "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]] || die "--min-free-gib must be a non-negative integer"
[[ "${WAIT_TIMEOUT}" =~ ^[0-9]+$ ]] || die "--wait-timeout must be a non-negative integer"

mkdir -p -- "${WORK_DIR}"
WORK_DIR="$(cd -- "${WORK_DIR}" && pwd -P)"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p -- "${LOG_DIR}"

if [[ -z "${PID_FILE}" ]]; then
  PID_FILE="${LOG_DIR}/vllm_${PORT}.pid"
elif [[ "${PID_FILE}" != /* ]]; then
  PID_FILE="${WORK_DIR}/${PID_FILE}"
fi
mkdir -p -- "$(dirname -- "${PID_FILE}")"

if [[ "${STOP}" -eq 1 ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[vLLM] Would stop PID file: %s\n' "${PID_FILE}"
    exit 0
  fi
  stop_vllm_from_pid_file "${PID_FILE}"
  exit 0
fi

[[ -n "${MODEL}" ]] || die "--model is required"
[[ -d "${MODEL}" ]] || die "Model directory not found: ${MODEL}"
MODEL="$(cd -- "${MODEL}" && pwd -P)"

if [[ -z "${SERVED_NAME}" ]]; then
  SERVED_NAME="$(basename -- "${MODEL}")"
fi
[[ -n "${SERVED_NAME}" ]] || die "--served-name must not be empty"

if [[ "${VLLM_BIN}" == */* ]]; then
  [[ -x "${VLLM_BIN}" ]] || die "vllm executable not found: ${VLLM_BIN}"
else
  require_command "${VLLM_BIN}"
fi

IFS=',' read -ra GPU_ARR <<<"${GPUS}"
[[ ${#GPU_ARR[@]} -gt 0 ]] || die "--gpus must not be empty"
for gpu in "${GPU_ARR[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "Invalid GPU id: ${gpu}"
done
TP="${TP:-${#GPU_ARR[@]}}"
[[ "${TP}" =~ ^[1-9][0-9]*$ ]] || die "--tp must be a positive integer"
export CUDA_VISIBLE_DEVICES="${GPUS}"

for pair in "${ENV_PAIRS[@]}"; do
  [[ "${pair}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || die "Invalid --env value (expected KEY=VALUE): ${pair}"
  export "${pair}"
done

if [[ -z "${LOG_FILE}" ]]; then
  SAFE_NAME="${SERVED_NAME//[^[:alnum:]._-]/_}"
  LOG_FILE="${LOG_DIR}/vllm_${SAFE_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"
fi
[[ "${LOG_FILE}" == /* ]] || LOG_FILE="${WORK_DIR}/${LOG_FILE}"
mkdir -p -- "$(dirname -- "${LOG_FILE}")"

ARGS=(
  serve "${MODEL}"
  --served-model-name "${SERVED_NAME}"
  --port "${PORT}"
  --tensor-parallel-size "${TP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
)
[[ -z "${MAX_NUM_SEQS}" ]] || ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[[ -z "${MAX_NUM_BATCHED_TOKENS}" ]] || ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[[ -z "${DTYPE}" ]] || ARGS+=(--dtype "${DTYPE}")
[[ -z "${KV_CACHE_DTYPE}" ]] || ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
[[ -z "${REASONING_PARSER}" ]] || ARGS+=(--reasoning-parser "${REASONING_PARSER}")
[[ -z "${TOOL_CALL_PARSER}" ]] || ARGS+=(--tool-call-parser "${TOOL_CALL_PARSER}")
[[ "${ENABLE_AUTO_TOOL_CHOICE}" -eq 0 ]] || ARGS+=(--enable-auto-tool-choice)
[[ "${TRUST_REMOTE_CODE}" -eq 0 ]] || ARGS+=(--trust-remote-code)
[[ "${LANGUAGE_MODEL_ONLY}" -eq 0 ]] || ARGS+=(--language-model-only)
[[ "${ENFORCE_EAGER}" -eq 0 ]] || ARGS+=(--enforce-eager)
[[ "${PREFIX_CACHING}" != enable ]] || ARGS+=(--enable-prefix-caching)
[[ "${PREFIX_CACHING}" != disable ]] || ARGS+=(--no-enable-prefix-caching)
[[ -z "${SPECULATIVE_CONFIG}" ]] || ARGS+=(--speculative-config "${SPECULATIVE_CONFIG}")
[[ -z "${SAFETENSORS_LOAD_STRATEGY}" ]] || ARGS+=(--safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}")
[[ -z "${BLOCK_SIZE}" ]] || ARGS+=(--block-size "${BLOCK_SIZE}")
ARGS+=("${EXTRA_VLLM_ARGS[@]}")

DISPLAY_ENV=()
for pair in "${ENV_PAIRS[@]}"; do
  DISPLAY_ENV+=("$(redact_env_pair "${pair}")")
done

DISPLAY_ARGS=()
skip_next=false
for arg in "${ARGS[@]}"; do
  if [[ "${skip_next}" == true ]]; then
    DISPLAY_ARGS+=("REDACTED")
    skip_next=false
    continue
  fi
  if [[ "${arg}" == --*=* ]]; then
    name="${arg%%=*}"
    if is_secret_name "${name#--}"; then
      DISPLAY_ARGS+=("${name}=REDACTED")
    else
      DISPLAY_ARGS+=("${arg}")
    fi
    continue
  fi
  if [[ "${arg}" == --* ]] && is_secret_name "${arg#--}"; then
    DISPLAY_ARGS+=("${arg}")
    skip_next=true
    continue
  fi
  DISPLAY_ARGS+=("${arg}")
done

printf '[vLLM] CUDA_VISIBLE_DEVICES=%s\n' "${GPUS}"
if [[ ${#DISPLAY_ENV[@]} -gt 0 ]]; then
  printf '[vLLM] env:'
  printf ' %s' "${DISPLAY_ENV[@]}"
  printf '\n'
fi
printf '[vLLM] Command:'
printf ' %q' "${VLLM_BIN}" "${DISPLAY_ARGS[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  read -r OLD_PID <"${PID_FILE}" || true
  if [[ "${OLD_PID}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    if [[ "${REPLACE}" -eq 1 ]]; then
      stop_vllm_from_pid_file "${PID_FILE}"
    else
      die "vLLM is already running (PID=${OLD_PID}, PID file=${PID_FILE}); use --replace or --stop"
    fi
  else
    rm -f -- "${PID_FILE}"
  fi
fi

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | awk -v suffix=":${port}" '$4 == suffix || $4 ~ (suffix "$") { found = 1 } END { exit !found }'
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

if port_in_use "${PORT}"; then
  if [[ "${REPLACE}" -eq 1 ]]; then
    die "Port ${PORT} is still in use after stopping the PID file; refusing to start"
  fi
  die "Port ${PORT} is already in use"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  for gpu in "${GPU_ARR[@]}"; do
    if read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | awk -F', *' '{print $1, $2}'); then
      [[ "${used}" =~ ^[0-9]+$ && "${total}" =~ ^[0-9]+$ ]] || continue
      free_gib=$(((total - used) / 1024))
      printf '[vLLM] GPU %s free ~%s GiB\n' "${gpu}" "${free_gib}"
      if ((MIN_FREE_GIB > 0 && free_gib < MIN_FREE_GIB)); then
        die "GPU ${gpu} has only ~${free_gib} GiB free (requires ${MIN_FREE_GIB} GiB)"
      fi
    fi
  done
fi

if [[ "${FOREGROUND}" -eq 1 ]]; then
  log "Running vLLM in the foreground"
  exec "${VLLM_BIN}" "${ARGS[@]}"
fi

require_command setsid
nohup setsid env VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}" \
  "${VLLM_BIN}" "${ARGS[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
VLLM_PID=$!
printf '%s\n' "${VLLM_PID}" >"${PID_FILE}"
printf '%s\n' "${VLLM_PID}" >"${LOG_FILE}.pid"

printf '[vLLM] Started in background (PID=%s)\n' "${VLLM_PID}"
printf '[vLLM] Log: %s\n' "${LOG_FILE}"
printf '[vLLM] PID file: %s\n' "${PID_FILE}"
printf '[vLLM] Ready: curl -s http://127.0.0.1:%s/v1/models\n' "${PORT}"
printf '[vLLM] Stop:  bash %s --stop --port %s --pid-file %s\n' "$0" "${PORT}" "${PID_FILE}"

if ((WAIT_TIMEOUT > 0)); then
  require_command curl
  deadline=$((SECONDS + WAIT_TIMEOUT))
  while ((SECONDS < deadline)); do
    if curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 \
      "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      log "vLLM is ready"
      exit 0
    fi
    sleep 2
  done
  die "vLLM did not become ready within ${WAIT_TIMEOUT}s; see ${LOG_FILE}"
fi
