#!/usr/bin/env bash

# Shared helpers for agent benchmark setup and run scripts.

readonly COMMON_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly AGENT_BENCHMARK_ROOT="$(cd -- "${COMMON_LIB_DIR}/.." && pwd)"

log() { printf '[INFO] %s\n' "$*" >&2; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_file() {
    [[ -f "$1" ]] || die "Required file not found: $1"
}

validate_port() {
    local port="$1"
    [[ "${port}" =~ ^[0-9]+$ ]] || die "Port must be an integer: ${port}"
    ((10#${port} >= 1 && 10#${port} <= 65535)) || die "Port must be between 1 and 65535: ${port}"
}

init_benchmark_paths() {
    BENCHMARK_DIR="${BENCHMARK_DIR:-${AGENT_BENCHMARK_ROOT}}"
    AGENT_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os/mini-swe-agent"
    AGENT_DIR_VERIFIED="${BENCHMARK_DIR}/mini-swe-agent"
    MCP_DIR="${BENCHMARK_DIR}/mcp-atlas"
    LOG_DIR="${BENCHMARK_DIR}/logs"
    OUTPUT_DIR="${BENCHMARK_DIR}/outputs"
}

init_vllm_endpoint() {
    VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
    VLLM_PORT="${VLLM_PORT:-8888}"
    VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
    validate_port "${VLLM_PORT}"

    local bypass_hosts="${VLLM_NO_PROXY:-${VLLM_HOST}}"
    local current_no_proxy="${NO_PROXY:-${no_proxy:-}}"
    local bypass_host
    IFS=',' read -ra bypass_host_list <<<"${bypass_hosts}"
    for bypass_host in "${bypass_host_list[@]}"; do
        [[ -n "${bypass_host}" ]] || continue
        if [[ ",${current_no_proxy}," != *",${bypass_host},"* ]]; then
            current_no_proxy="${current_no_proxy:+${current_no_proxy},}${bypass_host}"
        fi
    done
    export NO_PROXY="${current_no_proxy}"
    export no_proxy="${current_no_proxy}"

    VLLM_ORIGIN="${VLLM_ORIGIN:-http://${VLLM_HOST}:${VLLM_PORT}}"
    VLLM_ORIGIN="${VLLM_ORIGIN%/}"
    OPENAI_BASE_URL="${OPENAI_BASE_URL:-${VLLM_ORIGIN}/v1}"
    OPENAI_BASE_URL="${OPENAI_BASE_URL%/}"
}

vllm_curl() {
    local url="$1"
    shift
    local auth_args=()
    if [[ -n "${VLLM_API_KEY:-}" ]]; then
        auth_args=(-H "Authorization: Bearer ${VLLM_API_KEY}")
    fi
    curl --noproxy "${VLLM_NO_PROXY:-${VLLM_HOST}}" -fsS \
        "${auth_args[@]}" "$@" "${url}"
}

wait_for_vllm() {
    local timeout="${1:-300}"
    [[ "${timeout}" =~ ^[1-9][0-9]*$ ]] || die "vLLM wait timeout must be a positive integer: ${timeout}"
    local deadline=$((SECONDS + timeout))
    require_command curl
    init_vllm_endpoint

    log "Waiting up to ${timeout}s for ${VLLM_ORIGIN}/health"
    while ((SECONDS < deadline)); do
        if vllm_curl "${VLLM_ORIGIN}/health" --connect-timeout 2 --max-time 5 >/dev/null 2>&1; then
            log "vLLM is ready"
            return 0
        fi
        sleep 2
    done
    die "vLLM did not become ready within ${timeout}s: ${VLLM_ORIGIN}"
}

discover_vllm_model() {
    local requested_model="${1:-}"
    local python_executable="${PYTHON_EXECUTABLE:-python}"
    local response served_models
    require_command curl
    require_command "${python_executable}"
    init_vllm_endpoint

    response="$(vllm_curl "${OPENAI_BASE_URL}/models")" || \
        die "Unable to query vLLM models from ${OPENAI_BASE_URL}/models"

    served_models="$(${python_executable} -c '
import json, sys
payload = json.load(sys.stdin)
models = [item.get("id", "") for item in payload.get("data", [])]
print("\n".join(model for model in models if model))
' <<<"${response}")" || die "Invalid response from ${OPENAI_BASE_URL}/models"
    [[ -n "${served_models}" ]] || die "vLLM returned no served models"

    if [[ -n "${requested_model}" ]]; then
        grep -Fxq -- "${requested_model}" <<<"${served_models}" || \
            die "Requested model '${requested_model}' is not served. Available: $(tr '\n' ' ' <<<"${served_models}")"
        printf '%s\n' "${requested_model}"
    else
        printf '%s\n' "${served_models%%$'\n'*}"
    fi
}

stop_vllm_from_pid_file() {
    local pid_file="$1" pid command_line
    [[ -f "${pid_file}" ]] || { warn "vLLM PID file not found: ${pid_file}"; return; }

    read -r pid <"${pid_file}" || true
    if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
        warn "Removing stale vLLM PID file: ${pid_file}"
        rm -f -- "${pid_file}"
        return
    fi

    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ "${command_line}" == *vllm* ]] || { warn "PID ${pid} is not vLLM: ${command_line}"; return; }

    log "Stopping vLLM (PID=${pid})"
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for _ in {1..30}; do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
        warn "Force killing vLLM (PID=${pid})"
        kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
    rm -f -- "${pid_file}"
}

sanitize_run_tag() {
    local tag="$1"
    tag="${tag//[^[:alnum:]._-]/_}"
    [[ -n "${tag}" ]] || die "Run tag is empty after sanitization"
    printf '%s\n' "${tag}"
}

plan_batch_slices() {
    local total="$1"
    local num_tasks="$2"
    local slice_spec="$3"
    local batch_size="$4"
    local start=0
    local end="${total}"
    local batch_end

    if [[ -n "${slice_spec}" ]]; then
        IFS=: read -r start end <<<"${slice_spec}"
        start="${start:-0}"
        end="${end:-${total}}"
    elif [[ -n "${num_tasks}" ]]; then
        end="${num_tasks}"
    fi

    start=$((10#${start}))
    end=$((10#${end}))
    batch_size=$((10#${batch_size}))
    ((start < total)) || die "Slice starts beyond the ${total}-instance dataset: ${start}"
    ((end > total)) && end="${total}"
    ((start < end)) || die "Dataset selection is empty: ${start}:${end}"

    while ((start < end)); do
        batch_end=$((start + batch_size))
        ((batch_end > end)) && batch_end="${end}"
        printf '%s:%s\n' "${start}" "${batch_end}"
        start="${batch_end}"
    done
}
