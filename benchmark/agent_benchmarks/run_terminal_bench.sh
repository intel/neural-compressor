#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
	cat <<'EOF'
Usage:
	bash run_terminal_bench.sh --benchmark NAME [OPTIONS]

Run Terminal-Bench through Harbor against an existing OpenAI-compatible vLLM
endpoint. NAME can be terminal-bench-2.0, terminal-bench-2.1, or all.

Options:
	--benchmark NAME      Benchmark to run (required)
	--host HOST           vLLM host (default: 127.0.0.1)
	--port PORT           vLLM port (default: 8888)
	--served-name NAME    Served model ID (default: discover from /v1/models)
	--env-prefix PATH     Conda prefix containing Harbor (default: current environment)
	--output-dir PATH     Output directory (default: ./outputs/terminal-bench)
	--attempts N          Trials per task (default: 5)
	--workers N           Parallel Harbor trials (default: 1)
	--retry-attempts N    Retry a failed benchmark process up to N times (default: 1)
	--mini                Run one task
	--dry-run             Print commands without executing them
	-h, --help            Show this help message

Environment:
	VLLM_API_KEY          API key sent to vLLM (default: EMPTY)
EOF
}

require_positive_integer() {
	local name="$1" value="$2"
	[[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

run_command() {
	if [[ "${DRY_RUN}" == true ]]; then
		printf '[DRY-RUN]'
		printf ' %q' "$@"
		printf '\n'
	else
		"$@"
	fi
}

run_with_retries() {
	local label="$1"
	shift
	local attempt status
	for ((attempt = 1; attempt <= RETRY_ATTEMPTS; attempt++)); do
		log "Running ${label} (attempt ${attempt}/${RETRY_ATTEMPTS})"
		if run_command "$@"; then
			return 0
		else
			status=$?
		fi
		warn "${label} failed with status ${status}"
	done
	return "${status}"
}

run_benchmark() {
	local benchmark="$1" dataset job_name
	case "${benchmark}" in
		terminal-bench-2.0) dataset="terminal-bench@2.0" ;;
		terminal-bench-2.1) dataset="terminal-bench/terminal-bench-2-1@6" ;;
		*) die "Unknown benchmark: ${benchmark}" ;;
	esac
	job_name="${benchmark}-${RUN_TAG}"

	local command=(harbor run --yes)
	if [[ -n "${ENV_PREFIX}" ]]; then
		command=(conda run --no-capture-output -p "${ENV_PREFIX}" "${command[@]}")
	fi
	command+=(
		--dataset "${dataset}"
		--agent terminus-2
		--model "openai/${SERVED_MODEL_NAME}"
		--agent-kwarg "api_base=${OPENAI_BASE_URL}"
		--agent-kwarg "temperature=1.0"
		--agent-kwarg 'model_info={"max_input_tokens":262144,"max_output_tokens":80000}'
		--agent-kwarg 'llm_call_kwargs={"top_p":0.95,"max_tokens":80000,"extra_body":{"top_k":20}}'
		--agent-kwarg 'interleaved_thinking=true'
		--jobs-dir "${OUTPUT_DIR}"
		--job-name "${job_name}"
		--n-attempts "${ATTEMPTS}"
		--n-concurrent "${WORKERS}"
		--timeout-multiplier 1.0
		--agent-timeout-multiplier 12.0
		--cpus limit
		--memory limit
		--override-cpus 32
		--override-memory-mb 49152
	)
	[[ "${MINI}" == false ]] || command+=(--n-tasks 1)
	run_with_retries "${benchmark}" "${command[@]}"
}

BENCHMARK=""
ENV_PREFIX="${TERMINAL_BENCH_ENV_PREFIX:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/terminal-bench}"
ATTEMPTS=5
WORKERS=1
RETRY_ATTEMPTS=1
MINI=false
DRY_RUN=false
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--benchmark) [[ $# -ge 2 ]] || die "$1 requires a value"; BENCHMARK="$2"; shift 2 ;;
		--host) [[ $# -ge 2 ]] || die "$1 requires a value"; VLLM_HOST="$2"; shift 2 ;;
		--port) [[ $# -ge 2 ]] || die "$1 requires a value"; VLLM_PORT="$2"; shift 2 ;;
		--served-name) [[ $# -ge 2 ]] || die "$1 requires a value"; SERVED_MODEL_NAME="$2"; shift 2 ;;
		--env-prefix) [[ $# -ge 2 ]] || die "$1 requires a value"; ENV_PREFIX="$2"; shift 2 ;;
		--output-dir) [[ $# -ge 2 ]] || die "$1 requires a value"; OUTPUT_DIR="$2"; shift 2 ;;
		--attempts) [[ $# -ge 2 ]] || die "$1 requires a value"; ATTEMPTS="$2"; shift 2 ;;
		--workers) [[ $# -ge 2 ]] || die "$1 requires a value"; WORKERS="$2"; shift 2 ;;
		--retry-attempts) [[ $# -ge 2 ]] || die "$1 requires a value"; RETRY_ATTEMPTS="$2"; shift 2 ;;
		--mini) MINI=true; shift ;;
		--dry-run) DRY_RUN=true; shift ;;
		-h | --help) usage; exit 0 ;;
		*) die "Unknown argument: $1" ;;
	esac
done

[[ -n "${BENCHMARK}" ]] || die "--benchmark is required"
case "${BENCHMARK}" in
	terminal-bench-2.0 | terminal-bench-2.1 | all) ;;
	*) die "Unsupported --benchmark '${BENCHMARK}'" ;;
esac
require_positive_integer "--attempts" "${ATTEMPTS}"
require_positive_integer "--workers" "${WORKERS}"
require_positive_integer "--retry-attempts" "${RETRY_ATTEMPTS}"

init_vllm_endpoint
if [[ "${DRY_RUN}" == false ]]; then
	wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
	SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME:-}")"
	mkdir -p -- "${OUTPUT_DIR}"
	if [[ -n "${ENV_PREFIX}" ]]; then
		require_command conda
		require_file "${ENV_PREFIX}/bin/harbor"
	else
		require_command harbor
	fi
else
	SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B}"
fi

export OPENAI_API_KEY="${VLLM_API_KEY}"
export PYTHONNOUSERSITE=1

benchmarks=(terminal-bench-2.0 terminal-bench-2.1)
[[ "${BENCHMARK}" == all ]] || benchmarks=("${BENCHMARK}")
for benchmark in "${benchmarks[@]}"; do
	run_benchmark "${benchmark}"
done
