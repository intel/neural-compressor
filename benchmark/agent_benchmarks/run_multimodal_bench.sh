#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
	cat <<'EOF'
Usage:
	bash run_multimodal_bench.sh --benchmark NAME [OPTIONS]

Run multimodal benchmarks through lmms-eval against an existing
OpenAI-compatible vLLM endpoint. NAME can be mmmu, mmmu-pro, simplevqa,
omnidocbench-1.5, or all.

Options:
	--benchmark NAME      Benchmark to run (required)
	--host HOST           vLLM host (default: 127.0.0.1)
	--port PORT           vLLM port (default: 8888)
	--served-name NAME    Served model ID (default: discover from /v1/models)
	--env-prefix PATH     Conda prefix containing lmms-eval (default: current environment)
	--lmms-eval-root PATH lmms-eval checkout (default: ./lmms-eval)
	--output-dir PATH     Output directory (default: ./outputs/multimodal-bench)
	--run-id ID           Store this run under OUTPUT_DIR/ID
	--resume              Skip benchmarks completed by the same --run-id
	--response-cache PATH Reuse deterministic lmms-eval responses (default: RUN_DIR/.response-cache)
	--workers N           Parallel API requests (default: 1)
	--retry-attempts N    Retry a failed benchmark process up to N times (default: 1)
	--mini                Run the fixed 90-sample Mini dataset
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
	local benchmark="$1" task timeout thinking gen_kwargs completion_marker run_signature
	case "${benchmark}" in
		mmmu) task=mmmu_val; timeout=600; thinking=true ;;
		mmmu-pro) task=mmmu_pro_vision; timeout=600; thinking=true ;;
		simplevqa) task=simplevqa; timeout=180; thinking=false ;;
		omnidocbench-1.5) task=omnidocbench; timeout=7200; thinking=true ;;
		*) die "Unknown benchmark: ${benchmark}" ;;
	esac
	if [[ "${MINI}" == true ]]; then
		task="${task}_mini"
	fi
	if [[ "${thinking}" == true ]]; then
		gen_kwargs="max_new_tokens=32768,temperature=1.0,top_p=0.95,top_k=20,presence_penalty=1.5"
	else
		gen_kwargs=""
	fi
	completion_marker="${OUTPUT_DIR}/.completed/${benchmark}"
	run_signature="${SERVED_MODEL_NAME}|${OPENAI_BASE_URL}|${task}|mini=${MINI}|${gen_kwargs}"
	if [[ "${RESUME}" == true && -f "${completion_marker}" ]]; then
		if [[ "$(<"${completion_marker}")" == "${run_signature}" ]]; then
			log "Skipping completed benchmark ${benchmark}"
			return 0
		fi
		die "Completed benchmark ${benchmark} does not match the current model or run options: ${completion_marker}"
	fi

	local model_args="model=${SERVED_MODEL_NAME},base_url=${OPENAI_BASE_URL},api_key=${VLLM_API_KEY},num_concurrent=${WORKERS},timeout=${timeout},httpx_trust_env=false,enable_thinking_kwarg=${thinking}"
	local command=(python -m lmms_eval)
	if [[ -n "${ENV_PREFIX}" ]]; then
		command=(conda run --no-capture-output -p "${ENV_PREFIX}" "${command[@]}")
	fi
	command+=(
		--model openai
		--model_args "${model_args}"
		--tasks "${task}"
		--batch_size 1
		--log_samples
		--output_path "${OUTPUT_DIR}"
	)
	[[ -z "${RESPONSE_CACHE}" ]] || command+=(--use_cache "${RESPONSE_CACHE}")
	[[ -z "${gen_kwargs}" ]] || command+=(--gen_kwargs "${gen_kwargs}")

	if [[ "${DRY_RUN}" == true ]]; then
		run_with_retries "${benchmark}" "${command[@]}"
	else
		(cd "${LMMS_EVAL_ROOT}" && run_with_retries "${benchmark}" "${command[@]}")
		mkdir -p -- "$(dirname -- "${completion_marker}")"
		printf '%s\n' "${run_signature}" >"${completion_marker}"
	fi
}

BENCHMARK=""
ENV_PREFIX="${MULTIMODAL_BENCH_ENV_PREFIX:-}"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-${SCRIPT_DIR}/lmms-eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/multimodal-bench}"
RUN_ID=""
RESUME=false
RESPONSE_CACHE=""
WORKERS=1
RETRY_ATTEMPTS=1
MINI=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
	case "$1" in
		--benchmark) [[ $# -ge 2 ]] || die "$1 requires a value"; BENCHMARK="$2"; shift 2 ;;
		--host) [[ $# -ge 2 ]] || die "$1 requires a value"; VLLM_HOST="$2"; shift 2 ;;
		--port) [[ $# -ge 2 ]] || die "$1 requires a value"; VLLM_PORT="$2"; shift 2 ;;
		--served-name) [[ $# -ge 2 ]] || die "$1 requires a value"; SERVED_MODEL_NAME="$2"; shift 2 ;;
		--env-prefix) [[ $# -ge 2 ]] || die "$1 requires a value"; ENV_PREFIX="$2"; shift 2 ;;
		--lmms-eval-root) [[ $# -ge 2 ]] || die "$1 requires a value"; LMMS_EVAL_ROOT="$2"; shift 2 ;;
		--output-dir) [[ $# -ge 2 ]] || die "$1 requires a value"; OUTPUT_DIR="$2"; shift 2 ;;
		--run-id) [[ $# -ge 2 ]] || die "$1 requires a value"; RUN_ID="$2"; shift 2 ;;
		--resume) RESUME=true; shift ;;
		--response-cache) [[ $# -ge 2 ]] || die "$1 requires a value"; RESPONSE_CACHE="$2"; shift 2 ;;
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
	mmmu | mmmu-pro | simplevqa | omnidocbench-1.5 | all) ;;
	*) die "Unsupported --benchmark '${BENCHMARK}'" ;;
esac
require_positive_integer "--workers" "${WORKERS}"
require_positive_integer "--retry-attempts" "${RETRY_ATTEMPTS}"
if [[ -n "${RUN_ID}" ]]; then
	[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-id contains unsupported characters: ${RUN_ID}"
	OUTPUT_DIR="${OUTPUT_DIR}/${RUN_ID}"
	RESPONSE_CACHE="${RESPONSE_CACHE:-${OUTPUT_DIR}/.response-cache}"
elif [[ "${RESUME}" == true ]]; then
	die "--resume requires --run-id"
fi

init_vllm_endpoint
if [[ "${DRY_RUN}" == false ]]; then
	wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
	SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME:-}")"
	mkdir -p -- "${OUTPUT_DIR}"
	[[ -z "${ENV_PREFIX}" ]] || require_command conda
	require_file "${LMMS_EVAL_ROOT}/lmms_eval/__main__.py"
else
	SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B}"
fi

export PYTHONNOUSERSITE=1

benchmarks=(mmmu mmmu-pro simplevqa omnidocbench-1.5)
[[ "${BENCHMARK}" == all ]] || benchmarks=("${BENCHMARK}")
for benchmark in "${benchmarks[@]}"; do
	run_benchmark "${benchmark}"
done
