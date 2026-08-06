#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
	cat <<'EOF'
Usage:
	bash run_swe_verified.sh [OPTIONS]

Run mini-SWE-agent on SWE-bench Verified using an already-running vLLM server,
then evaluate the generated predictions with the local SWE-bench harness.

Options:
	--host HOST          vLLM host (default: 127.0.0.1)
	--port PORT          vLLM port (default: 8888)
	--served-name NAME   Served model ID (default: discover from /v1/models)
	--num-tasks N        Run the first N instances
	--slice START:END    Run an explicit dataset slice
	--workers N          Parallel agent workers (default: 2)
	--eval-workers N     Parallel evaluation workers (default: --workers)
	--step-limit N       Maximum model calls per instance (default: 250)
	--pull-timeout N     Docker image pull/start timeout in seconds (default: 600)
	--tag TAG            Run tag (default: UTC timestamp)
	--skip-eval          Generate predictions without local evaluation
	-h, --help           Show this help message

Environment:
	VLLM_API_KEY         API key sent to vLLM (default: EMPTY)
	VLLM_NO_PROXY        Hosts that bypass HTTP proxies (default: vLLM host)
	VLLM_WAIT_TIMEOUT    Readiness timeout in seconds (default: 300)
	VLLM_PID_FILE        vLLM PID file (default: logs/vllm_<PORT>.pid)
	OPENAI_BASE_URL      Override the OpenAI-compatible API base URL
EOF
}

stop_vllm() {
	local pid_file="${VLLM_PID_FILE:-${LOG_DIR}/vllm_${VLLM_PORT}.pid}"
	stop_vllm_from_pid_file "${pid_file}"
}

require_positive_integer() {
	local name="$1"
	local value="$2"
	[[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

WORKERS=2
EVAL_WORKERS=""
STEP_LIMIT=250
PULL_TIMEOUT=600
NUM_TASKS=""
SLICE_ARG=""
SERVED_MODEL_NAME=""
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
	case "$1" in
		--host)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			VLLM_HOST="$2"
			shift 2
			;;
		--port)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			VLLM_PORT="$2"
			shift 2
			;;
		--served-name)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			SERVED_MODEL_NAME="$2"
			shift 2
			;;
		--num-tasks)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			NUM_TASKS="$2"
			shift 2
			;;
		--slice)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			SLICE_ARG="$2"
			shift 2
			;;
		--workers)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			WORKERS="$2"
			shift 2
			;;
		--eval-workers)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			EVAL_WORKERS="$2"
			shift 2
			;;
		--step-limit)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			STEP_LIMIT="$2"
			shift 2
			;;
		--pull-timeout)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			PULL_TIMEOUT="$2"
			shift 2
			;;
		--tag)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			RUN_TAG="$2"
			shift 2
			;;
		--skip-eval)
			SKIP_EVAL=true
			shift
			;;
		-h | --help)
			usage
			exit 0
			;;
		*)
			die "Unknown argument: $1"
			;;
	esac
done

[[ -z "${NUM_TASKS}" || -z "${SLICE_ARG}" ]] || \
	die "--num-tasks and --slice cannot be used together"
require_positive_integer "--workers" "${WORKERS}"
require_positive_integer "--step-limit" "${STEP_LIMIT}"
require_positive_integer "--pull-timeout" "${PULL_TIMEOUT}"
[[ -z "${EVAL_WORKERS}" ]] || require_positive_integer "--eval-workers" "${EVAL_WORKERS}"
[[ -z "${NUM_TASKS}" ]] || require_positive_integer "--num-tasks" "${NUM_TASKS}"
[[ -z "${SLICE_ARG}" || "${SLICE_ARG}" =~ ^[0-9]*:[0-9]*$ ]] || \
	die "--slice must use START:END format: ${SLICE_ARG}"

init_benchmark_paths
init_vllm_endpoint
trap stop_vllm EXIT
RUN_TAG="$(sanitize_run_tag "${RUN_TAG}")"
readonly OUT_DIR="${AGENT_DIR_VERIFIED}/results/swe_verified_${RUN_TAG}"
readonly PREDS_JSON="${OUT_DIR}/preds.json"
readonly PREDS_JSONL="${OUT_DIR}/preds.jsonl"
readonly LOG_FILE="${LOG_DIR}/swe_verified_${RUN_TAG}.log"

[[ -d "${AGENT_DIR_VERIFIED}/.git" ]] || \
	die "mini-SWE-agent is not set up; run: bash setup_swe_verified.sh"
require_command mini-extra
require_command python
require_command docker
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ "${SKIP_EVAL}" == false ]]; then
	python -c 'import swebench.harness.run_evaluation' 2>/dev/null || \
		die "SWE-bench harness is unavailable; run: bash setup_swe_verified.sh"
fi

wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME}")"
EVAL_WORKERS="${EVAL_WORKERS:-${WORKERS}}"
readonly REPORT_FILE="${BENCHMARK_DIR}/${SERVED_MODEL_NAME}.${RUN_TAG}.json"

SLICE_OPTIONS=()
if [[ -n "${SLICE_ARG}" ]]; then
	SLICE_OPTIONS=(--slice "${SLICE_ARG}")
elif [[ -n "${NUM_TASKS}" ]]; then
	SLICE_OPTIONS=(--slice "0:${NUM_TASKS}")
fi

log "Running SWE-bench Verified with model ${SERVED_MODEL_NAME}"
(
	cd "${AGENT_DIR_VERIFIED}"
	MSWEA_COST_TRACKING=ignore_errors mini-extra swebench \
		--model "${SERVED_MODEL_NAME}" \
		--subset verified \
		--split test \
		--workers "${WORKERS}" \
		--output "${OUT_DIR}" \
		--config swebench.yaml \
		--config "model.model_kwargs.base_url=${OPENAI_BASE_URL}" \
		--config "model.model_kwargs.api_key=${VLLM_API_KEY}" \
		--config "agent.step_limit=${STEP_LIMIT}" \
		--config "environment.pull_timeout=${PULL_TIMEOUT}" \
		"${SLICE_OPTIONS[@]}"
)

require_file "${PREDS_JSON}"
python "${SCRIPT_DIR}/lib/benchmark_data.py" verified-jsonl \
	--source "${PREDS_JSON}" --output "${PREDS_JSONL}"

if [[ "${SKIP_EVAL}" == true ]]; then
	log "Skipping local SWE-bench evaluation"
	exit 0
fi

mapfile -t INSTANCE_IDS < <(
	python "${SCRIPT_DIR}/lib/benchmark_data.py" verified-ids --source "${PREDS_JSONL}"
)
[[ ${#INSTANCE_IDS[@]} -gt 0 ]] || die "No predictions found in ${PREDS_JSONL}"

log "Evaluating ${#INSTANCE_IDS[@]} predictions with ${EVAL_WORKERS} workers"
(
	cd "${BENCHMARK_DIR}"
	python -m swebench.harness.run_evaluation \
		--dataset_name princeton-nlp/SWE-bench_Verified \
		--split test \
		--predictions_path "${PREDS_JSONL}" \
		--instance_ids "${INSTANCE_IDS[@]}" \
		--max_workers "${EVAL_WORKERS}" \
		--run_id "${RUN_TAG}"
)

if [[ -f "${REPORT_FILE}" ]]; then
	python "${SCRIPT_DIR}/lib/benchmark_data.py" verified-report --report "${REPORT_FILE}"
else
	warn "Local harness report not found: ${REPORT_FILE}"
fi

log "Predictions: ${PREDS_JSONL}"
log "Report: ${REPORT_FILE}"
log "Log: ${LOG_FILE}"
