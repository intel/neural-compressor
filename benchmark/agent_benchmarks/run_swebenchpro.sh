#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
	cat <<'EOF'
Usage:
	bash run_swebenchpro.sh [OPTIONS]

Run mini-SWE-agent on SWE-bench Pro using an already-running vLLM server, then
evaluate the generated patches with the official local-Docker evaluator.

Options:
	--host HOST          vLLM host (default: 127.0.0.1)
	--port PORT          vLLM port (default: 8888)
	--served-name NAME   Served model ID (default: discover from /v1/models)
	--num-tasks N        Run the first N instances (default: all 731)
	--slice START:END    Run an explicit dataset slice
	--workers N          Parallel agent workers (default: 2)
	--eval-workers N     Parallel evaluation workers (default: --workers)
	--step-limit N       Maximum model calls per instance (default: 250)
	--pull-timeout N     Docker image pull/start timeout in seconds (default: 1800)
	--command-timeout N  In-container command timeout in seconds (default: 600)
	--tag TAG            Run tag (default: UTC timestamp)
	--redo               Re-run existing generation and evaluation results
	--skip-eval          Generate patches without local evaluation
	--block-network      Disable container networking during local evaluation
	--keep-images        Keep SWE-bench Pro Docker images after the run
	-h, --help           Show this help message

Environment:
	AGENT_MAX_TOKENS     Maximum tokens per model response (default: 8192)
	VLLM_API_KEY         API key sent to vLLM (default: EMPTY)
	VLLM_NO_PROXY        Hosts that bypass HTTP proxies (default: vLLM host)
	VLLM_WAIT_TIMEOUT    Readiness timeout in seconds (default: 300)
	OPENAI_BASE_URL      Override the OpenAI-compatible API base URL
EOF
}

remove_swebench_pro_images() {
	[[ "${KEEP_IMAGES}" == false && -f "${IMAGE_LIST}" ]] || return
	mapfile -t images < <(sort -u "${IMAGE_LIST}" | grep -E '^jefzda/sweap-images:' || true)
	[[ ${#images[@]} -gt 0 ]] || return
	log "Removing ${#images[@]} SWE-bench Pro images"
	docker image rm -f "${images[@]}" >/dev/null 2>&1 || \
		warn "Some SWE-bench Pro images could not be removed"
}

cleanup() {
	remove_swebench_pro_images
}

require_positive_integer() {
	local name="$1"
	local value="$2"
	[[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

WORKERS=2
EVAL_WORKERS=""
STEP_LIMIT=250
PULL_TIMEOUT=1800
COMMAND_TIMEOUT=600
NUM_TASKS=""
SLICE_ARG=""
SERVED_MODEL_NAME=""
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
REDO=false
SKIP_EVAL=false
BLOCK_NETWORK=false
KEEP_IMAGES=false

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
		--command-timeout)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			COMMAND_TIMEOUT="$2"
			shift 2
			;;
		--tag)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			RUN_TAG="$2"
			shift 2
			;;
		--redo)
			REDO=true
			shift
			;;
		--skip-eval)
			SKIP_EVAL=true
			shift
			;;
		--block-network)
			BLOCK_NETWORK=true
			shift
			;;
		--keep-images)
			KEEP_IMAGES=true
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
	die "--num-tasks and --slice cannot be combined"
require_positive_integer "--workers" "${WORKERS}"
require_positive_integer "--step-limit" "${STEP_LIMIT}"
require_positive_integer "--pull-timeout" "${PULL_TIMEOUT}"
require_positive_integer "--command-timeout" "${COMMAND_TIMEOUT}"
[[ -z "${EVAL_WORKERS}" ]] || require_positive_integer "--eval-workers" "${EVAL_WORKERS}"
[[ -z "${NUM_TASKS}" ]] || require_positive_integer "--num-tasks" "${NUM_TASKS}"
[[ -z "${SLICE_ARG}" || "${SLICE_ARG}" =~ ^[0-9]*:[0-9]*$ ]] || \
	die "--slice must use START:END format: ${SLICE_ARG}"

init_benchmark_paths
init_vllm_endpoint
RUN_TAG="$(sanitize_run_tag "${RUN_TAG}")"
readonly SWEBENCH_PRO_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os"
readonly OUT_DIR="${AGENT_DIR}/results/swebench_pro_${RUN_TAG}"
readonly PREDS_JSON="${OUT_DIR}/preds.json"
readonly PATCHES_JSON="${OUT_DIR}/patches.json"
readonly INSTANCES_CSV="${OUT_DIR}/instances.csv"
readonly IMAGE_LIST="${OUT_DIR}/images.txt"
readonly EVAL_OUT="${OUT_DIR}/evaluation"
readonly RUNTIME_CONFIG="${OUT_DIR}/swebench_pro.yaml"
readonly LOG_FILE="${LOG_DIR}/swebench_pro_${RUN_TAG}.log"

require_file "${SWEBENCH_PRO_DIR}/swe_bench_pro_eval.py"
require_file "${AGENT_DIR}/pyproject.toml"
require_file "${AGENT_DIR}/src/minisweagent/run/extra/swebench.py"
require_command mini-extra
require_command python
require_command docker
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap cleanup EXIT

python -c 'import datasets, yaml' 2>/dev/null || \
	die "Generation dependencies are unavailable; run: bash setup_swebenchpro.sh"
if [[ "${SKIP_EVAL}" == false ]]; then
	python -c 'import docker, pandas' 2>/dev/null || \
		die "Local evaluation dependencies are unavailable; run: bash setup_swebenchpro.sh"
fi

wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME}")"
EVAL_WORKERS="${EVAL_WORKERS:-${WORKERS}}"
readonly EVAL_PREFIX="$(sanitize_run_tag "${SERVED_MODEL_NAME}_step${STEP_LIMIT}_${RUN_TAG}")"

SLICE_SPEC="${SLICE_ARG}"
[[ -n "${SLICE_SPEC}" || -z "${NUM_TASKS}" ]] || SLICE_SPEC="0:${NUM_TASKS}"
SLICE_OPTIONS=()
[[ -z "${SLICE_SPEC}" ]] || SLICE_OPTIONS=(--slice "${SLICE_SPEC}")
REDO_OPTIONS=()
[[ "${REDO}" == false ]] || REDO_OPTIONS=(--redo-existing)

python "${SCRIPT_DIR}/lib/benchmark_data.py" pro-config \
	--output "${RUNTIME_CONFIG}" --model "${SERVED_MODEL_NAME}" \
	--base-url "${OPENAI_BASE_URL}" --api-key "${VLLM_API_KEY}" \
	--step-limit "${STEP_LIMIT}" --pull-timeout "${PULL_TIMEOUT}" \
	--command-timeout "${COMMAND_TIMEOUT}"

log "Running SWE-bench Pro with model ${SERVED_MODEL_NAME}"
(
	cd "${AGENT_DIR}"
	mini-extra swebench \
		--subset ScaleAI/SWE-bench_Pro \
		--split test \
		--model "${SERVED_MODEL_NAME}" \
		--workers "${WORKERS}" \
		--output "${OUT_DIR}" \
		--config "${RUNTIME_CONFIG}" \
		"${SLICE_OPTIONS[@]}" \
		"${REDO_OPTIONS[@]}"
)

require_file "${PREDS_JSON}"
python "${SCRIPT_DIR}/lib/benchmark_data.py" normalize-pro \
	--source "${PREDS_JSON}" --output "${PATCHES_JSON}" --prefix "${EVAL_PREFIX}"
python "${SCRIPT_DIR}/lib/benchmark_data.py" select-pro \
	--csv "${INSTANCES_CSV}" --images "${IMAGE_LIST}" --slice "${SLICE_SPEC}"

if [[ "${SKIP_EVAL}" == true ]]; then
	log "Skipping local SWE-bench Pro evaluation"
	log "Patches: ${PATCHES_JSON}"
	log "Log: ${LOG_FILE}"
	exit 0
fi

EVAL_OPTIONS=(--use_local_docker)
[[ "${REDO}" == false ]] || EVAL_OPTIONS+=(--redo)
[[ "${BLOCK_NETWORK}" == false ]] || EVAL_OPTIONS+=(--block_network)

log "Evaluating patches with ${EVAL_WORKERS} workers"
(
	cd "${SWEBENCH_PRO_DIR}"
	python swe_bench_pro_eval.py \
		--raw_sample_path "${INSTANCES_CSV}" \
		--patch_path "${PATCHES_JSON}" \
		--output_dir "${EVAL_OUT}" \
		--scripts_dir run_scripts \
		--dockerhub_username jefzda \
		--num_workers "${EVAL_WORKERS}" \
		"${EVAL_OPTIONS[@]}"
)

readonly REPORT_FILE="${EVAL_OUT}/eval_results.json"
require_file "${REPORT_FILE}"
python "${SCRIPT_DIR}/lib/benchmark_data.py" pro-report --report "${REPORT_FILE}"

log "Patches: ${PATCHES_JSON}"
log "Report: ${REPORT_FILE}"
log "Log: ${LOG_FILE}"
