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
	--workers N          Parallel agent workers (default: 4)
	--eval-workers N     Parallel evaluation workers (default: 8)
	--step-limit N       Maximum model calls per instance (default: 250)
	--pull-timeout N     Docker image pull/start timeout in seconds (default: 600)
	--batch-size N       Instances per generation/evaluation batch (default: 25)
	--tag TAG            Run tag (default: UTC timestamp)
	--redo               Re-run existing generation and evaluation results
	--skip-eval          Generate predictions without local evaluation
	--keep-images        Keep benchmark Docker images after each batch
	-h, --help           Show this help message

Environment:
	VLLM_API_KEY         API key sent to vLLM (default: EMPTY)
	VLLM_NO_PROXY        Hosts that bypass HTTP proxies (default: vLLM host)
	VLLM_WAIT_TIMEOUT    Readiness timeout in seconds (default: 300)
	OPENAI_BASE_URL      Override the OpenAI-compatible API base URL
EOF
}

remove_verified_images() {
	local image_list="${CURRENT_IMAGE_LIST}"
	CURRENT_IMAGE_LIST=""
	[[ "${KEEP_IMAGES}" == false && -n "${image_list}" && -f "${image_list}" ]] || return
	mapfile -t images < <(sort -u "${image_list}" | grep -E '^docker.io/swebench/sweb\.eval\.' || true)
	[[ ${#images[@]} -gt 0 ]] || return
	log "Removing ${#images[@]} SWE-bench Verified images"
	docker image rm -f "${images[@]}" >/dev/null 2>&1 || \
		warn "Some SWE-bench Verified images could not be removed"
}

cleanup() {
	remove_verified_images
}

require_positive_integer() {
	local name="$1"
	local value="$2"
	[[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

WORKERS=4
EVAL_WORKERS=8
STEP_LIMIT=250
PULL_TIMEOUT=600
BATCH_SIZE=25
NUM_TASKS=""
SLICE_ARG=""
SERVED_MODEL_NAME=""
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
REDO=false
SKIP_EVAL=false
KEEP_IMAGES=false
CURRENT_IMAGE_LIST=""

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
		--batch-size)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			BATCH_SIZE="$2"
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
	die "--num-tasks and --slice cannot be used together"
require_positive_integer "--workers" "${WORKERS}"
require_positive_integer "--step-limit" "${STEP_LIMIT}"
require_positive_integer "--pull-timeout" "${PULL_TIMEOUT}"
require_positive_integer "--batch-size" "${BATCH_SIZE}"
[[ -z "${EVAL_WORKERS}" ]] || require_positive_integer "--eval-workers" "${EVAL_WORKERS}"
[[ -z "${NUM_TASKS}" ]] || require_positive_integer "--num-tasks" "${NUM_TASKS}"
[[ -z "${SLICE_ARG}" || "${SLICE_ARG}" =~ ^[0-9]*:[0-9]*$ ]] || \
	die "--slice must use START:END format: ${SLICE_ARG}"

init_benchmark_paths
init_vllm_endpoint
trap cleanup EXIT
RUN_TAG="$(sanitize_run_tag "${RUN_TAG}")"
readonly OUT_DIR="${AGENT_DIR_VERIFIED}/results/swe_verified_${RUN_TAG}"
readonly BATCH_ROOT="${OUT_DIR}/batches"
readonly BATCH_LIST="${OUT_DIR}/batch_list.txt"
readonly PREDS_JSON="${OUT_DIR}/preds.json"
readonly PREDS_JSONL="${OUT_DIR}/preds.jsonl"
readonly LOG_FILE="${LOG_DIR}/swe_verified_${RUN_TAG}.log"

[[ -d "${AGENT_DIR_VERIFIED}/.git" ]] || \
	die "mini-SWE-agent is not set up; run: bash setup_swe_verified.sh"
require_command mini-extra
require_command python
require_command docker
mkdir -p "${BATCH_ROOT}" "${LOG_DIR}"
: >"${BATCH_LIST}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ "${SKIP_EVAL}" == false ]]; then
	python -c 'import swebench.harness.run_evaluation' 2>/dev/null || \
		die "SWE-bench harness is unavailable; run: bash setup_swe_verified.sh"
fi

wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME}")"
EVAL_WORKERS="${EVAL_WORKERS:-${WORKERS}}"
readonly REPORT_FILE="${BENCHMARK_DIR}/${SERVED_MODEL_NAME}.${RUN_TAG}.json"

REDO_OPTIONS=()
[[ "${REDO}" == false ]] || REDO_OPTIONS=(--redo-existing)
batch_plan="$(plan_batch_slices 500 "${NUM_TASKS}" "${SLICE_ARG}" "${BATCH_SIZE}")"
mapfile -t BATCH_SLICES <<<"${batch_plan}"
[[ ${#BATCH_SLICES[@]} -gt 0 ]] || die "No SWE-bench Verified batches were selected"

for batch_slice in "${BATCH_SLICES[@]}"; do
	IFS=: read -r batch_start batch_end <<<"${batch_slice}"
	printf -v batch_name 'batch_%06d_%06d' "${batch_start}" "${batch_end}"
	batch_dir="${BATCH_ROOT}/${batch_name}"
	batch_preds_json="${batch_dir}/preds.json"
	batch_preds_jsonl="${batch_dir}/preds.jsonl"
	batch_report="${batch_dir}/report.json"
	batch_run_id="${RUN_TAG}_${batch_name}"
	[[ "${REDO}" == false ]] || batch_run_id+="_redo_$(date -u +%Y%m%dT%H%M%SZ)_$$"
	harness_report="${BENCHMARK_DIR}/${SERVED_MODEL_NAME}.${batch_run_id}.json"
	CURRENT_IMAGE_LIST="${batch_dir}/images.txt"
	mkdir -p "${batch_dir}"
	printf '%s\n' "${batch_dir}" >>"${BATCH_LIST}"

	python "${SCRIPT_DIR}/lib/benchmark_data.py" select-verified \
		--images "${CURRENT_IMAGE_LIST}" --slice "${batch_slice}"

	log "Generating SWE-bench Verified batch ${batch_slice} with model ${SERVED_MODEL_NAME}"
	(
		cd "${AGENT_DIR_VERIFIED}"
		MSWEA_COST_TRACKING=ignore_errors mini-extra swebench \
			--model "${SERVED_MODEL_NAME}" \
			--subset verified \
			--split test \
			--workers "${WORKERS}" \
			--output "${batch_dir}" \
			--config swebench.yaml \
			--config "model.model_kwargs.base_url=${OPENAI_BASE_URL}" \
			--config "model.model_kwargs.api_key=${VLLM_API_KEY}" \
			--config "agent.step_limit=${STEP_LIMIT}" \
			--config "environment.pull_timeout=${PULL_TIMEOUT}" \
			--slice "${batch_slice}" \
			"${REDO_OPTIONS[@]}"
	)

	require_file "${batch_preds_json}"
	python "${SCRIPT_DIR}/lib/benchmark_data.py" verified-jsonl \
		--source "${batch_preds_json}" --output "${batch_preds_jsonl}"

	if [[ "${SKIP_EVAL}" == false ]]; then
		rm -f -- "${batch_report}" "${harness_report}"
		mapfile -t INSTANCE_IDS < <(
			python "${SCRIPT_DIR}/lib/benchmark_data.py" verified-ids --source "${batch_preds_jsonl}"
		)
		[[ ${#INSTANCE_IDS[@]} -gt 0 ]] || die "No predictions found in ${batch_preds_jsonl}"

		log "Evaluating SWE-bench Verified batch ${batch_slice} with ${EVAL_WORKERS} workers"
		(
			cd "${BENCHMARK_DIR}"
			python -m swebench.harness.run_evaluation \
				--dataset_name princeton-nlp/SWE-bench_Verified \
				--split test \
				--predictions_path "${batch_preds_jsonl}" \
				--instance_ids "${INSTANCE_IDS[@]}" \
				--max_workers "${EVAL_WORKERS}" \
				--run_id "${batch_run_id}"
		)
		if [[ -f "${harness_report}" ]]; then
			mv -- "${harness_report}" "${batch_report}"
		else
			warn "Local harness report not found: ${harness_report}"
		fi
	else
		rm -f -- "${batch_report}"
		log "Skipping local evaluation for SWE-bench Verified batch ${batch_slice}"
	fi

	remove_verified_images
done

python "${SCRIPT_DIR}/lib/benchmark_data.py" merge-verified \
	--batch-list "${BATCH_LIST}" --predictions "${PREDS_JSON}" \
	--jsonl "${PREDS_JSONL}" --report "${REPORT_FILE}"

log "Predictions: ${PREDS_JSONL}"
if [[ -f "${REPORT_FILE}" ]]; then
	python "${SCRIPT_DIR}/lib/benchmark_data.py" verified-report --report "${REPORT_FILE}"
	log "Report: ${REPORT_FILE}"
fi
log "Log: ${LOG_FILE}"
