#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ORIGINAL_ARGS=("$@")
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
	cat <<'EOF'
Usage:
	bash run_swe_verified.sh [OPTIONS]

Run mini-SWE-agent on SWE-bench Verified using an already-running vLLM server,
then evaluate the generated predictions with the local SWE-bench harness.

Generation runs as a single continuous process across the whole selection so
the vLLM server always has work queued. Finished instances are evaluated in
chunks as they become available, overlapping the CPU/Docker-bound evaluation
with the next instances being generated on the GPU instead of alternating
between the two phases.

Options:
	--host HOST          vLLM host (default: 127.0.0.1)
	--port PORT          vLLM port (default: 8888)
	--served-name NAME   Served model ID (default: discover from /v1/models)
	--num-tasks N        Run the first N instances
	--slice START:END    Run an explicit dataset slice
	--workers N          Parallel agent workers (default: 16)
	--eval-workers N     Parallel evaluation workers (default: 8)
	--step-limit N       Maximum model calls per instance (default: 250)
	--pull-timeout N     Docker image pull/start timeout in seconds (default: 600)
	--eval-chunk-size N  Finished instances gathered before dispatching an evaluation chunk (default: 24)
	--poll-interval N    Seconds between checks for newly finished instances (default: 60)
	--health-interval N  Seconds between vLLM health checks (default: 30)
	--health-failures N  Consecutive failed health checks before stopping (default: 3)
	--tag TAG            Run tag (default: UTC timestamp)
	--retry-errors       Retry error cases; reuse valid patches and regenerate invalid submissions
	--retry-empty-patches
	                     Regenerate and evaluate cases previously reported with empty patches
	--retry-attempts N   Retry errors and empty patches for at most N rounds (default: 1)
	--skip-eval          Generate predictions without local evaluation
	--keep-images        Keep benchmark Docker images after each evaluation chunk
	-h, --help           Show this help message

Environment:
	VLLM_API_KEY         API key sent to vLLM (default: EMPTY)
	VLLM_NO_PROXY        Hosts that bypass HTTP proxies (default: vLLM host)
	VLLM_WAIT_TIMEOUT    Readiness timeout in seconds (default: 300)
	OPENAI_BASE_URL      Override the OpenAI-compatible API base URL
EOF
}

verified_image_name() {
	local instance_id="$1"
	local id="${instance_id//__/_1776_}"
	id="${id,,}"
	printf 'docker.io/swebench/sweb.eval.x86_64.%s:latest' "${id}"
}

remove_chunk_images() {
	local ids_file="$1"
	[[ "${KEEP_IMAGES}" == false && -s "${ids_file}" ]] || return
	local images=() instance_id
	while IFS= read -r instance_id; do
		[[ -n "${instance_id}" ]] || continue
		images+=("$(verified_image_name "${instance_id}")")
	done <"${ids_file}"
	[[ ${#images[@]} -gt 0 ]] || return
	log "Removing ${#images[@]} SWE-bench Verified images"
	docker image rm -f "${images[@]}" >/dev/null 2>&1 || \
		warn "Some SWE-bench Verified images could not be removed"
}

readonly RUNNER_PID=$$
GEN_PID=""
EVAL_PID=""
MONITOR_PID=""

cleanup() {
	local pid
	for pid in "${MONITOR_PID}" "${EVAL_PID}" "${GEN_PID}"; do
		[[ -n "${pid}" ]] || continue
		kill -0 "${pid}" 2>/dev/null || continue
		kill -TERM "${pid}" 2>/dev/null || true
	done
}

termination_requested() {
	warn "SWE-bench Verified run was interrupted"
	exit 1
}

require_positive_integer() {
	local name="$1"
	local value="$2"
	[[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

WORKERS=16
EVAL_WORKERS=8
STEP_LIMIT=250
PULL_TIMEOUT=600
EVAL_CHUNK_SIZE=24
POLL_INTERVAL=60
HEALTH_INTERVAL=30
HEALTH_FAILURES=3
NUM_TASKS=""
SLICE_ARG=""
SERVED_MODEL_NAME=""
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
TAG_SPECIFIED=false
SKIP_EVAL=false
KEEP_IMAGES=false
RETRY_ERRORS=false
RETRY_EMPTY_PATCHES=false
RETRY_ATTEMPTS=1
RETRY_ATTEMPTS_SPECIFIED=false

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
		--eval-chunk-size)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			EVAL_CHUNK_SIZE="$2"
			shift 2
			;;
		--poll-interval)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			POLL_INTERVAL="$2"
			shift 2
			;;
		--health-interval)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			HEALTH_INTERVAL="$2"
			shift 2
			;;
		--health-failures)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			HEALTH_FAILURES="$2"
			shift 2
			;;
		--tag)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			RUN_TAG="$2"
			TAG_SPECIFIED=true
			shift 2
			;;
		--retry-errors)
			RETRY_ERRORS=true
			shift
			;;
		--retry-empty-patches)
			RETRY_EMPTY_PATCHES=true
			shift
			;;
		--retry-attempts)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			RETRY_ATTEMPTS="$2"
			RETRY_ATTEMPTS_SPECIFIED=true
			shift 2
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

if [[ "${RETRY_ATTEMPTS_SPECIFIED}" == true ]]; then
	RETRY_ERRORS=true
	RETRY_EMPTY_PATCHES=true
fi

[[ -z "${NUM_TASKS}" || -z "${SLICE_ARG}" ]] || \
	die "--num-tasks and --slice cannot be used together"
if [[ "${SKIP_EVAL}" == true && ("${RETRY_ERRORS}" == true || "${RETRY_EMPTY_PATCHES}" == true) ]]; then
	die "Retry options cannot be combined with --skip-eval"
fi
if [[ "${TAG_SPECIFIED}" == false && ("${RETRY_ERRORS}" == true || "${RETRY_EMPTY_PATCHES}" == true) ]]; then
	die "Retry options require the --tag of an existing run"
fi
require_positive_integer "--workers" "${WORKERS}"
require_positive_integer "--step-limit" "${STEP_LIMIT}"
require_positive_integer "--pull-timeout" "${PULL_TIMEOUT}"
require_positive_integer "--eval-chunk-size" "${EVAL_CHUNK_SIZE}"
require_positive_integer "--poll-interval" "${POLL_INTERVAL}"
require_positive_integer "--health-interval" "${HEALTH_INTERVAL}"
require_positive_integer "--health-failures" "${HEALTH_FAILURES}"
require_positive_integer "--retry-attempts" "${RETRY_ATTEMPTS}"
[[ -z "${EVAL_WORKERS}" ]] || require_positive_integer "--eval-workers" "${EVAL_WORKERS}"
[[ -z "${NUM_TASKS}" ]] || require_positive_integer "--num-tasks" "${NUM_TASKS}"
[[ -z "${SLICE_ARG}" || "${SLICE_ARG}" =~ ^[0-9]*:[0-9]*$ ]] || \
	die "--slice must use START:END format: ${SLICE_ARG}"

init_benchmark_paths
init_vllm_endpoint
trap cleanup EXIT
trap termination_requested INT TERM
RUN_TAG="$(sanitize_run_tag "${RUN_TAG}")"
readonly OUT_DIR="${AGENT_DIR_VERIFIED}/results/swe_verified_${RUN_TAG}"
readonly GEN_DIR="${OUT_DIR}/generation"
readonly EVAL_ROOT="${OUT_DIR}/eval_chunks"
readonly CLAIMED_FILE="${OUT_DIR}/claimed_ids.txt"
readonly EVAL_REPORT_LIST="${OUT_DIR}/eval_report_list.txt"
readonly PREDS_JSON="${OUT_DIR}/preds.json"
readonly PREDS_JSONL="${OUT_DIR}/preds.jsonl"
readonly GEN_LOG="${OUT_DIR}/generation.log"
readonly LOG_FILE="${LOG_DIR}/swe_verified_${RUN_TAG}.log"
readonly REPORT_FILE="${OUT_DIR}/report.json"

[[ -d "${AGENT_DIR_VERIFIED}/.git" ]] || \
	die "mini-SWE-agent is not set up; run: bash setup_swe_verified.sh"
require_command mini-extra
require_command python
require_command docker

if [[ "${RETRY_ERRORS}" == true || "${RETRY_EMPTY_PATCHES}" == true ]]; then
	require_file "${REPORT_FILE}"
	require_file "${GEN_DIR}/preds.json"
	require_file "${CLAIMED_FILE}"
	require_file "${EVAL_REPORT_LIST}"
else
	mkdir -p "${GEN_DIR}" "${EVAL_ROOT}"
	: >>"${CLAIMED_FILE}"
	: >>"${EVAL_REPORT_LIST}"
fi

retry_remaining_count() {
	local retry_options=()
	[[ "${RETRY_ERRORS}" == false ]] || retry_options+=(--retry-errors)
	[[ "${RETRY_EMPTY_PATCHES}" == false ]] || retry_options+=(--retry-empty-patches)
	python "${SCRIPT_DIR}/lib/benchmark_data.py" verified-retry-count \
		--report "${REPORT_FILE}" "${retry_options[@]}"
}

if [[ "${SWE_VERIFIED_RETRY_LOOP_CHILD:-0}" != 1 &&
	("${RETRY_ERRORS}" == true || "${RETRY_EMPTY_PATCHES}" == true) &&
	"${RETRY_ATTEMPTS}" -gt 1 ]]; then
	for ((retry_attempt = 1; retry_attempt <= RETRY_ATTEMPTS; retry_attempt++)); do
		remaining="$(retry_remaining_count)"
		if [[ "${remaining}" -eq 0 ]]; then
			log "Retry categories are empty; stopping before attempt ${retry_attempt}/${RETRY_ATTEMPTS}"
			exit 0
		fi
		log "Starting retry attempt ${retry_attempt}/${RETRY_ATTEMPTS} for ${remaining} instances"
		SWE_VERIFIED_RETRY_LOOP_CHILD=1 bash "$0" "${ORIGINAL_ARGS[@]}"
	done
	exit 0
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ "${RETRY_ERRORS}" == true || "${RETRY_EMPTY_PATCHES}" == true ]]; then
	retry_dir="${OUT_DIR}/retries/retry_$(date -u +%Y%m%dT%H%M%SZ)_$$"
	retry_options=()
	[[ "${RETRY_ERRORS}" == false ]] || retry_options+=(--retry-errors)
	[[ "${RETRY_EMPTY_PATCHES}" == false ]] || retry_options+=(--retry-empty-patches)
	python "${SCRIPT_DIR}/lib/benchmark_data.py" prepare-verified-retry \
		--report "${REPORT_FILE}" --preds "${GEN_DIR}/preds.json" \
		--claimed "${CLAIMED_FILE}" --eval-report-list "${EVAL_REPORT_LIST}" \
		--backup-dir "${retry_dir}" "${retry_options[@]}"
fi

if [[ "${SKIP_EVAL}" == false ]]; then
	python -c 'import swebench.harness.run_evaluation' 2>/dev/null || \
		die "SWE-bench harness is unavailable; run: bash setup_swe_verified.sh"
fi

wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME}")"
EVAL_WORKERS="${EVAL_WORKERS:-${WORKERS}}"
full_slice_plan="$(plan_batch_slices 500 "${NUM_TASKS}" "${SLICE_ARG}" 500)"
mapfile -t FULL_SLICE_LINES <<<"${full_slice_plan}"
[[ ${#FULL_SLICE_LINES[@]} -eq 1 ]] || die "Unexpected slice plan for SWE-bench Verified selection"
readonly FULL_SLICE="${FULL_SLICE_LINES[0]}"

log "Generating SWE-bench Verified selection ${FULL_SLICE} with model ${SERVED_MODEL_NAME}"
(
	cd "${AGENT_DIR_VERIFIED}"
	exec env MSWEA_COST_TRACKING=ignore_errors mini-extra swebench \
		--model "${SERVED_MODEL_NAME}" \
		--subset verified \
		--split test \
		--workers "${WORKERS}" \
		--output "${GEN_DIR}" \
		--config swebench.yaml \
		--config "model.model_kwargs.base_url=${OPENAI_BASE_URL}" \
		--config "model.model_kwargs.api_key=${VLLM_API_KEY}" \
		--config "agent.step_limit=${STEP_LIMIT}" \
		--config "environment.pull_timeout=${PULL_TIMEOUT}" \
		--slice "${FULL_SLICE}"
) >"${GEN_LOG}" 2>&1 &
GEN_PID=$!
log "Generation console progress: tail -f ${GEN_LOG}"
log "mini-SWE-agent execution log: tail -f ${GEN_DIR}/minisweagent.log"

monitor_vllm() {
	local failures=0
	while true; do
		sleep "${HEALTH_INTERVAL}"
		if vllm_curl "${VLLM_ORIGIN}/health" --connect-timeout 2 --max-time 5 >/dev/null 2>&1; then
			failures=0
			continue
		fi

		failures=$((failures + 1))
		warn "vLLM health check failed (${failures}/${HEALTH_FAILURES}): ${VLLM_ORIGIN}/health"
		if ((failures >= HEALTH_FAILURES)); then
			warn "vLLM is unavailable; stopping SWE-bench Verified"
			kill -TERM "${GEN_PID}" 2>/dev/null || true
			kill -TERM "${RUNNER_PID}" 2>/dev/null || true
			return
		fi
	done
}

stop_vllm_monitor() {
	[[ -n "${MONITOR_PID}" ]] || return
	kill -TERM "${MONITOR_PID}" 2>/dev/null || true
	wait "${MONITOR_PID}" 2>/dev/null || true
	MONITOR_PID=""
}

monitor_vllm &
MONITOR_PID=$!

# Evaluates one chunk of newly finished instance IDs (given as $2..) and marks
# them claimed. Runs while generation continues for the remaining instances.
process_pending_chunk() {
	local chunk_dir="$1"
	shift
	local chunk_ids_file="${chunk_dir}/ids.txt"
	mkdir -p "${chunk_dir}"
	printf '%s\n' "$@" >"${chunk_ids_file}"

	if [[ "${SKIP_EVAL}" == false ]]; then
		local chunk_preds_jsonl="${chunk_dir}/preds.jsonl"
		local chunk_report="${chunk_dir}/report.json"
		local run_id
		run_id="${RUN_TAG}_$(basename "${chunk_dir}")_$$"
		local harness_report="${BENCHMARK_DIR}/${SERVED_MODEL_NAME}.${run_id}.json"

		python "${SCRIPT_DIR}/lib/benchmark_data.py" verified-jsonl \
			--source "${GEN_DIR}/preds.json" --output "${chunk_preds_jsonl}" \
			--ids "${chunk_ids_file}"

		log "Evaluating $# SWE-bench Verified instances with ${EVAL_WORKERS} workers"
		rm -f -- "${harness_report}"
		(
			cd "${BENCHMARK_DIR}"
			exec python -m swebench.harness.run_evaluation \
				--dataset_name princeton-nlp/SWE-bench_Verified \
				--split test \
				--predictions_path "${chunk_preds_jsonl}" \
				--instance_ids "$@" \
				--max_workers "${EVAL_WORKERS}" \
				--run_id "${run_id}"
		) &
		EVAL_PID=$!
		wait "${EVAL_PID}"
		EVAL_PID=""
		if [[ -f "${harness_report}" ]]; then
			mv -- "${harness_report}" "${chunk_report}"
			printf '%s\n' "${chunk_report}" >>"${EVAL_REPORT_LIST}"
			python "${SCRIPT_DIR}/lib/benchmark_data.py" merge-eval-reports \
				--batch-list "${EVAL_REPORT_LIST}" --report "${REPORT_FILE}"
		else
			warn "Local harness report not found: ${harness_report}"
		fi
	fi

	remove_chunk_images "${chunk_ids_file}"
	cat "${chunk_ids_file}" >>"${CLAIMED_FILE}"
}

gen_running() {
	kill -0 "${GEN_PID}" 2>/dev/null
}

pending_instance_ids() {
	python "${SCRIPT_DIR}/lib/benchmark_data.py" pending-verified \
		--preds "${GEN_DIR}/preds.json" --claimed "${CLAIMED_FILE}"
}

chunk_counter=0
for existing_chunk in "${EVAL_ROOT}"/chunk_*; do
	[[ -d "${existing_chunk}" ]] || continue
	existing_index="${existing_chunk##*/chunk_}"
	[[ "${existing_index}" =~ ^[0-9]+$ ]] || continue
	if ((10#${existing_index} >= chunk_counter)); then
		chunk_counter=$((10#${existing_index} + 1))
	fi
done
while true; do
	mapfile -t PENDING_IDS < <(pending_instance_ids)
	if gen_running; then
		if [[ ${#PENDING_IDS[@]} -lt ${EVAL_CHUNK_SIZE} ]]; then
			sleep "${POLL_INTERVAL}"
			continue
		fi
	else
		stop_vllm_monitor
		if [[ ${#PENDING_IDS[@]} -eq 0 ]]; then
			break
		fi
	fi

	chunk_size=${EVAL_CHUNK_SIZE}
	[[ ${#PENDING_IDS[@]} -lt ${chunk_size} ]] && chunk_size=${#PENDING_IDS[@]}
	printf -v chunk_name 'chunk_%06d' "${chunk_counter}"
	chunk_counter=$((chunk_counter + 1))
	process_pending_chunk "${EVAL_ROOT}/${chunk_name}" "${PENDING_IDS[@]:0:${chunk_size}}"
done

GEN_EXIT=0
wait "${GEN_PID}" || GEN_EXIT=$?
GEN_PID=""
stop_vllm_monitor
require_file "${GEN_DIR}/preds.json"
cp -- "${GEN_DIR}/preds.json" "${PREDS_JSON}"
python "${SCRIPT_DIR}/lib/benchmark_data.py" verified-jsonl \
	--source "${PREDS_JSON}" --output "${PREDS_JSONL}"

if [[ "${SKIP_EVAL}" == false ]]; then
	python "${SCRIPT_DIR}/lib/benchmark_data.py" merge-eval-reports \
		--batch-list "${EVAL_REPORT_LIST}" --report "${REPORT_FILE}"
fi

log "Predictions: ${PREDS_JSONL}"
if [[ -f "${REPORT_FILE}" ]]; then
	python "${SCRIPT_DIR}/lib/benchmark_data.py" verified-report --report "${REPORT_FILE}"
	log "Report: ${REPORT_FILE}"
fi
log "Log: ${LOG_FILE}"

[[ "${GEN_EXIT}" -eq 0 ]] || die "mini-SWE-agent generation failed (exit ${GEN_EXIT}); see ${GEN_LOG}"
