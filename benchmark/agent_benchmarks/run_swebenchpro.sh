#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ORIGINAL_ARGS=("$@")
printf -v RUNNER_COMMAND '%q ' bash "${SCRIPT_DIR}/run_swebenchpro.sh" "${ORIGINAL_ARGS[@]}"
readonly RUNNER_COMMAND="${RUNNER_COMMAND% }"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
	cat <<'EOF'
Usage:
	bash run_swebenchpro.sh [OPTIONS]

Run mini-SWE-agent on SWE-bench Pro using an already-running vLLM server, then
evaluate generated patches with the official local-Docker evaluator.

Generation runs continuously across the whole selection. Finished instances
are evaluated in chunks while generation continues, overlapping CPU/Docker
evaluation with GPU inference.

Options:
	--host HOST          vLLM host (default: 127.0.0.1)
	--port PORT          vLLM port (default: 8888)
	--served-name NAME   Served model ID (default: discover from /v1/models)
	--num-tasks N        Run the first N instances (default: all instances in the dataset)
	--slice START:END    Run an explicit dataset slice
	--dataset-repo REPO  Hugging Face dataset to evaluate (default: ScaleAI/SWE-bench_Pro)
	--dataset-split S    Dataset split (default: test)
	--workers N          Parallel agent workers (default: 8)
	--eval-workers N     Parallel evaluation workers (default: 4)
	--step-limit N       Maximum model calls per instance (default: 250)
	--pull-timeout N     Docker image pull/start timeout in seconds (default: 1800)
	--command-timeout N  In-container command timeout in seconds (default: 600)
	--eval-chunk-size N  Finished instances gathered before dispatching an evaluation chunk (default: 12)
	--poll-interval N    Seconds between checks for newly finished instances (default: 60)
	--health-interval N  Seconds between vLLM health checks (default: 30)
	--health-failures N  Consecutive failed health checks before stopping (default: 3)
	--tag TAG            Run tag (default: UTC timestamp)
	--runs N             Repeat the whole benchmark N times sequentially, then report
	                     mean/std with per-run details (default: 7; 1 disables repeats)
	--skip-eval          Generate patches without local evaluation
	--block-network      Disable container networking during local evaluation
	--keep-images        Keep SWE-bench Pro Docker images after each evaluation chunk
	-h, --help           Show this help message

Environment:
	AGENT_MAX_TOKENS     Maximum tokens per model response (default: 8192)
	VLLM_API_KEY         API key sent to vLLM (default: EMPTY)
	VLLM_NO_PROXY        Hosts that bypass HTTP proxies (default: vLLM host)
	VLLM_WAIT_TIMEOUT    Readiness timeout in seconds (default: 300)
	OPENAI_BASE_URL      Override the OpenAI-compatible API base URL
EOF
}

remove_chunk_images() {
	local image_list="$1"
	[[ "${KEEP_IMAGES}" == false && -s "${image_list}" ]] || return 0
	mapfile -t images < <(sort -u "${image_list}" | grep -E '^jefzda/sweap-images:' || true)
	[[ ${#images[@]} -gt 0 ]] || return 0
	log "Removing ${#images[@]} SWE-bench Pro images"
	docker image rm -f "${images[@]}" >/dev/null 2>&1 || \
		warn "Some SWE-bench Pro images could not be removed"
}

readonly RUNNER_PID=$$
GEN_PID=""
EVAL_PID=""
MONITOR_PID=""
SUITE_CHILD_PID=""
STATE_FILE=""
REPORT_FILE=""
SUITE_STATE_FILE=""
INTERRUPTED=false

finish_state_file() {
	local state_file="$1" status="$2" exit_code="$3" report="${4:-}"
	[[ -n "${state_file}" && -f "${state_file}" ]] || return 0
	local report_args=()
	[[ -z "${report}" ]] || report_args=(--report "${report}")
	python "${SCRIPT_DIR}/lib/run_state.py" finish --state "${state_file}" \
		--status "${status}" --exit-code "${exit_code}" "${report_args[@]}" \
		>/dev/null 2>&1 || true
}

cleanup() {
	local exit_code=$?
	local pid status
	for pid in "${MONITOR_PID}" "${EVAL_PID}" "${GEN_PID}" "${SUITE_CHILD_PID}"; do
		[[ -n "${pid}" ]] || continue
		kill -0 "${pid}" 2>/dev/null || continue
		kill -TERM "${pid}" 2>/dev/null || true
	done
	if [[ -n "${SUITE_STATE_FILE}" ]]; then
		finish_state_file "${SUITE_STATE_FILE}" interrupted "${exit_code}"
	fi
	if [[ -n "${STATE_FILE}" ]]; then
		status=completed
		((exit_code == 0)) || status=failed
		[[ "${INTERRUPTED}" == true ]] && status=interrupted
		finish_state_file "${STATE_FILE}" "${status}" "${exit_code}" "${REPORT_FILE}"
	fi
}

termination_requested() {
	INTERRUPTED=true
	warn "SWE-bench Pro run was interrupted"
	exit 1
}

require_positive_integer() {
	local name="$1"
	local value="$2"
	[[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

WORKERS=8
EVAL_WORKERS=4
STEP_LIMIT=250
PULL_TIMEOUT=1800
COMMAND_TIMEOUT=600
EVAL_CHUNK_SIZE=12
POLL_INTERVAL=60
HEALTH_INTERVAL=30
HEALTH_FAILURES=3
NUM_TASKS=""
SLICE_ARG=""
DATASET_REPO="ScaleAI/SWE-bench_Pro"
DATASET_SPLIT="test"
SERVED_MODEL_NAME=""
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
RUNS=1
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
		--dataset-repo)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			DATASET_REPO="$2"
			shift 2
			;;
		--dataset-split)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			DATASET_SPLIT="$2"
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
			shift 2
			;;
		--runs)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			RUNS="$2"
			shift 2
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
require_positive_integer "--runs" "${RUNS}"
require_positive_integer "--workers" "${WORKERS}"
require_positive_integer "--step-limit" "${STEP_LIMIT}"
require_positive_integer "--pull-timeout" "${PULL_TIMEOUT}"
require_positive_integer "--command-timeout" "${COMMAND_TIMEOUT}"
require_positive_integer "--eval-chunk-size" "${EVAL_CHUNK_SIZE}"
require_positive_integer "--poll-interval" "${POLL_INTERVAL}"
require_positive_integer "--health-interval" "${HEALTH_INTERVAL}"
require_positive_integer "--health-failures" "${HEALTH_FAILURES}"
[[ -z "${EVAL_WORKERS}" ]] || require_positive_integer "--eval-workers" "${EVAL_WORKERS}"
[[ -z "${NUM_TASKS}" ]] || require_positive_integer "--num-tasks" "${NUM_TASKS}"
[[ -z "${SLICE_ARG}" || "${SLICE_ARG}" =~ ^[0-9]*:[0-9]*$ ]] || \
	die "--slice must use START:END format: ${SLICE_ARG}"
[[ -n "${DATASET_REPO}" ]] || die "--dataset-repo must not be empty"
[[ -n "${DATASET_SPLIT}" ]] || die "--dataset-split must not be empty"

init_benchmark_paths
init_vllm_endpoint
trap cleanup EXIT
trap termination_requested INT TERM
RUN_TAG="$(sanitize_run_tag "${RUN_TAG}")"
readonly PYTHON_BIN="$(command -v python)"
abspath() {
	"${PYTHON_BIN}" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}
RESULTS_ROOT="$(abspath "${SWEBENCH_PRO_RESULTS_ROOT:-${AGENT_DIR}/results}")"
readonly RESULTS_ROOT

dataset_label() {
	if [[ "${DATASET_REPO}" == "ScaleAI/SWE-bench_Pro" ]]; then
		printf 'swebench-pro'
	else
		printf '%s' "${DATASET_REPO##*/}"
	fi
}
readonly DATASET_LABEL="$(dataset_label)"

environment_label() {
	if [[ "${PYTHON_BIN}" == */bin/python* ]]; then
		basename "$(dirname "$(dirname "${PYTHON_BIN}")")"
	elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
		printf '%s' "${CONDA_DEFAULT_ENV}"
	else
		basename "$(dirname "${PYTHON_BIN}")"
	fi
}
readonly ENV_LABEL="$(environment_label)"

preflight_pro_env() {
	require_file "${BENCHMARK_DIR}/SWE-bench_Pro-os/swe_bench_pro_eval.py"
	require_file "${AGENT_DIR}/pyproject.toml"
	require_command mini-extra
	require_command python
	require_command docker
	python -c 'import datasets, yaml, pandas' 2>/dev/null || \
		die "SWE-bench Pro generation dependencies are missing in '${PYTHON_BIN}'. Activate the SWE-bench Pro environment (for example: conda activate swe_pro) and rerun with that python on PATH."
	if [[ "${SKIP_EVAL}" == false ]]; then
		python -c 'import docker' 2>/dev/null || \
			die "SWE-bench Pro evaluation dependencies are missing in '${PYTHON_BIN}'."
	fi
}
preflight_pro_env

run_suite() {
	local suite_dir="${RESULTS_ROOT}/swebench_pro_${RUN_TAG}_suite"
	local summary_json="${suite_dir}/summary.json"
	local summary_md="${suite_dir}/summary.md"
	local child_args=() arg skip_next=false
	local run_index run_tag run_started run_finished exit_code report_path log_path
	local failed_runs=0 final_status

	SUITE_STATE_FILE="${suite_dir}/state.json"
	wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
	SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME}")"

	for arg in "${ORIGINAL_ARGS[@]}"; do
		if [[ "${skip_next}" == true ]]; then
			skip_next=false
			continue
		fi
		if [[ "${arg}" == "--runs" ]]; then
			skip_next=true
			continue
		fi
		child_args+=("${arg}")
	done

	mkdir -p "${suite_dir}" "${LOG_DIR}"
	python "${SCRIPT_DIR}/lib/run_state.py" start \
		--state "${SUITE_STATE_FILE}" --name "${RUN_TAG}" --kind swebench-pro-suite \
		--runs-total "${RUNS}" --current-run 1 --pid $$ \
		--meta "dataset=${DATASET_LABEL}" --meta "dataset_repo=${DATASET_REPO}" \
		--meta "host=${VLLM_HOST}" \
		--meta "port=${VLLM_PORT}" --meta "step_limit=${STEP_LIMIT}" \
		--meta "model=${SERVED_MODEL_NAME}" \
		--meta "env=${ENV_LABEL}" --meta "python=${PYTHON_BIN}" \
		--meta "command=${RUNNER_COMMAND}" --message "suite started"

	log "Suite ${RUN_TAG}: ${RUNS} sequential runs; summary -> ${suite_dir}"
	for ((run_index = 1; run_index <= RUNS; run_index++)); do
		printf -v run_tag '%s_run%02d' "${RUN_TAG}" "${run_index}"
		printf -v run_command '%q ' env "SWEBENCH_PRO_RESULTS_ROOT=${suite_dir}/runs" \
			bash "${SCRIPT_DIR}/run_swebenchpro.sh" "${child_args[@]}" \
			--tag "${run_tag}" --runs 1
		run_command="${run_command% }"
		run_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
		log "[suite] Starting run ${run_index}/${RUNS}: ${run_tag}"
		python "${SCRIPT_DIR}/lib/run_state.py" run-start \
			--state "${SUITE_STATE_FILE}" --index "${run_index}" --tag "${run_tag}" \
			--started-at "${run_started}" --command "${run_command}" \
			--log "${LOG_DIR}/swebench_pro_${run_tag}.log"
		exit_code=0
		SWEBENCH_PRO_RESULTS_ROOT="${suite_dir}/runs" \
			bash "${SCRIPT_DIR}/run_swebenchpro.sh" "${child_args[@]}" \
			--tag "${run_tag}" --runs 1 &
		SUITE_CHILD_PID=$!
		wait "${SUITE_CHILD_PID}" || exit_code=$?
		SUITE_CHILD_PID=""
		run_finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
		report_path="${suite_dir}/runs/swebench_pro_${run_tag}/report.json"
		log_path="${LOG_DIR}/swebench_pro_${run_tag}.log"
		python "${SCRIPT_DIR}/lib/run_state.py" run-finish \
			--state "${SUITE_STATE_FILE}" --index "${run_index}" \
			--exit-code "${exit_code}" --report "${report_path}" --log "${log_path}" \
			--finished-at "${run_finished}"
		if ((exit_code == 0)); then
			log "[suite] Run ${run_index}/${RUNS} finished"
		else
			warn "[suite] Run ${run_index}/${RUNS} failed with status ${exit_code}; continuing"
			failed_runs=$((failed_runs + 1))
		fi
	done

	python "${SCRIPT_DIR}/lib/run_state.py" aggregate \
		--state "${SUITE_STATE_FILE}" --output "${summary_json}" --markdown "${summary_md}"
	final_status=completed
	((failed_runs == 0)) || final_status=partial
	python "${SCRIPT_DIR}/lib/run_state.py" finish \
		--state "${SUITE_STATE_FILE}" --status "${final_status}" \
		--exit-code "$((failed_runs > 0))"
	SUITE_STATE_FILE=""
	log "Suite summary (JSON): ${summary_json}"
	log "Suite summary (Markdown): ${summary_md}"
	((failed_runs == 0))
}

if ((RUNS > 1)); then
	run_suite
	exit $?
fi

readonly SWEBENCH_PRO_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os"
readonly OUT_DIR="${RESULTS_ROOT}/swebench_pro_${RUN_TAG}"
readonly GEN_DIR="${OUT_DIR}/generation"
readonly EVAL_ROOT="${OUT_DIR}/eval_chunks"
readonly CLAIMED_FILE="${OUT_DIR}/claimed_ids.txt"
readonly EVAL_REPORT_LIST="${OUT_DIR}/eval_report_list.txt"
readonly PREDS_JSON="${OUT_DIR}/preds.json"
readonly PATCHES_JSON="${OUT_DIR}/patches.json"
readonly INSTANCES_CSV="${OUT_DIR}/instances.csv"
readonly IMAGE_LIST="${OUT_DIR}/images.txt"
readonly REPORT_FILE="${OUT_DIR}/report.json"
readonly STATE_FILE="${OUT_DIR}/state.json"
readonly RUNTIME_CONFIG="${OUT_DIR}/swebench_pro.yaml"
readonly GEN_LOG="${OUT_DIR}/generation.log"
readonly LOG_FILE="${LOG_DIR}/swebench_pro_${RUN_TAG}.log"

require_file "${SWEBENCH_PRO_DIR}/swe_bench_pro_eval.py"
require_file "${AGENT_DIR}/pyproject.toml"
require_file "${AGENT_DIR}/src/minisweagent/run/extra/swebench.py"
require_command mini-extra
require_command python
require_command docker
mkdir -p "${GEN_DIR}" "${EVAL_ROOT}" "${LOG_DIR}"
: >>"${CLAIMED_FILE}"
: >>"${EVAL_REPORT_LIST}"
exec > >(tee -a "${LOG_FILE}") 2>&1

python "${SCRIPT_DIR}/lib/run_state.py" start \
	--state "${STATE_FILE}" --name "${RUN_TAG}" --kind swebench-pro \
	--runs-total 1 --current-run 1 --pid $$ \
	--meta "dataset=${DATASET_LABEL}" --meta "dataset_repo=${DATASET_REPO}" \
	--meta "host=${VLLM_HOST}" \
	--meta "port=${VLLM_PORT}" --meta "step_limit=${STEP_LIMIT}" \
	--meta "env=${ENV_LABEL}" --meta "python=${PYTHON_BIN}" \
	--meta "command=${RUNNER_COMMAND}" --message "waiting for vLLM"

python -c 'import datasets, yaml' 2>/dev/null || \
	die "Generation dependencies are unavailable; run: bash setup_swebenchpro.sh"
if [[ "${SKIP_EVAL}" == false ]]; then
	python -c 'import docker, pandas' 2>/dev/null || \
		die "Local evaluation dependencies are unavailable; run: bash setup_swebenchpro.sh"
fi

wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME}")"
python "${SCRIPT_DIR}/lib/run_state.py" meta --state "${STATE_FILE}" \
	--meta "model=${SERVED_MODEL_NAME}" --message "generating"
EVAL_WORKERS="${EVAL_WORKERS:-${WORKERS}}"
readonly EVAL_PREFIX="$(sanitize_run_tag "${SERVED_MODEL_NAME}_step${STEP_LIMIT}_${RUN_TAG}")"

DATASET_SIZE="$(python "${SCRIPT_DIR}/lib/benchmark_data.py" pro-dataset-size \
	--dataset "${DATASET_REPO}" --split "${DATASET_SPLIT}")"
[[ "${DATASET_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
	die "Could not determine dataset size for ${DATASET_REPO} (${DATASET_SPLIT})"
log "Dataset ${DATASET_REPO} (${DATASET_SPLIT}): ${DATASET_SIZE} instances"
full_slice_plan="$(plan_batch_slices "${DATASET_SIZE}" "${NUM_TASKS}" "${SLICE_ARG}" "${DATASET_SIZE}")"
mapfile -t FULL_SLICE_LINES <<<"${full_slice_plan}"
[[ ${#FULL_SLICE_LINES[@]} -eq 1 ]] || die "Unexpected slice plan for SWE-bench Pro selection"
readonly FULL_SLICE="${FULL_SLICE_LINES[0]}"

python "${SCRIPT_DIR}/lib/benchmark_data.py" select-pro \
	--csv "${INSTANCES_CSV}" --images "${IMAGE_LIST}" --slice "${FULL_SLICE}" \
	--dataset "${DATASET_REPO}" --split "${DATASET_SPLIT}"
python "${SCRIPT_DIR}/lib/benchmark_data.py" pro-config \
	--output "${RUNTIME_CONFIG}" --model "${SERVED_MODEL_NAME}" \
	--base-url "${OPENAI_BASE_URL}" --api-key "${VLLM_API_KEY}" \
	--step-limit "${STEP_LIMIT}" --pull-timeout "${PULL_TIMEOUT}" \
	--command-timeout "${COMMAND_TIMEOUT}"

EVAL_OPTIONS=(--use_local_docker)
[[ "${BLOCK_NETWORK}" == false ]] || EVAL_OPTIONS+=(--block_network)

log "Generating SWE-bench Pro selection ${FULL_SLICE} with model ${SERVED_MODEL_NAME}"
(
	cd "${AGENT_DIR}"
	exec mini-extra swebench \
		--subset "${DATASET_REPO}" \
		--split "${DATASET_SPLIT}" \
		--model "${SERVED_MODEL_NAME}" \
		--workers "${WORKERS}" \
		--output "${GEN_DIR}" \
		--config "${RUNTIME_CONFIG}" \
		--slice "${FULL_SLICE}"
) >"${GEN_LOG}" 2>&1 &
GEN_PID=$!
log "Generation console progress: tail -f ${GEN_LOG}"
log "mini-SWE-agent execution log: tail -f ${GEN_DIR}/minisweagent.log"

monitor_vllm() {
	local failures=0
	while true; do
		sleep "${HEALTH_INTERVAL}"
		if vllm_curl "${VLLM_ORIGIN}/health" --connect-timeout 5 --max-time 30 >/dev/null 2>&1; then
			failures=0
			continue
		fi

		failures=$((failures + 1))
		warn "vLLM health check failed (${failures}/${HEALTH_FAILURES}): ${VLLM_ORIGIN}/health"
		if ((failures >= HEALTH_FAILURES)); then
			warn "vLLM is unavailable; stopping SWE-bench Pro"
			kill -TERM "${GEN_PID}" 2>/dev/null || true
			kill -TERM "${RUNNER_PID}" 2>/dev/null || true
			return
		fi
	done
}

stop_vllm_monitor() {
	if [[ -n "${MONITOR_PID}" ]]; then
		kill -TERM "${MONITOR_PID}" 2>/dev/null || true
		wait "${MONITOR_PID}" 2>/dev/null || true
		MONITOR_PID=""
	fi
	return 0
}

monitor_vllm &
MONITOR_PID=$!

process_pending_chunk() {
	local chunk_dir="$1"
	shift
	local chunk_ids_file="${chunk_dir}/ids.txt"
	local chunk_instances="${chunk_dir}/instances.csv"
	local chunk_patches="${chunk_dir}/patches.json"
	local chunk_images="${chunk_dir}/images.txt"
	local chunk_eval="${chunk_dir}/evaluation"
	mkdir -p "${chunk_dir}"
	printf '%s\n' "$@" >"${chunk_ids_file}"

	python "${SCRIPT_DIR}/lib/benchmark_data.py" prepare-pro-chunk \
		--preds "${GEN_DIR}/preds.json" --ids "${chunk_ids_file}" \
		--instances "${INSTANCES_CSV}" --output-instances "${chunk_instances}" \
		--patches "${chunk_patches}" --images "${chunk_images}" --prefix "${EVAL_PREFIX}"

	if [[ "${SKIP_EVAL}" == false ]]; then
		log "Evaluating $# SWE-bench Pro instances with ${EVAL_WORKERS} workers"
		rm -f -- "${chunk_eval}/eval_results.json"
		(
			cd "${SWEBENCH_PRO_DIR}"
			exec python swe_bench_pro_eval.py \
				--raw_sample_path "${chunk_instances}" \
				--patch_path "${chunk_patches}" \
				--output_dir "${chunk_eval}" \
				--scripts_dir run_scripts \
				--dockerhub_username jefzda \
				--num_workers "${EVAL_WORKERS}" \
				"${EVAL_OPTIONS[@]}"
		) &
		EVAL_PID=$!
		wait "${EVAL_PID}"
		EVAL_PID=""
		require_file "${chunk_eval}/eval_results.json"
		printf '%s\n' "${chunk_eval}/eval_results.json" >>"${EVAL_REPORT_LIST}"
		python "${SCRIPT_DIR}/lib/benchmark_data.py" merge-pro-reports \
			--chunk-list "${EVAL_REPORT_LIST}" --report "${REPORT_FILE}"
	fi

	remove_chunk_images "${chunk_images}"
	cat "${chunk_ids_file}" >>"${CLAIMED_FILE}"
}

gen_running() {
	kill -0 "${GEN_PID}" 2>/dev/null
}

pending_instance_ids() {
	python "${SCRIPT_DIR}/lib/benchmark_data.py" pending-predictions \
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
python "${SCRIPT_DIR}/lib/benchmark_data.py" normalize-pro \
	--source "${PREDS_JSON}" --output "${PATCHES_JSON}" --prefix "${EVAL_PREFIX}"

if [[ "${SKIP_EVAL}" == false ]]; then
	python "${SCRIPT_DIR}/lib/benchmark_data.py" merge-pro-reports \
		--chunk-list "${EVAL_REPORT_LIST}" --report "${REPORT_FILE}"
fi

log "Patches: ${PATCHES_JSON}"
if [[ -f "${REPORT_FILE}" ]]; then
	python "${SCRIPT_DIR}/lib/benchmark_data.py" pro-report --report "${REPORT_FILE}"
	log "Report: ${REPORT_FILE}"
fi
log "Log: ${LOG_FILE}"

[[ "${GEN_EXIT}" -eq 0 ]] || die "mini-SWE-agent generation failed (exit ${GEN_EXIT}); see ${GEN_LOG}"
