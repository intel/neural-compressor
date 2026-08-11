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
	--batch-size N       Instances per generation/evaluation batch (default: 25)
	--tag TAG            Run tag (default: UTC timestamp)
	--redo               Re-run existing generation and evaluation results
	--skip-eval          Generate patches without local evaluation
	--block-network      Disable container networking during local evaluation
	--keep-images        Keep SWE-bench Pro Docker images after each batch
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
	local image_list="${CURRENT_IMAGE_LIST}"
	CURRENT_IMAGE_LIST=""
	[[ "${KEEP_IMAGES}" == false && -n "${image_list}" && -f "${image_list}" ]] || return
	mapfile -t images < <(sort -u "${image_list}" | grep -E '^jefzda/sweap-images:' || true)
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
BATCH_SIZE=25
NUM_TASKS=""
SLICE_ARG=""
SERVED_MODEL_NAME=""
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
REDO=false
SKIP_EVAL=false
BLOCK_NETWORK=false
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
		--command-timeout)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			COMMAND_TIMEOUT="$2"
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
require_positive_integer "--batch-size" "${BATCH_SIZE}"
[[ -z "${EVAL_WORKERS}" ]] || require_positive_integer "--eval-workers" "${EVAL_WORKERS}"
[[ -z "${NUM_TASKS}" ]] || require_positive_integer "--num-tasks" "${NUM_TASKS}"
[[ -z "${SLICE_ARG}" || "${SLICE_ARG}" =~ ^[0-9]*:[0-9]*$ ]] || \
	die "--slice must use START:END format: ${SLICE_ARG}"

init_benchmark_paths
init_vllm_endpoint
RUN_TAG="$(sanitize_run_tag "${RUN_TAG}")"
readonly SWEBENCH_PRO_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os"
readonly OUT_DIR="${AGENT_DIR}/results/swebench_pro_${RUN_TAG}"
readonly BATCH_ROOT="${OUT_DIR}/batches"
readonly BATCH_LIST="${OUT_DIR}/batch_list.txt"
readonly PREDS_JSON="${OUT_DIR}/preds.json"
readonly PATCHES_JSON="${OUT_DIR}/patches.json"
readonly INSTANCES_CSV="${OUT_DIR}/instances.csv"
readonly IMAGE_LIST="${OUT_DIR}/images.txt"
readonly EVAL_OUT="${OUT_DIR}/evaluation"
readonly REPORT_FILE="${EVAL_OUT}/eval_results.json"
readonly RUNTIME_CONFIG="${OUT_DIR}/swebench_pro.yaml"
readonly LOG_FILE="${LOG_DIR}/swebench_pro_${RUN_TAG}.log"

require_file "${SWEBENCH_PRO_DIR}/swe_bench_pro_eval.py"
require_file "${AGENT_DIR}/pyproject.toml"
require_file "${AGENT_DIR}/src/minisweagent/run/extra/swebench.py"
require_command mini-extra
require_command python
require_command docker
mkdir -p "${BATCH_ROOT}" "${LOG_DIR}"
: >"${BATCH_LIST}"
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

REDO_OPTIONS=()
[[ "${REDO}" == false ]] || REDO_OPTIONS=(--redo-existing)
batch_plan="$(plan_batch_slices 731 "${NUM_TASKS}" "${SLICE_ARG}" "${BATCH_SIZE}")"
mapfile -t BATCH_SLICES <<<"${batch_plan}"
[[ ${#BATCH_SLICES[@]} -gt 0 ]] || die "No SWE-bench Pro batches were selected"

python "${SCRIPT_DIR}/lib/benchmark_data.py" pro-config \
	--output "${RUNTIME_CONFIG}" --model "${SERVED_MODEL_NAME}" \
	--base-url "${OPENAI_BASE_URL}" --api-key "${VLLM_API_KEY}" \
	--step-limit "${STEP_LIMIT}" --pull-timeout "${PULL_TIMEOUT}" \
	--command-timeout "${COMMAND_TIMEOUT}"

EVAL_OPTIONS=(--use_local_docker)
[[ "${REDO}" == false ]] || EVAL_OPTIONS+=(--redo)
[[ "${BLOCK_NETWORK}" == false ]] || EVAL_OPTIONS+=(--block_network)

for batch_slice in "${BATCH_SLICES[@]}"; do
	IFS=: read -r batch_start batch_end <<<"${batch_slice}"
	printf -v batch_name 'batch_%06d_%06d' "${batch_start}" "${batch_end}"
	batch_dir="${BATCH_ROOT}/${batch_name}"
	batch_preds="${batch_dir}/preds.json"
	batch_patches="${batch_dir}/patches.json"
	batch_instances="${batch_dir}/instances.csv"
	batch_eval="${batch_dir}/evaluation"
	CURRENT_IMAGE_LIST="${batch_dir}/images.txt"
	mkdir -p "${batch_dir}"
	printf '%s\n' "${batch_dir}" >>"${BATCH_LIST}"

	python "${SCRIPT_DIR}/lib/benchmark_data.py" select-pro \
		--csv "${batch_instances}" --images "${CURRENT_IMAGE_LIST}" --slice "${batch_slice}"

	log "Generating SWE-bench Pro batch ${batch_slice} with model ${SERVED_MODEL_NAME}"
	(
		cd "${AGENT_DIR}"
		mini-extra swebench \
			--subset ScaleAI/SWE-bench_Pro \
			--split test \
			--model "${SERVED_MODEL_NAME}" \
			--workers "${WORKERS}" \
			--output "${batch_dir}" \
			--config "${RUNTIME_CONFIG}" \
			--slice "${batch_slice}" \
			"${REDO_OPTIONS[@]}"
	)

	require_file "${batch_preds}"
	python "${SCRIPT_DIR}/lib/benchmark_data.py" normalize-pro \
		--source "${batch_preds}" --output "${batch_patches}" --prefix "${EVAL_PREFIX}"

	if [[ "${SKIP_EVAL}" == false ]]; then
		rm -f -- "${batch_eval}/eval_results.json"
		log "Evaluating SWE-bench Pro batch ${batch_slice} with ${EVAL_WORKERS} workers"
		(
			cd "${SWEBENCH_PRO_DIR}"
			python swe_bench_pro_eval.py \
				--raw_sample_path "${batch_instances}" \
				--patch_path "${batch_patches}" \
				--output_dir "${batch_eval}" \
				--scripts_dir run_scripts \
				--dockerhub_username jefzda \
				--num_workers "${EVAL_WORKERS}" \
				"${EVAL_OPTIONS[@]}"
		)
		require_file "${batch_eval}/eval_results.json"
	else
		rm -f -- "${batch_eval}/eval_results.json"
		log "Skipping local evaluation for SWE-bench Pro batch ${batch_slice}"
	fi

	remove_swebench_pro_images
done

python "${SCRIPT_DIR}/lib/benchmark_data.py" merge-pro \
	--batch-list "${BATCH_LIST}" --predictions "${PREDS_JSON}" \
	--patches "${PATCHES_JSON}" --instances "${INSTANCES_CSV}" \
	--images "${IMAGE_LIST}" --report "${REPORT_FILE}"

log "Patches: ${PATCHES_JSON}"
if [[ -f "${REPORT_FILE}" ]]; then
	python "${SCRIPT_DIR}/lib/benchmark_data.py" pro-report --report "${REPORT_FILE}"
	log "Report: ${REPORT_FILE}"
fi
log "Log: ${LOG_FILE}"
