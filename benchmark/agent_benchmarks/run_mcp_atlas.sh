#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=versions.env
source "${SCRIPT_DIR}/versions.env"

usage() {
	cat <<'EOF'
Usage:
	bash run_mcp_atlas.sh [OPTIONS]

Run MCP-Atlas against an already-running OpenAI-compatible vLLM server. The
script starts the MCP sandbox and TypeScript agent harness, runs the benchmark,
and scores results with an LLM judge unless --skip-score is specified.

Options:
	--host HOST                   vLLM host (default: 127.0.0.1)
	--port PORT                   vLLM port (default: 8888)
	--served-name NAME            Served model ID (default: discover)
	--sandbox-port PORT           MCP sandbox port (default: 1984)
	--harness-port PORT           Agent harness port (default: 3001)
	--workers N                   Parallel benchmark tasks (default: 5)
	--score-workers N             Parallel judge requests (default: 10)
	--num-tasks N                 Run the first N tasks (default: all 500)
	--timeout N                   Per-task timeout in seconds (default: 1800)
	--max-turns N                 Maximum agent turns (default: harness default)
	--max-tool-calls N            Maximum tool calls per task (default: harness default)
	--tool-output-cap N           Truncate each tool result to N characters
	--context-window-management compact
	                              Enable compact context management
	--extra-llm-params JSON       Extra completion parameters as JSON
	--system-prompt TEXT          Prepend a system prompt to every task
	--judge-model NAME            Scoring model (default: served model)
	--tag TAG                     Run tag (default: UTC timestamp)
	--skip-health-check           Skip real-call checks for all MCP servers
	--skip-score                  Generate responses without LLM-as-judge scoring
	--reuse-sandbox               Require and reuse an existing sandbox
	--reuse-harness               Require and reuse an existing harness
	--keep-image                  Keep the MCP sandbox image after the run
	-h, --help                    Show this help message

Environment:
	VLLM_API_KEY                  vLLM API key (default: EMPTY)
	VLLM_WAIT_TIMEOUT             vLLM readiness timeout (default: 300)
	VLLM_PID_FILE                 vLLM PID file (default: logs/vllm_<PORT>.pid)
	OPENAI_BASE_URL               Override vLLM API base URL
	MCP_SANDBOX_WAIT_TIMEOUT      Sandbox readiness timeout (default: 300)
	MCP_HARNESS_WAIT_TIMEOUT      Harness readiness timeout (default: 120)
	TOOL_CALL_TIMEOUT_MS          Harness tool-call timeout (default: 60000)
	LIST_TOOLS_TIMEOUT_MS         Harness list-tools timeout (default: 180000)
	LLM_TIMEOUT_MS                Harness LLM timeout (default: 600000)
EOF
}

require_positive_integer() {
	local name="$1"
	local value="$2"
	[[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

stop_vllm() {
	local pid_file="${VLLM_PID_FILE:-${LOG_DIR}/vllm_${VLLM_PORT}.pid}"
	stop_vllm_from_pid_file "${pid_file}"
}

cleanup() {
	if [[ -n "${HARNESS_PID}" ]] && kill -0 "${HARNESS_PID}" 2>/dev/null; then
		log "Stopping MCP agent harness (PID=${HARNESS_PID})"
		kill -TERM -- "-${HARNESS_PID}" 2>/dev/null || kill -TERM "${HARNESS_PID}" 2>/dev/null || true
	fi
	if [[ -n "${SANDBOX_CONTAINER}" ]]; then
		log "Stopping MCP sandbox ${SANDBOX_CONTAINER}"
		docker rm -f "${SANDBOX_CONTAINER}" >/dev/null 2>&1 || true
	fi
	if [[ "${KEEP_IMAGE}" == false && "${OWN_SANDBOX_IMAGE}" == true ]]; then
		log "Removing MCP sandbox image ${MCP_ATLAS_IMAGE}"
		docker image rm agent-environment:latest "${MCP_ATLAS_IMAGE}" >/dev/null 2>&1 || \
			warn "MCP sandbox image is still in use and could not be removed"
	fi
	stop_vllm
}

wait_for_http() {
	local name="$1"
	local url="$2"
	local timeout="$3"
	local deadline=$((SECONDS + timeout))
	log "Waiting up to ${timeout}s for ${name}: ${url}"
	while ((SECONDS < deadline)); do
		if curl --noproxy "127.0.0.1,localhost" -fsS --connect-timeout 2 --max-time 10 \
			"${url}" >/dev/null 2>&1; then
			log "${name} is ready"
			return 0
		fi
		sleep 2
	done
	die "${name} did not become ready within ${timeout}s: ${url}"
}

WORKERS=5
SCORE_WORKERS=10
NUM_TASKS=""
TASK_TIMEOUT=1800
MAX_TURNS=""
MAX_TOOL_CALLS=""
TOOL_OUTPUT_CAP=""
CONTEXT_MANAGEMENT=""
EXTRA_LLM_PARAMS=""
SYSTEM_PROMPT=""
SERVED_MODEL_NAME=""
JUDGE_MODEL=""
SANDBOX_PORT=1984
HARNESS_PORT=3001
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
SKIP_HEALTH_CHECK=false
SKIP_SCORE=false
REUSE_SANDBOX=false
REUSE_HARNESS=false
KEEP_IMAGE=false
OWN_SANDBOX_IMAGE=false
HARNESS_PID=""
SANDBOX_CONTAINER=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--host)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			VLLM_HOST="$2"; shift 2
			;;
		--port)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			VLLM_PORT="$2"; shift 2
			;;
		--served-name)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			SERVED_MODEL_NAME="$2"; shift 2
			;;
		--sandbox-port)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			SANDBOX_PORT="$2"; shift 2
			;;
		--harness-port)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			HARNESS_PORT="$2"; shift 2
			;;
		--workers)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			WORKERS="$2"; shift 2
			;;
		--score-workers)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			SCORE_WORKERS="$2"; shift 2
			;;
		--num-tasks)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			NUM_TASKS="$2"; shift 2
			;;
		--timeout)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			TASK_TIMEOUT="$2"; shift 2
			;;
		--max-turns)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			MAX_TURNS="$2"; shift 2
			;;
		--max-tool-calls)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			MAX_TOOL_CALLS="$2"; shift 2
			;;
		--tool-output-cap)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			TOOL_OUTPUT_CAP="$2"; shift 2
			;;
		--context-window-management)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			CONTEXT_MANAGEMENT="$2"; shift 2
			;;
		--extra-llm-params)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			EXTRA_LLM_PARAMS="$2"; shift 2
			;;
		--system-prompt)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			SYSTEM_PROMPT="$2"; shift 2
			;;
		--judge-model)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			JUDGE_MODEL="$2"; shift 2
			;;
		--tag)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			RUN_TAG="$2"; shift 2
			;;
		--skip-health-check)
			SKIP_HEALTH_CHECK=true; shift
			;;
		--skip-score)
			SKIP_SCORE=true; shift
			;;
		--reuse-sandbox)
			REUSE_SANDBOX=true; shift
			;;
		--reuse-harness)
			REUSE_HARNESS=true; shift
			;;
		--keep-image)
			KEEP_IMAGE=true; shift
			;;
		-h | --help)
			usage; exit 0
			;;
		*)
			die "Unknown argument: $1"
			;;
	esac
done

validate_port "${SANDBOX_PORT}"
validate_port "${HARNESS_PORT}"
[[ "${SANDBOX_PORT}" != "${HARNESS_PORT}" ]] || die "Sandbox and harness ports must differ"
require_positive_integer "--workers" "${WORKERS}"
require_positive_integer "--score-workers" "${SCORE_WORKERS}"
require_positive_integer "--timeout" "${TASK_TIMEOUT}"
[[ -z "${NUM_TASKS}" ]] || require_positive_integer "--num-tasks" "${NUM_TASKS}"
[[ -z "${MAX_TURNS}" ]] || require_positive_integer "--max-turns" "${MAX_TURNS}"
[[ -z "${MAX_TOOL_CALLS}" ]] || require_positive_integer "--max-tool-calls" "${MAX_TOOL_CALLS}"
[[ -z "${TOOL_OUTPUT_CAP}" ]] || require_positive_integer "--tool-output-cap" "${TOOL_OUTPUT_CAP}"
[[ -z "${CONTEXT_MANAGEMENT}" || "${CONTEXT_MANAGEMENT}" == "compact" ]] || \
	die "--context-window-management must be compact"

init_benchmark_paths
init_vllm_endpoint
RUN_TAG="$(sanitize_run_tag "${RUN_TAG}")"
readonly NODE_RUNTIME_DIR="${BENCHMARK_DIR}/.tools/node-v${MCP_ATLAS_NODE_VERSION}"
node_major=0
if command -v node >/dev/null 2>&1; then
	node_major="$(node -p 'Number(process.versions.node.split(".")[0])' 2>/dev/null || printf '0')"
fi
if ((node_major < 20)) && [[ -x "${NODE_RUNTIME_DIR}/bin/node" ]]; then
	export PATH="${NODE_RUNTIME_DIR}/bin:${PATH}"
fi
readonly OUT_DIR="${MCP_DIR}/outputs/run_${RUN_TAG}"
readonly OUTPUT_CSV="${OUT_DIR}/outputs.csv"
readonly GROUNDTRUTH_CSV="${MCP_DIR}/outputs/groundtruth.csv"
readonly SCORED_DIR="${OUT_DIR}/scored"
readonly HARNESS_LOG="${OUT_DIR}/harness.log"
readonly LOG_FILE="${LOG_DIR}/mcp_atlas_${RUN_TAG}.log"
readonly SANDBOX_URL="http://127.0.0.1:${SANDBOX_PORT}"
readonly HARNESS_URL_VALUE="http://127.0.0.1:${HARNESS_PORT}"

require_file "${MCP_DIR}/run_eval.py"
require_file "${MCP_DIR}/.env"
require_file "${MCP_DIR}/services/agent-harness/dist/index.js"
require_file "${MCP_DIR}/services/scoring/score_claims.py"
require_command python
require_command node
require_command docker
require_command curl
require_command setsid
node_major="$(node -p 'Number(process.versions.node.split(".")[0])')"
((node_major >= 20)) || \
	die "MCP-Atlas requires Node.js 20 or newer; run: bash setup_mcp_atlas.sh"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap cleanup EXIT

python -c 'import aiohttp, datasets, pandas' 2>/dev/null || \
	die "MCP-Atlas Python dependencies are unavailable; run: bash setup_mcp_atlas.sh"
wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME}")"
JUDGE_MODEL="${JUDGE_MODEL:-${SERVED_MODEL_NAME}}"
readonly SCORE_LABEL="$(sanitize_run_tag "${SERVED_MODEL_NAME}")"

if [[ "${REUSE_SANDBOX}" == true ]]; then
	wait_for_http "existing MCP sandbox" "${SANDBOX_URL}/enabled-servers" 10
elif curl --noproxy "127.0.0.1,localhost" -fsS "${SANDBOX_URL}/enabled-servers" >/dev/null 2>&1; then
	die "Port ${SANDBOX_PORT} already has an MCP sandbox; pass --reuse-sandbox or choose another port"
else
	SANDBOX_CONTAINER="mcp-atlas-${RUN_TAG}"
	log "Starting MCP sandbox ${SANDBOX_CONTAINER} from ${MCP_ATLAS_IMAGE}"
	docker run -d --name "${SANDBOX_CONTAINER}" \
		-p "${SANDBOX_PORT}:1984" \
		--env-file "${MCP_DIR}/.env" \
		"${MCP_ATLAS_IMAGE}" >/dev/null
	OWN_SANDBOX_IMAGE=true
	wait_for_http "MCP sandbox" "${SANDBOX_URL}/enabled-servers" \
		"${MCP_SANDBOX_WAIT_TIMEOUT:-300}"
fi

if [[ "${REUSE_HARNESS}" == true ]]; then
	wait_for_http "existing agent harness" "${HARNESS_URL_VALUE}/health" 10
elif curl --noproxy "127.0.0.1,localhost" -fsS "${HARNESS_URL_VALUE}/health" >/dev/null 2>&1; then
	die "Port ${HARNESS_PORT} already has an agent harness; pass --reuse-harness or choose another port"
else
	log "Starting MCP agent harness on port ${HARNESS_PORT}"
	(
		cd "${MCP_DIR}/services/agent-harness"
		exec setsid env \
			PORT="${HARNESS_PORT}" \
			LLM_BASE_URL="${OPENAI_BASE_URL}" \
			LLM_API_KEY="${VLLM_API_KEY}" \
			MCP_SANDBOX_URL="${SANDBOX_URL}" \
			TOOL_CALL_TIMEOUT_MS="${TOOL_CALL_TIMEOUT_MS:-60000}" \
			LIST_TOOLS_TIMEOUT_MS="${LIST_TOOLS_TIMEOUT_MS:-180000}" \
			LLM_TIMEOUT_MS="${LLM_TIMEOUT_MS:-600000}" \
			node dist/index.js
	) >"${HARNESS_LOG}" 2>&1 &
	HARNESS_PID=$!
	wait_for_http "agent harness" "${HARNESS_URL_VALUE}/health" \
		"${MCP_HARNESS_WAIT_TIMEOUT:-120}"
fi

RUN_OPTIONS=()
[[ -z "${NUM_TASKS}" ]] || RUN_OPTIONS+=(--num-tasks "${NUM_TASKS}")
[[ -z "${MAX_TURNS}" ]] || RUN_OPTIONS+=(--max-turns "${MAX_TURNS}")
[[ -z "${MAX_TOOL_CALLS}" ]] || RUN_OPTIONS+=(--max-tool-calls "${MAX_TOOL_CALLS}")
[[ -z "${TOOL_OUTPUT_CAP}" ]] || RUN_OPTIONS+=(--tool-output-cap "${TOOL_OUTPUT_CAP}")
[[ -z "${CONTEXT_MANAGEMENT}" ]] || RUN_OPTIONS+=(--context-window-management "${CONTEXT_MANAGEMENT}")
[[ -z "${EXTRA_LLM_PARAMS}" ]] || RUN_OPTIONS+=(--extra-llm-params "${EXTRA_LLM_PARAMS}")
[[ -z "${SYSTEM_PROMPT}" ]] || RUN_OPTIONS+=(--system-prompt "${SYSTEM_PROMPT}")
[[ "${SKIP_HEALTH_CHECK}" == false ]] || RUN_OPTIONS+=(--skip-health-check)

log "Running MCP-Atlas with model ${SERVED_MODEL_NAME} and ${WORKERS} workers"
(
	cd "${MCP_DIR}"
	HARNESS_URL="${HARNESS_URL_VALUE}" MCP_SANDBOX_URL="${SANDBOX_URL}" \
		python run_eval.py \
			--model "${SERVED_MODEL_NAME}" \
			--output "${OUTPUT_CSV}" \
			--concurrency "${WORKERS}" \
			--timeout "${TASK_TIMEOUT}" \
			--image "${MCP_ATLAS_IMAGE}" \
			"${RUN_OPTIONS[@]}"
)
require_file "${OUTPUT_CSV}"

if [[ "${SKIP_SCORE}" == true ]]; then
	log "Skipping MCP-Atlas LLM-as-judge scoring"
	log "Responses: ${OUTPUT_CSV}"
	log "Log: ${LOG_FILE}"
	exit 0
fi

if [[ ! -f "${GROUNDTRUTH_CSV}" ]]; then
	log "Exporting MCP-Atlas ground truth from Hugging Face"
	python - "${GROUNDTRUTH_CSV}" <<'PY'
import os
import sys
from datasets import load_dataset

destination = sys.argv[1]
os.makedirs(os.path.dirname(destination), exist_ok=True)
dataset = load_dataset("ScaleAI/MCP-Atlas", split="train")
dataset.to_pandas().to_csv(destination, index=False)
print(f"Wrote {len(dataset)} tasks to {destination}")
PY
fi

SCORE_OPTIONS=()
[[ -z "${NUM_TASKS}" ]] || SCORE_OPTIONS+=(--num-tasks "${NUM_TASKS}")
mkdir -p "${SCORED_DIR}"
cp "${OUT_DIR}/run_config.json" "${SCORED_DIR}/run_config.json"
log "Scoring MCP-Atlas responses with judge ${JUDGE_MODEL}"
(
	cd "${MCP_DIR}"
	EVAL_LLM_MODEL="${JUDGE_MODEL}" \
	EVAL_LLM_API_KEY="${VLLM_API_KEY}" \
	EVAL_LLM_BASE_URL="${OPENAI_BASE_URL}" \
	LLM_API_KEY="${VLLM_API_KEY}" \
	LLM_BASE_URL="${OPENAI_BASE_URL}" \
		python services/scoring/score_claims.py \
			--groundtruth-file "${GROUNDTRUTH_CSV}" \
			--model-file "${OUTPUT_CSV}" \
			--model-name "${SCORE_LABEL}" \
			--evaluator-model "${JUDGE_MODEL}" \
			--api-key "${VLLM_API_KEY}" \
			--base-url "${OPENAI_BASE_URL}" \
			--concurrency "${SCORE_WORKERS}" \
			--output-dir "${SCORED_DIR}" \
			"${SCORE_OPTIONS[@]}"
)

python - "${SCORED_DIR}" <<'PY'
import glob
import json
import os
import sys

files = sorted(glob.glob(os.path.join(sys.argv[1], "coverage_stats_*_combined.json")))
if not files:
    raise RuntimeError(f"No combined coverage report found in {sys.argv[1]}")
with open(files[-1]) as file:
    report = json.load(file)
stats = report.get("all", report)
print(f"Tasks       : {stats.get('total_tasks', '?')}")
print(f"Pass@0.75   : {stats.get('pass_rate_0.75', '?')}%")
print(f"Mean coverage: {float(stats.get('mean_coverage', 0)):.4f}")
print(f"Report      : {files[-1]}")
PY

log "Responses: ${OUTPUT_CSV}"
log "Scored results: ${SCORED_DIR}"
log "Log: ${LOG_FILE}"
