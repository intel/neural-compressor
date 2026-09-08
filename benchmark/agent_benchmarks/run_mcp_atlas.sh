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
	--workers N                   Parallel benchmark tasks (default: 10)
	--score-workers N             Parallel judge requests (default: 10)
	--num-tasks N                 Run the first N tasks (default: all 500)
	--timeout N                   Per-task timeout in seconds (default: 1800)
	--health-interval N           Seconds between vLLM health checks (default: 30)
	--health-failures N           Consecutive failed health checks before stopping (default: 3)
	--tag TAG                     Run tag (default: UTC timestamp)
	--skip-health-check           Skip real-call checks for all MCP servers
	--skip-score                  Generate responses without LLM-as-judge scoring
	--keep-image                  Keep the MCP sandbox image after the run
	-h, --help                    Show this help message

Environment:
	VLLM_API_KEY                  vLLM API key (default: EMPTY)
	VLLM_WAIT_TIMEOUT             vLLM readiness timeout (default: 300)
	OPENAI_BASE_URL               Override vLLM API base URL
	MCP_LLM_BASE_URL              LLM API origin without /v1 (derived from OPENAI_BASE_URL)
	MCP_SANDBOX_WAIT_TIMEOUT      Sandbox readiness timeout (default: 300)
	MCP_UV_HTTP_TIMEOUT           uv download timeout in sandbox seconds (default: 120)
	MCP_HARNESS_WAIT_TIMEOUT      Harness readiness timeout (default: 120)
	TOOL_CALL_TIMEOUT_MS          Harness tool-call timeout (default: 60000)
	LIST_TOOLS_TIMEOUT_MS         Harness list-tools timeout (default: 180000)
	LLM_TIMEOUT_MS                Harness LLM timeout (default: 600000)
	EVAL_LLM_TIMEOUT_MS           Judge request timeout (default: LLM_TIMEOUT_MS)
EOF
}

require_positive_integer() {
	local name="$1"
	local value="$2"
	[[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

readonly RUNNER_PID=$$
MONITOR_PID=""
ACTIVE_PID=""

cleanup() {
	local pid
	for pid in "${MONITOR_PID}" "${ACTIVE_PID}"; do
		[[ -n "${pid}" ]] || continue
		kill -0 "${pid}" 2>/dev/null || continue
		kill -TERM "${pid}" 2>/dev/null || true
	done
	if [[ -n "${HARNESS_PID}" ]] && kill -0 "${HARNESS_PID}" 2>/dev/null; then
		log "Stopping MCP agent harness (PID=${HARNESS_PID})"
		kill -TERM -- "-${HARNESS_PID}" 2>/dev/null || kill -TERM "${HARNESS_PID}" 2>/dev/null || true
	fi
	if [[ -n "${SANDBOX_CONTAINER}" ]]; then
		docker logs "${SANDBOX_CONTAINER}" >"${SANDBOX_LOG}" 2>&1 || true
		log "MCP sandbox log: ${SANDBOX_LOG}"
		log "Stopping MCP sandbox ${SANDBOX_CONTAINER}"
		docker rm -f "${SANDBOX_CONTAINER}" >/dev/null 2>&1 || true
	fi
	if [[ "${KEEP_IMAGE}" == false && "${OWN_SANDBOX_IMAGE}" == true ]]; then
		log "Removing MCP sandbox image ${MCP_ATLAS_IMAGE}"
		docker image rm agent-environment:latest "${MCP_ATLAS_IMAGE}" >/dev/null 2>&1 || \
			warn "MCP sandbox image is still in use and could not be removed"
	fi
}

termination_requested() {
	warn "MCP-Atlas run was interrupted"
	exit 1
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
			warn "vLLM is unavailable; stopping MCP-Atlas"
			[[ -z "${ACTIVE_PID}" ]] || kill -TERM "${ACTIVE_PID}" 2>/dev/null || true
			kill -TERM "${RUNNER_PID}" 2>/dev/null || true
			return
		fi
	done
}

WORKERS=10
SCORE_WORKERS=10
NUM_TASKS=""
TASK_TIMEOUT=1800
HEALTH_INTERVAL=30
HEALTH_FAILURES=3
SERVED_MODEL_NAME=""
SANDBOX_PORT=1984
HARNESS_PORT=3001
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
SKIP_HEALTH_CHECK=false
SKIP_SCORE=false
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
		--health-interval)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			HEALTH_INTERVAL="$2"; shift 2
			;;
		--health-failures)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			HEALTH_FAILURES="$2"; shift 2
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
require_positive_integer "--health-interval" "${HEALTH_INTERVAL}"
require_positive_integer "--health-failures" "${HEALTH_FAILURES}"
[[ -z "${NUM_TASKS}" ]] || require_positive_integer "--num-tasks" "${NUM_TASKS}"

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
readonly SANDBOX_LOG="${OUT_DIR}/sandbox.log"
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
trap termination_requested INT TERM

python -c 'import aiohttp, datasets, pandas' 2>/dev/null || \
	die "MCP-Atlas Python dependencies are unavailable; run: bash setup_mcp_atlas.sh"
wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME}")"
JUDGE_MODEL="${SERVED_MODEL_NAME}"
MCP_LLM_BASE_URL="${MCP_LLM_BASE_URL:-${OPENAI_BASE_URL%/v1}}"
MCP_LLM_BASE_URL="${MCP_LLM_BASE_URL%/}"
readonly MCP_LLM_BASE_URL
readonly SCORE_LABEL="$(sanitize_run_tag "${SERVED_MODEL_NAME}")"
log "MCP-Atlas LLM API origin: ${MCP_LLM_BASE_URL}"

monitor_vllm &
MONITOR_PID=$!

if curl --noproxy "127.0.0.1,localhost" -fsS "${SANDBOX_URL}/enabled-servers" >/dev/null 2>&1; then
	die "Port ${SANDBOX_PORT} already has an MCP sandbox"
else
	SANDBOX_CONTAINER="mcp-atlas-${RUN_TAG}"
	log "Starting MCP sandbox ${SANDBOX_CONTAINER} from ${MCP_ATLAS_IMAGE}"
	PROXY_OPTIONS=()
	for proxy_name in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY \
		http_proxy https_proxy all_proxy no_proxy; do
		[[ -z "${!proxy_name:-}" ]] || PROXY_OPTIONS+=(-e "${proxy_name}")
	done
	docker run -d --name "${SANDBOX_CONTAINER}" \
		-p "${SANDBOX_PORT}:1984" \
		--env-file "${MCP_DIR}/.env" \
		-e "PYTHONPATH=/opt/mcp-sitecustomize" \
		-e "UV_CONSTRAINT=/opt/mcp-sitecustomize/constraints.txt" \
		-e "UV_HTTP_TIMEOUT=${MCP_UV_HTTP_TIMEOUT:-120}" \
		-v "${SCRIPT_DIR}/lib/mcp_sitecustomize:/opt/mcp-sitecustomize:ro" \
		"${PROXY_OPTIONS[@]}" \
		"${MCP_ATLAS_IMAGE}" >/dev/null
	OWN_SANDBOX_IMAGE=true
	wait_for_http "MCP sandbox" "${SANDBOX_URL}/enabled-servers" \
		"${MCP_SANDBOX_WAIT_TIMEOUT:-300}"
fi

if curl --noproxy "127.0.0.1,localhost" -fsS "${HARNESS_URL_VALUE}/health" >/dev/null 2>&1; then
	die "Port ${HARNESS_PORT} already has an agent harness"
else
	log "Starting MCP agent harness on port ${HARNESS_PORT}"
	(
		cd "${MCP_DIR}/services/agent-harness"
		exec setsid env \
			PORT="${HARNESS_PORT}" \
			LLM_BASE_URL="${MCP_LLM_BASE_URL}" \
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
[[ "${SKIP_HEALTH_CHECK}" == false ]] || RUN_OPTIONS+=(--skip-health-check)

log "Running MCP-Atlas with model ${SERVED_MODEL_NAME} and ${WORKERS} workers"
(
	cd "${MCP_DIR}"
	HARNESS_URL="${HARNESS_URL_VALUE}" MCP_SANDBOX_URL="${SANDBOX_URL}" exec python run_eval.py \
			--model "${SERVED_MODEL_NAME}" \
			--output "${OUTPUT_CSV}" \
			--concurrency "${WORKERS}" \
			--timeout "${TASK_TIMEOUT}" \
			--image "${MCP_ATLAS_IMAGE}" \
			"${RUN_OPTIONS[@]}"
) &
ACTIVE_PID=$!
RUN_EXIT=0
wait "${ACTIVE_PID}" || RUN_EXIT=$?
ACTIVE_PID=""
[[ "${RUN_EXIT}" -eq 0 ]] || die "MCP-Atlas generation failed (exit ${RUN_EXIT})"
require_file "${OUTPUT_CSV}"

if [[ "${SKIP_SCORE}" == true ]]; then
	log "Skipping MCP-Atlas LLM-as-judge scoring"
	log "Responses: ${OUTPUT_CSV}"
	log "Log: ${LOG_FILE}"
	exit 0
fi

if [[ ! -f "${GROUNDTRUTH_CSV}" ]]; then
	log "Exporting MCP-Atlas ground truth from Hugging Face"
	python "${SCRIPT_DIR}/lib/benchmark_data.py" atlas-groundtruth \
		--output "${GROUNDTRUTH_CSV}"
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
	EVAL_LLM_BASE_URL="${MCP_LLM_BASE_URL}" \
	EVAL_LLM_TIMEOUT_MS="${EVAL_LLM_TIMEOUT_MS:-${LLM_TIMEOUT_MS:-600000}}" \
	LLM_API_KEY="${VLLM_API_KEY}" \
	LLM_BASE_URL="${MCP_LLM_BASE_URL}" \
		exec python services/scoring/score_claims.py \
			--groundtruth-file "${GROUNDTRUTH_CSV}" \
			--model-file "${OUTPUT_CSV}" \
			--model-name "${SCORE_LABEL}" \
			--evaluator-model "${JUDGE_MODEL}" \
			--api-key "${VLLM_API_KEY}" \
			--base-url "${MCP_LLM_BASE_URL}" \
			--concurrency "${SCORE_WORKERS}" \
			--output-dir "${SCORED_DIR}" \
			"${SCORE_OPTIONS[@]}"
) &
ACTIVE_PID=$!
SCORE_EXIT=0
wait "${ACTIVE_PID}" || SCORE_EXIT=$?
ACTIVE_PID=""
[[ "${SCORE_EXIT}" -eq 0 ]] || die "MCP-Atlas scoring failed (exit ${SCORE_EXIT})"

python "${SCRIPT_DIR}/lib/benchmark_data.py" atlas-report --directory "${SCORED_DIR}"

log "Responses: ${OUTPUT_CSV}"
log "Scored results: ${SCORED_DIR}"
log "Log: ${LOG_FILE}"
