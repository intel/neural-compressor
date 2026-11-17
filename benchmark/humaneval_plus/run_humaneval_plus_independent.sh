#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ORIGINAL_ARGS=("$@")
printf -v RUNNER_COMMAND '%q ' bash "${SCRIPT_DIR}/run_humaneval_plus_independent.sh" "${ORIGINAL_ARGS[@]}"
readonly RUNNER_COMMAND="${RUNNER_COMMAND% }"

log() { printf '[INFO] %s\n' "$*" >&2; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

require_positive_integer() {
	local name="$1"
	local value="$2"
	[[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

usage() {
	cat <<'EOF'
Usage:
	bash run_humaneval_plus_independent.sh [OPTIONS]

Run HumanEval+ N independent times (one EvalPlus invocation per run, one
sample per problem) against an OpenAI-compatible vLLM server. After all runs,
writes per-run details plus mean/std and per-task pass rates.

Options:
	--host HOST          vLLM host (default: 127.0.0.1)
	--port PORT          vLLM port (default: 8000)
	--served-name NAME   Served model ID (default: discover from /v1/models)
	--runs N             Independent runs (default: 7)
	--temperature T      Sampling temperature (default: 0.6)
	--tag TAG            Suite tag (default: <model>_humaneval_indep_<UTC>)
	--root DIR           Results root (default: <script dir>/results)
	--parallel N         EvalPlus evaluation workers per run (default: half CPU)
	--expected N         Tasks per run (default: 164)
	--mini               Use HumanEval+ Mini tests per run
	--evalplus PATH      evalplus.evaluate executable
	--python PATH        Python executable
	--wait-timeout N     vLLM readiness timeout in seconds (default: 300)
	-h, --help           Show this help message

Layout:
	<root>/<tag>_suite/
	├── state.json
	├── summary.json
	├── summary.md
	├── run01.log ... runNN.log
	└── runs/
	    ├── <tag>_run01/{state.json,summary.json,evalplus/...}
	    └── ...
EOF
}

HOST="127.0.0.1"
PORT=8000
SERVED_NAME=""
RUNS=7
TEMPERATURE=0.6
TAG=""
ROOT="${SCRIPT_DIR}/results"
PARALLEL=""
EXPECTED=164
MINI=false
EVALPLUS="${SCRIPT_DIR}/.venv/bin/evalplus.evaluate"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
WAIT_TIMEOUT=300

while [[ $# -gt 0 ]]; do
	case "$1" in
		--host) HOST="$2"; shift 2 ;;
		--port) PORT="$2"; shift 2 ;;
		--served-name) SERVED_NAME="$2"; shift 2 ;;
		--runs) RUNS="$2"; shift 2 ;;
		--temperature) TEMPERATURE="$2"; shift 2 ;;
		--tag) TAG="$2"; shift 2 ;;
		--root) ROOT="$2"; shift 2 ;;
		--parallel) PARALLEL="$2"; shift 2 ;;
		--expected) EXPECTED="$2"; shift 2 ;;
		--mini) MINI=true; shift ;;
		--evalplus) EVALPLUS="$2"; shift 2 ;;
		--python) PYTHON="$2"; shift 2 ;;
		--wait-timeout) WAIT_TIMEOUT="$2"; shift 2 ;;
		-h | --help) usage; exit 0 ;;
		*) die "Unknown argument: $1" ;;
	esac
done

[[ -x "${EVALPLUS}" ]] || die "evalplus.evaluate not found: ${EVALPLUS}"
[[ -x "${PYTHON}" ]] || die "Python not found: ${PYTHON}"
require_positive_integer "--port" "${PORT}"
require_positive_integer "--runs" "${RUNS}"
require_positive_integer "--expected" "${EXPECTED}"
require_positive_integer "--wait-timeout" "${WAIT_TIMEOUT}"
[[ -z "${PARALLEL}" ]] || require_positive_integer "--parallel" "${PARALLEL}"
TEMPERATURE="$("${PYTHON}" -c 'import sys; value = float(sys.argv[1]); assert value > 0; print(value)' "${TEMPERATURE}")" || \
	die "--temperature must be positive for independent sampling"

ORIGIN="http://${HOST}:${PORT}"
BASE_URL="${ORIGIN}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

bypass_host="${VLLM_NO_PROXY:-${HOST}}"
current_no_proxy="${NO_PROXY:-${no_proxy:-}}"
if [[ ",${current_no_proxy}," != *",${bypass_host},"* ]]; then
	current_no_proxy="${current_no_proxy:+${current_no_proxy},}${bypass_host}"
fi
export NO_PROXY="${current_no_proxy}"
export no_proxy="${current_no_proxy}"

vllm_curl() {
	curl --noproxy "${bypass_host}" -fsS --connect-timeout 5 --max-time 30 "$@"
}

vllm_generation_tokens() {
	vllm_curl "${ORIGIN}/metrics" 2>/dev/null | \
		awk '/^vllm:generation_tokens_total/ {sum += $NF} END {if (sum == "") exit 1; printf "%.0f\n", sum}'
}

wait_for_vllm() {
	local deadline=$((SECONDS + WAIT_TIMEOUT))
	log "Waiting up to ${WAIT_TIMEOUT}s for ${ORIGIN}/health"
	while ((SECONDS < deadline)); do
		if vllm_curl "${ORIGIN}/health" >/dev/null 2>&1; then
			log "vLLM is ready"
			return 0
		fi
		sleep 2
	done
	die "vLLM did not become ready within ${WAIT_TIMEOUT}s: ${ORIGIN}"
}

discover_model() {
	local response models
	response="$(vllm_curl "${BASE_URL}/models")" || die "Unable to query ${BASE_URL}/models"
	models="$("${PYTHON}" -c '
import json, sys
payload = json.load(sys.stdin)
for item in payload.get("data", []):
    model = item.get("id", "")
    if model:
        print(model)
' <<<"${response}")" || die "Invalid response from ${BASE_URL}/models"
	[[ -n "${models}" ]] || die "vLLM returned no served models"
	if [[ -n "${SERVED_NAME}" ]]; then
		grep -Fxq -- "${SERVED_NAME}" <<<"${models}" || \
			die "Requested model '${SERVED_NAME}' is not served. Available: $(tr '\n' ' ' <<<"${models}")"
		printf '%s\n' "${SERVED_NAME}"
	else
		printf '%s\n' "${models%%$'\n'*}"
	fi
}

wait_for_vllm
SERVED_NAME="$(discover_model)"

if [[ -z "${TAG}" ]]; then
	TAG="$(printf '%s_%s' "${SERVED_NAME//\//--}" "humaneval_indep_$(date -u +%Y%m%dT%H%M%SZ)")"
fi
TAG="${TAG//[^[:alnum:]._-]/_}"
[[ -n "${TAG}" ]] || die "Suite tag is empty after sanitization"

SUITE_DIR="${ROOT}/${TAG}_suite"
RUNS_DIR="${SUITE_DIR}/runs"
STATE="${SUITE_DIR}/state.json"
SUMMARY_JSON="${SUITE_DIR}/summary.json"
SUMMARY_MD="${SUITE_DIR}/summary.md"

mkdir -p "${SUITE_DIR}" "${RUNS_DIR}"
"${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" suite-start \
	--state "${STATE}" --name "${TAG}" --pid $$ \
	--model "${SERVED_NAME}" --host "${HOST}" --port "${PORT}" \
	--runs "${RUNS}" --temperature "${TEMPERATURE}" --expected "${EXPECTED}" \
	--runs-dir "${RUNS_DIR}" --command "${RUNNER_COMMAND}"

CURRENT_PID=""
FINALIZED=false
INTERRUPTED=false

cleanup() {
	local exit_code=$?
	if [[ -n "${CURRENT_PID}" ]] && kill -0 "${CURRENT_PID}" 2>/dev/null; then
		kill -TERM "${CURRENT_PID}" 2>/dev/null || true
		wait "${CURRENT_PID}" 2>/dev/null || true
	fi
	if [[ "${FINALIZED}" == false && -f "${STATE}" ]]; then
		local status=failed
		[[ "${INTERRUPTED}" == true ]] && status=interrupted
		"${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" suite-fail \
			--state "${STATE}" --status "${status}" --exit-code "${exit_code}" \
			--message "suite ${status}" >/dev/null 2>&1 || true
	fi
}

termination_requested() {
	INTERRUPTED=true
	warn "HumanEval+ independent suite was interrupted"
	exit 1
}

trap cleanup EXIT
trap termination_requested INT TERM

for ((run_index = 1; run_index <= RUNS; run_index++)); do
	printf -v run_tag '%s_run%02d' "${TAG}" "${run_index}"
	printf -v run_log '%s/run%02d.log' "${SUITE_DIR}" "${run_index}"
	run_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	log "[suite] Starting run ${run_index}/${RUNS}: ${run_tag}"
	tokens_before="$(vllm_generation_tokens 2>/dev/null || true)"
	inner_args=(
		--host "${HOST}" --port "${PORT}" --served-name "${SERVED_NAME}"
		--tag "${run_tag}" --root "${RUNS_DIR}"
		--runs 1 --temperature "${TEMPERATURE}" --expected "${EXPECTED}"
		--evalplus "${EVALPLUS}" --python "${PYTHON}" --wait-timeout "${WAIT_TIMEOUT}"
	)
	[[ -z "${PARALLEL}" ]] || inner_args+=(--parallel "${PARALLEL}")
	[[ "${MINI}" == false ]] || inner_args+=(--mini)
	printf -v inner_command '%q ' bash "${SCRIPT_DIR}/run_humaneval_plus.sh" "${inner_args[@]}"
	inner_command="${inner_command% }"
	"${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" suite-run-start \
		--state "${STATE}" --index "${run_index}" --tag "${run_tag}" \
		--started-at "${run_started}" --log "${run_log}" --command "${inner_command}"

	exit_code=0
	bash "${SCRIPT_DIR}/run_humaneval_plus.sh" "${inner_args[@]}" >"${run_log}" 2>&1 &
	CURRENT_PID=$!
	wait "${CURRENT_PID}" || exit_code=$?
	CURRENT_PID=""
	run_finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	tokens_after="$(vllm_generation_tokens 2>/dev/null || true)"
	token_args=()
	if [[ "${tokens_before}" =~ ^[0-9]+$ && "${tokens_after}" =~ ^[0-9]+$ ]] && \
		((tokens_after >= tokens_before)); then
		token_args+=(--tokens-generated "$((tokens_after - tokens_before))")
	else
		warn "Could not read vllm:generation_tokens_total; token usage not recorded for run ${run_index}"
	fi
	"${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" suite-run-finish \
		--state "${STATE}" --index "${run_index}" --exit-code "${exit_code}" \
		--summary "${RUNS_DIR}/${run_tag}/summary.json" --log "${run_log}" \
		--finished-at "${run_finished}" "${token_args[@]}"
	if ((exit_code == 0)); then
		log "[suite] Run ${run_index}/${RUNS} finished"
	else
		warn "[suite] Run ${run_index}/${RUNS} failed with status ${exit_code}; continuing"
	fi
done

"${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" suite-aggregate \
	--state "${STATE}" --output "${SUMMARY_JSON}" --markdown "${SUMMARY_MD}" \
	--expected "${EXPECTED}"
FINALIZED=true

log "Summary: ${SUMMARY_MD}"
log "Summary JSON: ${SUMMARY_JSON}"
log "Run logs: ${SUITE_DIR}/runNN.log"
cat "${SUMMARY_MD}"

status="$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["status"])' "${STATE}")"
[[ "${status}" == completed ]]
