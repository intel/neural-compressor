#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ORIGINAL_ARGS=("$@")
printf -v RUNNER_COMMAND '%q ' bash "${SCRIPT_DIR}/run_humaneval_plus.sh" "${ORIGINAL_ARGS[@]}"
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
	bash run_humaneval_plus.sh [OPTIONS]

Run HumanEval+ (EvalPlus) against an already-running OpenAI-compatible vLLM
server. By default it draws N samples per problem in one EvalPlus invocation,
then reports per-sample and mean pass@1 plus per-task detail.

Options:
	--host HOST          vLLM host (default: 127.0.0.1)
	--port PORT          vLLM port (default: 8000)
	--served-name NAME   Served model ID (default: discover from /v1/models)
	--runs N             Samples per problem (default: 7; pass@1 = mean over N)
	--temperature T      Sampling temperature (default: 0.6; ignored with --greedy)
	--greedy             Greedy decoding: forces runs=1, temperature=0 (standard pass@1)
	--tag TAG            Run tag (default: <model>_<UTC timestamp>)
	--root DIR           Results root (default: <script dir>/results)
	--parallel N         EvalPlus evaluation workers (default: half of CPU cores)
	--mini               Use HumanEval+ Mini tests for faster evaluation
	--expected N         Expected number of tasks (default: 164)
	--evalplus PATH      evalplus.evaluate executable (default: .venv/bin/evalplus.evaluate)
	--python PATH        Python executable (default: .venv/bin/python)
	--wait-timeout N     vLLM readiness timeout in seconds (default: 300)
	-h, --help           Show this help message

Environment:
	OPENAI_API_KEY       API key sent to vLLM (default: EMPTY)
	VLLM_NO_PROXY        Host that bypasses HTTP proxies (default: vLLM host)

Output:
	<root>/<tag>/summary.md, summary.json, state.json
	<root>/<tag>/evalplus.log
	<root>/<tag>/evalplus/humaneval/<model>_openai_temp_<T>.jsonl
	<root>/<tag>/evalplus/humaneval/<model>_openai_temp_<T>_eval_results.json
EOF
}

HOST="127.0.0.1"
PORT=8000
SERVED_NAME=""
RUNS=7
TEMPERATURE=0.6
GREEDY=false
MINI=false
TAG=""
ROOT="${SCRIPT_DIR}/results"
PARALLEL=""
EXPECTED=164
EVALPLUS="${SCRIPT_DIR}/.venv/bin/evalplus.evaluate"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
WAIT_TIMEOUT=300

while [[ $# -gt 0 ]]; do
	case "$1" in
		--host)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			HOST="$2"
			shift 2
			;;
		--port)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			PORT="$2"
			shift 2
			;;
		--served-name)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			SERVED_NAME="$2"
			shift 2
			;;
		--runs)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			RUNS="$2"
			shift 2
			;;
		--temperature)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			TEMPERATURE="$2"
			shift 2
			;;
		--greedy)
			GREEDY=true
			shift
			;;
		--tag)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			TAG="$2"
			shift 2
			;;
		--root)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			ROOT="$2"
			shift 2
			;;
		--parallel)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			PARALLEL="$2"
			shift 2
			;;
		--mini)
			MINI=true
			shift
			;;
		--expected)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			EXPECTED="$2"
			shift 2
			;;
		--evalplus)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			EVALPLUS="$2"
			shift 2
			;;
		--python)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			PYTHON="$2"
			shift 2
			;;
		--wait-timeout)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			WAIT_TIMEOUT="$2"
			shift 2
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

[[ -x "${EVALPLUS}" ]] || die "evalplus.evaluate not found or not executable: ${EVALPLUS}"
[[ -x "${PYTHON}" ]] || die "Python not found or not executable: ${PYTHON}"
require_positive_integer "--port" "${PORT}"
require_positive_integer "--runs" "${RUNS}"
require_positive_integer "--expected" "${EXPECTED}"
require_positive_integer "--wait-timeout" "${WAIT_TIMEOUT}"
[[ -z "${PARALLEL}" ]] || require_positive_integer "--parallel" "${PARALLEL}"

if [[ "${GREEDY}" == true ]]; then
	[[ "${RUNS}" -eq 1 ]] || warn "--greedy forces --runs 1 (was ${RUNS})"
	RUNS=1
	TEMPERATURE=0.0
else
	TEMPERATURE="$("${PYTHON}" -c 'import sys; value = float(sys.argv[1]); assert value > 0; print(value)' "${TEMPERATURE}")" || \
		die "--temperature must be positive for sampling; use --greedy for temperature 0"
fi

ORIGIN="http://${HOST}:${PORT}"
BASE_URL="${ORIGIN}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
AUTH_ARGS=()
[[ "${OPENAI_API_KEY}" == "EMPTY" ]] || AUTH_ARGS=(-H "Authorization: Bearer ${OPENAI_API_KEY}")

bypass_host="${VLLM_NO_PROXY:-${HOST}}"
current_no_proxy="${NO_PROXY:-${no_proxy:-}}"
if [[ ",${current_no_proxy}," != *",${bypass_host},"* ]]; then
	current_no_proxy="${current_no_proxy:+${current_no_proxy},}${bypass_host}"
fi
export NO_PROXY="${current_no_proxy}"
export no_proxy="${current_no_proxy}"

vllm_curl() {
	curl --noproxy "${bypass_host}" -fsS --connect-timeout 5 --max-time 30 "${AUTH_ARGS[@]}" "$@"
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
	TAG="$(printf '%s_%s' "${SERVED_NAME//\//--}" "$(date -u +%Y%m%dT%H%M%SZ)")"
fi
TAG="${TAG//[^[:alnum:]._-]/_}"
[[ -n "${TAG}" ]] || die "Run tag is empty after sanitization"

OUT="${ROOT}/${TAG}"
MODEL_ID="$(printf '%s' "${SERVED_NAME}" | sed -E 's#^[./]+##; s#[./]+$##')"
MODEL_ID="${MODEL_ID//\//--}"
SAMPLES="${OUT}/evalplus/humaneval/${MODEL_ID}_openai_temp_${TEMPERATURE}.jsonl"
LOG="${OUT}/evalplus.log"
RESULT="${SAMPLES%.jsonl}_eval_results.json"
STATE="${OUT}/state.json"
SUMMARY_JSON="${OUT}/summary.json"
SUMMARY_MD="${OUT}/summary.md"

mkdir -p "${OUT}"
"${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" start \
	--state "${STATE}" --name "${TAG}" --pid $$ \
	--model "${SERVED_NAME}" --host "${HOST}" --port "${PORT}" \
	--runs "${RUNS}" --temperature "${TEMPERATURE}" --expected "${EXPECTED}" \
	--samples "${SAMPLES}" --result "${RESULT}" --log "${LOG}" \
	--command "${RUNNER_COMMAND}"

EVAL_PID=""
FINALIZED=false
INTERRUPTED=false

cleanup() {
	local exit_code=$?
	if [[ -n "${EVAL_PID}" ]] && kill -0 "${EVAL_PID}" 2>/dev/null; then
		pkill -TERM -P "${EVAL_PID}" 2>/dev/null || true
		kill -TERM "${EVAL_PID}" 2>/dev/null || true
		wait "${EVAL_PID}" 2>/dev/null || true
	fi
	if [[ "${FINALIZED}" == false && -f "${STATE}" ]]; then
		local message="run failed"
		[[ "${INTERRUPTED}" == true ]] && message="interrupted"
		"${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" fail \
			--state "${STATE}" --exit-code "${exit_code}" --message "${message}" \
			>/dev/null 2>&1 || true
	fi
}

termination_requested() {
	INTERRUPTED=true
	warn "HumanEval+ run was interrupted"
	exit 1
}

trap cleanup EXIT
trap termination_requested INT TERM

eval_args=(
	humaneval
	--model "${SERVED_NAME}"
	--backend openai
	--base-url "${BASE_URL}"
	--root "${OUT}/evalplus"
)
if [[ "${GREEDY}" == true ]]; then
	eval_args+=(--greedy)
else
	eval_args+=(--temperature "${TEMPERATURE}" --n-samples "${RUNS}")
fi
[[ "${MINI}" == false ]] || eval_args+=(--mini)
[[ -z "${PARALLEL}" ]] || eval_args+=(--parallel "${PARALLEL}")

log "Running HumanEval+ with model ${SERVED_NAME} (samples=${RUNS}, temperature=${TEMPERATURE})"
log "EvalPlus log: ${LOG}"
"${EVALPLUS}" "${eval_args[@]}" >"${LOG}" 2>&1 &
EVAL_PID=$!

while kill -0 "${EVAL_PID}" 2>/dev/null; do
	generated="$("${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" count --samples "${SAMPLES}" 2>/dev/null || echo 0)"
	message="generating"
	if grep -qE "Reading samples|Computing expected output" "${LOG}" 2>/dev/null; then
		message="evaluating"
	fi
	"${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" progress \
		--state "${STATE}" --generated "${generated}" --message "${message}" \
		>/dev/null 2>&1 || true
	sleep 10
done

EVAL_EXIT=0
wait "${EVAL_PID}" || EVAL_EXIT=$?
EVAL_PID=""

if [[ "${EVAL_EXIT}" -ne 0 || ! -f "${RESULT}" ]]; then
	warn "EvalPlus failed (exit ${EVAL_EXIT}); last log lines:"
	tail -20 "${LOG}" >&2 || true
	"${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" fail \
		--state "${STATE}" --exit-code "${EVAL_EXIT:-1}" --message "evalplus failed"
	FINALIZED=true
	exit 1
fi

summarize_rc=0
"${PYTHON}" "${SCRIPT_DIR}/humaneval_data.py" summarize \
	--state "${STATE}" --result "${RESULT}" --log "${LOG}" \
	--output "${SUMMARY_JSON}" --markdown "${SUMMARY_MD}" \
	--runs "${RUNS}" --expected "${EXPECTED}" || summarize_rc=$?
FINALIZED=true

log "Summary: ${SUMMARY_MD}"
log "Summary JSON: ${SUMMARY_JSON}"
log "Samples: ${SAMPLES}"
log "EvalPlus log: ${LOG}"
exit "${summarize_rc}"
