#!/usr/bin/env bash
# Run IFBench N independent times against an OpenAI-compatible endpoint and
# write a per-run suite (kind=ifbench-suite).
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ORIGINAL_ARGS=("$@")
printf -v RUNNER_COMMAND '%q ' bash "${SCRIPT_DIR}/run_ifbench_suite.sh" "${ORIGINAL_ARGS[@]}"
readonly RUNNER_COMMAND="${RUNNER_COMMAND% }"

log() { printf '[INFO] %s\n' "$*" >&2; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
	cat <<'EOF'
Usage:
	bash run_ifbench_suite.sh [OPTIONS]

Run the IFBench instruction-following benchmark N independent times against
an OpenAI-compatible vLLM endpoint.

Options:
	--host HOST            vLLM host (default: 127.0.0.1)
	--port PORT            vLLM port (default: 8000)
	--served-name NAME     Served model id (default: discover from /v1/models)
	--runs N               Independent rounds (default: 1)
	--workers N            Concurrent generation requests (default: 8)
	--repeats N            Consensus repeats inside one round (default: 1)
	--temperature T        Sampling temperature (default: 1.0)
	--top-p P              Nucleus sampling (default: 0.95)
	--max-tokens N         Max tokens; <=0 omits max_tokens (default: 0)
	--limit N              Only evaluate the first N prompts (smoke test)
	--no-thinking          Disable chat_template_kwargs.enable_thinking
	--seed N               Base seed; round i starts at seed+(i-1)*repeats (default: 42)
	--request-timeout N    Per-request timeout seconds (default: 3600)
	--max-retries N        Retries per request (default: 10)
	--tag TAG              Suite tag (default: <model>_ifbench_<UTC>)
	--root DIR             Results root (default: <script dir>/results)
	--workdir DIR          IFBench clone/venv root (default: ~/ifbench-run)
	--skip-setup           Do not clone/install; use the existing workdir
	--python PATH          Python used to create the venv (default: python3)
	--dry-run              Print the resolved configuration and exit
	-h, --help             Show this help
EOF
}

HOST="127.0.0.1"
PORT=8000
SERVED_NAME=""
RUNS=1
WORKERS=8
REPEATS=1
TEMPERATURE="1.0"
TOP_P="0.95"
MAX_TOKENS=0
LIMIT=""
ENABLE_THINKING=true
SEED=42
REQUEST_TIMEOUT=3600
MAX_RETRIES=10
TAG=""
ROOT="${SCRIPT_DIR}/results"
WORKDIR="${IFBENCH_WORKDIR:-$HOME/ifbench-run}"
SKIP_SETUP=false
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
	case "$1" in
		--host) HOST="$2"; shift 2 ;;
		--port) PORT="$2"; shift 2 ;;
		--served-name) SERVED_NAME="$2"; shift 2 ;;
		--runs) RUNS="$2"; shift 2 ;;
		--workers) WORKERS="$2"; shift 2 ;;
		--repeats) REPEATS="$2"; shift 2 ;;
		--temperature) TEMPERATURE="$2"; shift 2 ;;
		--top-p) TOP_P="$2"; shift 2 ;;
		--max-tokens) MAX_TOKENS="$2"; shift 2 ;;
		--limit) LIMIT="$2"; shift 2 ;;
		--no-thinking) ENABLE_THINKING=false; shift ;;
		--seed) SEED="$2"; shift 2 ;;
		--request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
		--max-retries) MAX_RETRIES="$2"; shift 2 ;;
		--tag) TAG="$2"; shift 2 ;;
		--root) ROOT="$2"; shift 2 ;;
		--workdir) WORKDIR="$2"; shift 2 ;;
		--skip-setup) SKIP_SETUP=true; shift ;;
		--python) PYTHON_BIN="$2"; shift 2 ;;
		--dry-run) DRY_RUN=true; shift ;;
		-h|--help) usage; exit 0 ;;
		*) die "Unknown argument: $1" ;;
	esac
done

[[ "${RUNS}" =~ ^[1-9][0-9]*$ ]] || die "--runs must be a positive integer"
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || die "--workers must be a positive integer"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "--repeats must be a positive integer"

abspath() {
	"${PYTHON_BIN}" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}
ROOT="$(abspath "${ROOT}")"
WORKDIR="$(abspath "${WORKDIR}")"

ORIGIN="http://${HOST}:${PORT}"
API_BASE="${ORIGIN}/v1"
current_no_proxy="${NO_PROXY:-${no_proxy:-}}"
if [[ ",${current_no_proxy}," != *",${HOST},"* ]]; then
	current_no_proxy="${current_no_proxy:+${current_no_proxy},}${HOST}"
fi
export NO_PROXY="${current_no_proxy}"
export no_proxy="${current_no_proxy}"
API_KEY="${IFBENCH_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
AUTH_ARGS=()
[[ "${API_KEY}" == "EMPTY" ]] || AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
CLONE_DIR="${WORKDIR}/IFBench"
VENV_DIR="${WORKDIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
NO_PROXY_HOST="${VLLM_NO_PROXY:-${HOST}}"

vllm_curl() {
	curl --noproxy "${NO_PROXY_HOST}" -fsS --connect-timeout 5 --max-time 30 "${AUTH_ARGS[@]}" "$@"
}

metric_sum() {
	local name="$1"
	vllm_curl "${ORIGIN}/metrics" 2>/dev/null | \
		awk -v key="vllm:${name}" 'index($0,key)==1 {sum += $NF} END {if (sum == "") exit 1; printf "%.0f\n", sum}'
}

wait_for_vllm() {
	local deadline=$((SECONDS + ${VLLM_WAIT_TIMEOUT:-300}))
	log "Waiting up to ${VLLM_WAIT_TIMEOUT:-300}s for ${ORIGIN}/health"
	while ((SECONDS < deadline)); do
		vllm_curl "${ORIGIN}/health" >/dev/null 2>&1 && { log "vLLM is ready"; return 0; }
		sleep 2
	done
	die "vLLM did not become ready: ${ORIGIN}"
}

discover_model() {
	local response model
	response="$(vllm_curl "${API_BASE}/models")" || die "Unable to query ${API_BASE}/models"
	if [[ -n "${SERVED_NAME}" ]]; then
		printf '%s' "${response}" | "${PYTHON_BIN}" -c '
import json, sys
model = sys.argv[1]
data = json.load(sys.stdin).get("data") or []
ids = [item.get("id") for item in data]
if model not in ids:
    raise SystemExit(f"requested model {model!r} not served; available: {ids}")
print(model)
' "${SERVED_NAME}" || die "Requested model is not served"
	else
		printf '%s' "${response}" | "${PYTHON_BIN}" -c '
import json, sys
data = json.load(sys.stdin).get("data") or []
if not data:
    raise SystemExit("no models served")
print(data[0]["id"])
'
	fi
}

ensure_setup() {
	mkdir -p "${WORKDIR}"
	if [[ ! -d "${CLONE_DIR}/.git" ]]; then
		log "Cloning IFBench into ${CLONE_DIR}"
		git clone --depth 1 --branch "${REPO_REF:-main}" \
			"${REPO_URL:-https://github.com/allenai/IFBench.git}" "${CLONE_DIR}"
	fi
	cp "${SCRIPT_DIR}/evaluate_model.py" "${CLONE_DIR}/evaluate_model.py"
	cp "${SCRIPT_DIR}/configs/config.py" "${CLONE_DIR}/config.py"
	if [[ ! -x "${PYTHON}" ]]; then
		log "Creating IFBench virtualenv at ${VENV_DIR}"
		"${PYTHON_BIN}" -m venv "${VENV_DIR}"
		"${PYTHON}" -m pip install --upgrade pip >/dev/null
		"${PYTHON}" -m pip install -q -r "${CLONE_DIR}/requirements.txt" \
			httpx tqdm pydantic-settings
		"${PYTHON}" - <<'PY' || true
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
PY
	fi
	[[ -f "${CLONE_DIR}/data/IFBench_test.jsonl" ]] || die "IFBench test file not found in ${CLONE_DIR}/data"
	[[ -f "${CLONE_DIR}/evaluate_model.py" ]] || die "IFBench evaluate_model.py missing"
	[[ -x "${PYTHON}" ]] || die "IFBench venv missing: ${PYTHON}"
	if ! "${PYTHON}" -c 'import httpx, tqdm, pydantic_settings' >/dev/null 2>&1; then
		log "Installing missing IFBench Python dependencies"
		"${PYTHON}" -m pip install -q -r "${CLONE_DIR}/requirements.txt" httpx tqdm pydantic-settings
	fi
}

wait_for_vllm
SERVED_NAME="$(discover_model)"
[[ -n "${SERVED_NAME}" ]] || die "Unable to discover served model"

if [[ -z "${TAG}" ]]; then
	TAG="$(printf '%s_ifbench_%s' "${SERVED_NAME//\//-}" "$(date -u +%Y%m%dT%H%M%SZ)")"
fi
TAG="${TAG//[^[:alnum:]._-]/_}"
[[ -n "${TAG}" ]] || die "Suite tag is empty"

SUITE_DIR="${ROOT}/${TAG}_suite"
RUNS_DIR="${SUITE_DIR}/runs"
STATE="${SUITE_DIR}/state.json"
SUMMARY_JSON="${SUITE_DIR}/summary.json"
SUMMARY_MD="${SUITE_DIR}/summary.md"

log "Model: ${SERVED_NAME}"
log "API: ${API_BASE}"
log "Tag: ${TAG}"
log "Suite: ${SUITE_DIR}"

if [[ "${DRY_RUN}" == true ]]; then
	NUM_PROMPTS=300
	[[ -f "${CLONE_DIR}/data/IFBench_test.jsonl" ]] && NUM_PROMPTS="$("${PYTHON_BIN}" -c 'import sys; print(sum(1 for _ in open(sys.argv[1])))' "${CLONE_DIR}/data/IFBench_test.jsonl")"
	printf 'model=%s\napi_base=%s\nprompts=%s\nruns=%s\nrepeats=%s\ntag=%s\nroot=%s\nclone=%s\nvenv=%s\n' \
		"${SERVED_NAME}" "${API_BASE}" "${NUM_PROMPTS}" "${RUNS}" "${REPEATS}" "${TAG}" "${ROOT}" "${CLONE_DIR}" "${VENV_DIR}"
	exit 0
fi

if [[ "${SKIP_SETUP}" == false ]]; then
	ensure_setup
else
	[[ -x "${PYTHON}" ]] || die "IFBench venv missing: ${PYTHON}"
fi
NUM_PROMPTS="$("${PYTHON_BIN}" -c 'import sys; print(sum(1 for _ in open(sys.argv[1])))' "${CLONE_DIR}/data/IFBench_test.jsonl")"
[[ -z "${LIMIT}" ]] || NUM_PROMPTS="${LIMIT}"
log "IFBench prompts: ${NUM_PROMPTS}; runs=${RUNS}; repeats/run=${REPEATS}"

mkdir -p "${SUITE_DIR}" "${RUNS_DIR}"
thinking_args=()
[[ "${ENABLE_THINKING}" == true ]] && thinking_args+=(--enable-thinking)
"${PYTHON_BIN}" "${SCRIPT_DIR}/ifbench_data.py" suite-start \
	--state "${STATE}" --name "${TAG}" --runs "${RUNS}" \
	--model "${SERVED_NAME}" --host "${HOST}" --port "${PORT}" --api-url "${API_BASE}" \
	--temperature "${TEMPERATURE}" --top-p "${TOP_P}" --max-tokens "${MAX_TOKENS}" \
	--workers "${WORKERS}" --num-prompts "${NUM_PROMPTS}" --pid $$ \
	--env ifbench --python "${PYTHON}" --command "${RUNNER_COMMAND}" "${thinking_args[@]}"

CURRENT_PID=""
FINALIZED=false
cleanup() {
	local rc=$?
	if [[ -n "${CURRENT_PID}" ]] && kill -0 "${CURRENT_PID}" 2>/dev/null; then
		kill -TERM "${CURRENT_PID}" 2>/dev/null || true
		wait "${CURRENT_PID}" 2>/dev/null || true
	fi
	if [[ "${FINALIZED}" == false && -f "${STATE}" ]]; then
		"${PYTHON_BIN}" "${SCRIPT_DIR}/ifbench_data.py" suite-fail \
			--state "${STATE}" --status interrupted --exit-code "${rc}" \
			--message "interrupted" >/dev/null 2>&1 || true
	fi
}
trap cleanup EXIT
trap 'exit 1' INT TERM

for ((index = 1; index <= RUNS; index++)); do
	printf -v run_tag '%s_run%02d' "${TAG}" "${index}"
	run_dir="${RUNS_DIR}/${run_tag}"
	run_log="${run_dir}/eval.log"
	mkdir -p "${run_dir}"
	started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	seed=$((SEED + (index - 1) * REPEATS))
	args=(
		"${PYTHON}" evaluate_model.py
		--api-base "${API_BASE}" --model "${SERVED_NAME}" --api-key "${API_KEY}"
		--input-file data/IFBench_test.jsonl --output-dir "${run_dir}/eval"
		--temperature "${TEMPERATURE}" --top-p "${TOP_P}" --max-tokens "${MAX_TOKENS}"
		--workers "${WORKERS}" --request-timeout "${REQUEST_TIMEOUT}" --max-retries "${MAX_RETRIES}"
		--num-repeats "${REPEATS}" --seed "${seed}" --save-responses
	)
	[[ -z "${LIMIT}" ]] || args+=(--limit "${LIMIT}")
	[[ "${ENABLE_THINKING}" == true ]] && args+=(--enable-thinking) || args+=(--no-thinking)
	recorded_args=()
	skip_next=false
	for arg in "${args[@]}"; do
		if [[ "${skip_next}" == true ]]; then
			recorded_args+=("***")
			skip_next=false
			continue
		fi
		if [[ "${arg}" == "--api-key" ]]; then
			recorded_args+=("${arg}")
			skip_next=true
			continue
		fi
		recorded_args+=("${arg}")
	done
	printf -v run_command '%q ' "${recorded_args[@]}"
	run_command="${run_command% }"
	"${PYTHON_BIN}" "${SCRIPT_DIR}/ifbench_data.py" suite-run-start \
		--state "${STATE}" --index "${index}" --tag "${run_tag}" \
		--started-at "${started_at}" --log "${run_log}" --command "${run_command}"

	tokens_before="$(metric_sum generation_tokens_total 2>/dev/null || true)"
	prompt_before="$(metric_sum prompt_tokens_total 2>/dev/null || true)"
	log "[suite] Starting run ${index}/${RUNS}: ${run_tag}"
	(
		cd "${CLONE_DIR}"
		exec "${args[@]}"
	) >"${run_log}" 2>&1 &
	CURRENT_PID=$!
	rc=0
	wait "${CURRENT_PID}" || rc=$?
	CURRENT_PID=""
	finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	tokens_after="$(metric_sum generation_tokens_total 2>/dev/null || true)"
	prompt_after="$(metric_sum prompt_tokens_total 2>/dev/null || true)"
	token_args=()
	if [[ "${tokens_before}" =~ ^[0-9]+$ && "${tokens_after}" =~ ^[0-9]+$ ]]; then
		token_args+=(--tokens-generated "$((tokens_after - tokens_before))")
	fi
	if [[ "${prompt_before}" =~ ^[0-9]+$ && "${prompt_after}" =~ ^[0-9]+$ ]]; then
		token_args+=(--prompt-tokens "$((prompt_after - prompt_before))")
	fi
	safe_model="${SERVED_NAME//\//-}"
	results="${run_dir}/eval/${safe_model}-results.json"
	"${PYTHON_BIN}" "${SCRIPT_DIR}/ifbench_data.py" suite-run-finish \
		--state "${STATE}" --index "${index}" --exit-code "${rc}" \
		--results "${results}" --log "${run_log}" --finished-at "${finished_at}" \
		"${token_args[@]}"
	if ((rc == 0)); then
		log "[suite] Run ${index}/${RUNS} finished"
	else
		warn "[suite] Run ${index}/${RUNS} failed with status ${rc}; continuing"
	fi
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/ifbench_data.py" suite-aggregate \
	--state "${STATE}" --output "${SUMMARY_JSON}" --markdown "${SUMMARY_MD}"
FINALIZED=true
log "Summary: ${SUMMARY_MD}"
cat "${SUMMARY_MD}"
status="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","failed"))' "${STATE}")"
[[ "${status}" == "completed" ]]
