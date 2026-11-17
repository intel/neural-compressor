#!/usr/bin/env bash
# Run the EvalScope AIME 2026 + MATH-500 + GPQA Diamond suite N times.
#
# MODEL and reasoning_effort are discovered from the user-provided
# OpenAI-compatible endpoint. Override either with flags or environment.
set -uo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ORIGINAL_ARGS=("$@")
printf -v RUNNER_COMMAND '%q ' bash "${SCRIPT_DIR}/run_evalscope_7r.sh" "${ORIGINAL_ARGS[@]}"
readonly RUNNER_COMMAND="${RUNNER_COMMAND% }"

log() { printf '[INFO] %s\n' "$*" >&2; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
	cat <<'EOF'
Usage:
	bash run_evalscope_7r.sh [OPTIONS]

Run EvalScope's AIME 2026, MATH-500 and GPQA Diamond tasks N times against an
OpenAI-compatible vLLM endpoint, then write per-run and aggregate scores.

MODEL is discovered from /v1/models unless --served-name/MODEL is set.
reasoning_effort is discovered from model metadata or the model id/root path;
it falls back to xhigh.

Options:
	--host HOST             vLLM host (default: 127.0.0.1)
	--port PORT             vLLM port (default: 8000)
	--api-url URL           Full API base, e.g. http://host:8000/v1
	--served-name NAME      Model id (default: discover from /v1/models)
	--api-key KEY           API key (default: EMPTY)
	--reasoning-effort E    reasoning_effort (default: discover, fallback xhigh)
	--datasets LIST         Space-separated tasks (default: aime26 math_500 gpqa_diamond)
	--runs N                Independent runs (default: 7)
	--temperature T         Sampling temperature (default: 1.0)
	--top-p P               top_p (default: 0.95)
	--max-tokens N          max_tokens (default: 131072)
	--timeout N             Per-request timeout seconds (default: 18000)
	--eval-batch-size N     EvalScope batch size (default: 16)
	--tag TAG               Suite tag (default: <model>_evalscope_7r_<UTC>)
	--root DIR              Results root (default: <script dir>/results)
	--evalscope PATH        evalscope executable (default: EVALSCOPE_BIN or PATH)
	--python PATH           Python for the state helper (default: python3)
	--hf-home DIR           HF_HOME for dataset downloads
	--dry-run               Print the resolved configuration and exit
	-h, --help              Show this help
EOF
}

HOST="127.0.0.1"
PORT=8000
API_URL=""
MODEL=""
API_KEY="${EVALSCOPE_API_KEY:-EMPTY}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
DATASETS="aime26 math_500 gpqa_diamond"
RUNS=7
TEMPERATURE=1.0
TOP_P=0.95
MAX_TOKENS=131072
TIMEOUT=18000
BATCH=16
TAG=""
ROOT="${SCRIPT_DIR}/results"
EVALSCOPE="${EVALSCOPE_BIN:-evalscope}"
PYTHON="${PYTHON:-python3}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		--host) HOST="$2"; shift 2 ;;
		--port) PORT="$2"; shift 2 ;;
		--api-url) API_URL="$2"; shift 2 ;;
		--served-name) MODEL="$2"; shift 2 ;;
		--api-key) API_KEY="$2"; shift 2 ;;
		--reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
		--datasets) DATASETS="$2"; shift 2 ;;
		--runs) RUNS="$2"; shift 2 ;;
		--temperature) TEMPERATURE="$2"; shift 2 ;;
		--top-p) TOP_P="$2"; shift 2 ;;
		--max-tokens) MAX_TOKENS="$2"; shift 2 ;;
		--timeout) TIMEOUT="$2"; shift 2 ;;
		--eval-batch-size) BATCH="$2"; shift 2 ;;
		--tag) TAG="$2"; shift 2 ;;
		--root) ROOT="$2"; shift 2 ;;
		--evalscope) EVALSCOPE="$2"; shift 2 ;;
		--python) PYTHON="$2"; shift 2 ;;
		--hf-home) HF_HOME="$2"; shift 2 ;;
		--dry-run) DRY_RUN=1; shift ;;
		-h|--help) usage; exit 0 ;;
		*) die "Unknown argument: $1" ;;
	esac
done

[[ "${RUNS}" =~ ^[1-9][0-9]*$ ]] || die "--runs must be a positive integer"
[[ -n "${DATASETS}" ]] || die "--datasets must not be empty"
[[ -n "${API_URL}" ]] || API_URL="http://${HOST}:${PORT}/v1"
API_URL="${API_URL%/}"
[[ -n "${HF_HOME:-}" ]] && export HF_HOME

require_command() {
	command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}
require_command curl
require_command "$PYTHON"

if [[ "${DRY_RUN}" -eq 0 ]]; then
	command -v "$EVALSCOPE" >/dev/null 2>&1 || \
		die "evalscope not found: ${EVALSCOPE}. Set EVALSCOPE_BIN or --evalscope."
fi

AUTH_ARGS=()
if [[ -n "${API_KEY}" && "${API_KEY}" != "EMPTY" ]]; then
	AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
fi

MODELS_JSON="$(curl -sf --max-time 20 "${AUTH_ARGS[@]}" "${API_URL}/models")" || \
	die "vLLM is not ready: ${API_URL}/models"
[[ -n "${MODELS_JSON}" ]] || die "Empty response from ${API_URL}/models"

if [[ -z "${MODEL}" ]]; then
	MODEL="$(printf '%s' "${MODELS_JSON}" | "$PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
data = payload.get("data") or []
if not data:
    raise SystemExit("no models served")
print(data[0]["id"])
')" || die "Unable to discover a served model"
else
	printf '%s' "${MODELS_JSON}" | "$PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
model = sys.argv[1]
ids = [item.get("id") for item in payload.get("data") or []]
if model not in ids:
    raise SystemExit(f"requested model {model!r} not served; available: {ids}")
' "${MODEL}" || die "Requested model is not served"
fi

if [[ -z "${REASONING_EFFORT}" ]]; then
	REASONING_EFFORT="$(printf '%s' "${MODELS_JSON}" | "$PYTHON" -c '
import json, re, sys
payload = json.load(sys.stdin)
model = sys.argv[1]
data = payload.get("data") or []
entry = next((item for item in data if item.get("id") == model), data[0] if data else {})

def find(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if "reasoning_effort" in key.lower() and isinstance(item, str) and item.strip():
                return item.strip()
            found = find(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find(item)
            if found:
                return found
    return None

effort = find(entry)
if not effort:
    text = " ".join(str(entry.get(key, "")) for key in ("id", "root")).lower()
    for token in ("xhigh", "high", "medium", "low", "minimal", "none"):
        if re.search(r"(^|[-_/])" + re.escape(token) + r"($|[-_/])", text):
            effort = token
            break
print(effort or "xhigh")
' "${MODEL}")" || die "Unable to discover reasoning_effort"
fi

case "${REASONING_EFFORT}" in
	xhigh|high|medium|low|minimal|none) ;;
	*) warn "Unusual reasoning_effort value: ${REASONING_EFFORT}" ;;
esac

sanitize() {
	printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | \
		sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//'
}
if [[ -z "${TAG}" ]]; then
	TAG="$(sanitize "${MODEL}_evalscope_7r_$(date -u +%Y%m%dT%H%M%SZ)")"
else
	TAG="$(sanitize "${TAG}")"
fi
[[ -n "${TAG}" ]] || die "Unable to derive a tag"

SUITE_DIR="${ROOT}/${TAG}_suite"
RUNS_DIR="${SUITE_DIR}/runs"
STATE="${SUITE_DIR}/state.json"
mkdir -p "${SUITE_DIR}" "${RUNS_DIR}"

GEN_CFG="$("$PYTHON" -c '
import json, sys
print(json.dumps({
    "temperature": float(sys.argv[1]),
    "top_p": float(sys.argv[2]),
    "max_tokens": int(sys.argv[3]),
    "reasoning_effort": sys.argv[4],
    "n": 1,
    "timeout": int(sys.argv[5]),
}))
' "${TEMPERATURE}" "${TOP_P}" "${MAX_TOKENS}" "${REASONING_EFFORT}" "${TIMEOUT}")" || die "Unable to build generation config"

log "Model: ${MODEL}"
log "reasoning_effort: ${REASONING_EFFORT}"
log "API: ${API_URL}"
log "Tasks: ${DATASETS}; runs: ${RUNS}"
log "Tag: ${TAG}"
log "Suite: ${SUITE_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
	printf 'model=%s\nreasoning_effort=%s\napi_url=%s\ndatasets=%s\nruns=%s\ntag=%s\nroot=%s\ngen_cfg=%s\n' \
		"${MODEL}" "${REASONING_EFFORT}" "${API_URL}" "${DATASETS}" "${RUNS}" "${TAG}" "${ROOT}" "${GEN_CFG}"
	exit 0
fi

FINISHED=0
cleanup() {
	local rc=$?
	if [[ "${FINISHED}" -eq 0 && -f "${STATE}" ]]; then
		"$PYTHON" "${SCRIPT_DIR}/evalscope_data.py" suite-finish \
			--state "${STATE}" --status interrupted --exit-code "${rc}" \
			--message "interrupted" >/dev/null 2>&1 || true
	fi
}
trap cleanup EXIT INT TERM

EVALSCOPE_PATH="$(command -v "$EVALSCOPE" 2>/dev/null || true)"
ENV_LABEL=""
PYTHON_BIN=""
if [[ -n "$EVALSCOPE_PATH" ]]; then
	EVALSCOPE_ENV_DIR="$(cd "$(dirname "$EVALSCOPE_PATH")/.." && pwd -P)"
	ENV_LABEL="$(basename "$EVALSCOPE_ENV_DIR")"
	PYTHON_BIN="$EVALSCOPE_ENV_DIR/bin/python"
	[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=""
fi

"$PYTHON" "${SCRIPT_DIR}/evalscope_data.py" suite-start \
	--state "${STATE}" --name "${TAG}" --runs "${RUNS}" \
	--model "${MODEL}" --host "${HOST}" --port "${PORT}" --api-url "${API_URL}" \
	--tasks "${DATASETS}" --reasoning-effort "${REASONING_EFFORT}" \
	--temperature "${TEMPERATURE}" --top-p "${TOP_P}" --max-tokens "${MAX_TOKENS}" \
	--env "${ENV_LABEL}" --python "${PYTHON_BIN}" \
	--pid $$ --command "${RUNNER_COMMAND}" || die "Unable to write suite state"

for ((index = 1; index <= RUNS; index++)); do
	printf -v run_tag '%s_run%02d' "${TAG}" "${index}"
	run_dir="${RUNS_DIR}/${run_tag}"
	mkdir -p "${run_dir}"
	started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	recorded_key="${API_KEY}"
	[[ "${API_KEY}" == "EMPTY" ]] || recorded_key="***"
	run_command_args=(
		"$EVALSCOPE" eval
		--model "${MODEL}" --eval-type openai_api --api-key "${recorded_key}"
		--datasets ${DATASETS} --generation-config "${GEN_CFG}"
		--eval-batch-size "${BATCH}" --api-url "${API_URL}"
		--work-dir "${run_dir}/evalscope"
	)
	run_command="$("$PYTHON" -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "${run_command_args[@]}")"
	"$PYTHON" "${SCRIPT_DIR}/evalscope_data.py" suite-run-start \
		--state "${STATE}" --index "${index}" --tag "${run_tag}" \
		--started-at "${started_at}" --log "${run_dir}/eval.log" \
		--command "${run_command}" || \
		die "Unable to update run ${index} state"

	log "run ${index}/${RUNS}: ${run_tag}"
	"$EVALSCOPE" eval \
		--model "${MODEL}" \
		--eval-type openai_api \
		--api-key "${API_KEY}" \
		--datasets ${DATASETS} \
		--generation-config "${GEN_CFG}" \
		--eval-batch-size "${BATCH}" \
		--api-url "${API_URL}" \
		--work-dir "${run_dir}/evalscope" \
		>"${run_dir}/eval.log" 2>&1
	rc=$?
	if [[ "${rc}" -ne 0 ]]; then
		warn "run ${index}/${RUNS} exited with ${rc}; see ${run_dir}/eval.log"
	fi

	"$PYTHON" "${SCRIPT_DIR}/evalscope_data.py" suite-run-finish \
		--state "${STATE}" --index "${index}" --exit-code "${rc}" \
		--work-dir "${run_dir}/evalscope" --tasks "${DATASETS}" \
		--log "${run_dir}/eval.log" || warn "Unable to parse run ${index} reports"
done

"$PYTHON" "${SCRIPT_DIR}/evalscope_data.py" suite-aggregate \
	--state "${STATE}" --output "${SUITE_DIR}/summary.json" \
	--markdown "${SUITE_DIR}/summary.md" || die "Unable to aggregate results"

FINISHED=1
status="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "failed"))' "${STATE}")"
log "Suite status: ${status}"
log "Summary: ${SUITE_DIR}/summary.json"
[[ "${status}" == "completed" ]] || exit 1
