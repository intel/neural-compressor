#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/versions.env"

usage() {
    cat <<'EOF'
Usage: bash run_swebenchpro.sh [OPTIONS]

Run SWE-bench Pro V2 with the official locked mini-SWE agent, then replay
captured patches in pristine containers with the official V2 verifier.

  --host HOST          vLLM host (default: 127.0.0.1; mapped to Docker gateway)
  --port PORT          vLLM port (default: 8888)
  --served-name NAME   Model ID exposed by vLLM (default: discover)
  --agent-base-url URL Sandbox-reachable model API URL, including /v1
  --environment NAME   docker (default) or modal; Modal needs credentials
  --subset NAME        full (642 tasks, default) or hard (51 tasks)
  --num-tasks N        Select the first N tasks in sorted task-directory order
  --slice START:END    Select a slice instead of --num-tasks
  --workers N          Parallel generation trials (default: 8)
  --eval-workers N     Parallel fresh-sandbox replay trials (default: 4)
  --step-limit N       Agent model-call limit (default: 250)
  --tag TAG            Unique output label (default: UTC timestamp)
  --skip-eval          Capture patches only; do not report an accuracy
  -h, --help           Show this help

Environment: AGENT_MAX_TOKENS (8192), VLLM_API_KEY (EMPTY), OPENAI_BASE_URL,
BENCHMARK_DIR. The shared vLLM server is never stopped. The agent phase has
no network access except the model host; setup and verification are online.
V1 chunking, image cleanup, and --block-network options are not supported.
EOF
}

WORKERS=8
EVAL_WORKERS=4
STEP_LIMIT=250
SUBSET=full
ENVIRONMENT=docker
NUM_TASKS=""
SLICE_ARG=""
SERVED_MODEL_NAME=""
AGENT_BASE_URL=""
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
SKIP_EVAL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --host|--port|--served-name|--agent-base-url|--environment|--subset|--num-tasks|--slice|--workers|--eval-workers|--step-limit|--tag)
            [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
            case "$1" in
                --host) VLLM_HOST="$2" ;;
                --port) VLLM_PORT="$2" ;;
                --served-name) SERVED_MODEL_NAME="$2" ;;
                --agent-base-url) AGENT_BASE_URL="$2" ;;
                --environment) ENVIRONMENT="$2" ;;
                --subset) SUBSET="$2" ;;
                --num-tasks) NUM_TASKS="$2" ;;
                --slice) SLICE_ARG="$2" ;;
                --workers) WORKERS="$2" ;;
                --eval-workers) EVAL_WORKERS="$2" ;;
                --step-limit) STEP_LIMIT="$2" ;;
                --tag) RUN_TAG="$2" ;;
            esac
            shift 2 ;;
        *) die "Unknown V2 option: $1 (see --help; V1 options have changed)" ;;
    esac
done

for value in "${WORKERS}" "${EVAL_WORKERS}" "${STEP_LIMIT}" "${AGENT_MAX_TOKENS:-8192}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "Worker, step and token limits must be positive integers"
done
[[ -z "${NUM_TASKS}" || "${NUM_TASKS}" =~ ^[1-9][0-9]*$ ]] || die "--num-tasks must be positive"
[[ -z "${SLICE_ARG}" || "${SLICE_ARG}" =~ ^[0-9]*:[0-9]*$ ]] || die "--slice must use START:END"
[[ -z "${NUM_TASKS}" || -z "${SLICE_ARG}" ]] || die "--num-tasks and --slice are mutually exclusive"
[[ "${SUBSET}" == full || "${SUBSET}" == hard ]] || die "--subset must be full or hard"
[[ "${ENVIRONMENT}" == docker || "${ENVIRONMENT}" == modal ]] || die "--environment must be docker or modal"
require_command python
require_command harbor
require_command git
if [[ "${ENVIRONMENT}" == docker ]]; then
    require_command docker
    docker compose version >/dev/null
fi
init_benchmark_paths
init_vllm_endpoint
readonly PRO_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os"
require_file "${PRO_DIR}/v2/tooling/locked_mini_swe.py"
[[ "$(git -C "${PRO_DIR}" rev-parse HEAD)" == "${SWEBENCH_PRO_COMMIT}" ]] || die "Run setup_swebenchpro.sh for the pinned V2 checkout"
python -c 'import sys; from importlib.metadata import version; assert version("harbor") == sys.argv[1], "Re-run setup_swebenchpro.sh for the pinned Harbor version"' "${SWEBENCH_PRO_HARBOR_VERSION}"
(cd "${PRO_DIR}/v2" && sha256sum --check --quiet SHA256SUMS)

RUN_TAG="$(sanitize_run_tag "${RUN_TAG}")"
readonly OUT_DIR="${OUTPUT_DIR}/swebench_pro_v2_${RUN_TAG}"
[[ ! -e "${OUT_DIR}" ]] || die "Output exists: ${OUT_DIR}; use a new --tag (never mix V1/V2 or different selections)"
# Harbor persists agent credentials in job configs; keep run artifacts private.
umask 077
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/swebench_pro_v2_${RUN_TAG}.log") 2>&1

wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME}")"
if [[ -z "${AGENT_BASE_URL}" ]]; then
    AGENT_BASE_URL="${OPENAI_BASE_URL}"
    if [[ "${ENVIRONMENT}" == docker ]]; then
        # The service must listen on 0.0.0.0, not only the host loopback.
        DOCKER_GATEWAY="$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')"
        [[ -n "${DOCKER_GATEWAY}" ]] || die "Cannot discover Docker gateway; pass --agent-base-url"
        AGENT_BASE_URL="${AGENT_BASE_URL/127.0.0.1/${DOCKER_GATEWAY}}"
        AGENT_BASE_URL="${AGENT_BASE_URL/localhost/${DOCKER_GATEWAY}}"
    fi
fi
python "${SCRIPT_DIR}/lib/swebench_pro_v2.py" prepare \
    --root "${PRO_DIR}/v2" --output "${OUT_DIR}" --subset "${SUBSET}" \
    --num-tasks "${NUM_TASKS}" --slice "${SLICE_ARG}" --step-limit "${STEP_LIMIT}" \
    --max-tokens "${AGENT_MAX_TOKENS:-8192}" --base-url "${AGENT_BASE_URL}" \
    --commit "${SWEBENCH_PRO_COMMIT}" --agent-version "${SWEBENCH_PRO_AGENT_VERSION}"
MODEL_HOST="$(python -c 'import sys; from urllib.parse import urlsplit; print(urlsplit(sys.argv[1]).hostname)' "${AGENT_BASE_URL}")"
export PYTHONPATH="${SCRIPT_DIR}/lib:${PRO_DIR}/v2/tooling${PYTHONPATH:+:${PYTHONPATH}}"
COMMON=(--path "${OUT_DIR}/tasks" -e "${ENVIRONMENT}" --jobs-dir "${OUT_DIR}/jobs" --yes)
AGENT_ENV=()
VERIFIER_ENV=()
# Proxy settings are usable during online setup/verification, but the proxy
# host is deliberately NOT added to the agent network allowlist.
for key in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
    if [[ -n "${!key:-}" ]]; then
        AGENT_ENV+=(--ae "${key}=${!key}")
        VERIFIER_ENV+=(--ve "${key}=${!key}")
    fi
done
SANDBOX_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
SANDBOX_NO_PROXY="${SANDBOX_NO_PROXY:+${SANDBOX_NO_PROXY},}${MODEL_HOST}"
AGENT_ENV+=(--ae "NO_PROXY=${SANDBOX_NO_PROXY}" --ae "no_proxy=${SANDBOX_NO_PROXY}")
VERIFIER_ENV+=(--ve "NO_PROXY=${SANDBOX_NO_PROXY}" --ve "no_proxy=${SANDBOX_NO_PROXY}")

log "Generating V2 patches (agent network allowlist: ${MODEL_HOST})"
harbor run "${COMMON[@]}" --job-name generation --n-concurrent "${WORKERS}" \
    --agent swebench_pro_agent:V2MiniSwe --model "openai/${SERVED_MODEL_NAME}" \
    --ak "version=${SWEBENCH_PRO_AGENT_VERSION}" --ak "config_file=${OUT_DIR}/agent.yaml" \
    --ae "OPENAI_API_KEY=${VLLM_API_KEY}" --ae "MSWEA_API_KEY=${VLLM_API_KEY}" \
    --ae "OPENAI_API_BASE=${AGENT_BASE_URL}" --ae "OPENAI_BASE_URL=${AGENT_BASE_URL}" \
    "${AGENT_ENV[@]}" \
    --allow-agent-host "${MODEL_HOST}" --disable-verification
python "${SCRIPT_DIR}/lib/swebench_pro_v2.py" check-generation --output "${OUT_DIR}"

if [[ "${SKIP_EVAL}" == true ]]; then
    log "Generation only; no V2 accuracy reported. Artifacts: ${OUT_DIR}"
    exit 0
fi
log "Replaying patches on pristine V2 images; only this phase determines accuracy"
harbor run "${COMMON[@]}" --job-name regrade --n-concurrent "${EVAL_WORKERS}" \
    --agent patch_replay:PatchReplayAgent --model replay \
    --ak "source_job=${OUT_DIR}/jobs/generation" "${VERIFIER_ENV[@]}"
python "${SCRIPT_DIR}/lib/swebench_pro_v2.py" report --output "${OUT_DIR}"
log "V2 report: ${OUT_DIR}/report.json (both Harbor jobs retained)"