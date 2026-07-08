#!/bin/bash
set -euo pipefail

# Usage: bash run_agent_v2.sh --task [swebp|swe-verified|mcp-atlas] [OPTIONS]
#
# Common:
#   --task TASK           swebp | swe-verified | mcp-atlas     (default: swebp)
#   --port N              vLLM API port                         (default: 8888)
#   --model PATH          Model path; starts vLLM when provided
#   --max-model-len N     vLLM max_model_len                    (default: 32768)
#   --served-name NAME    vLLM served-model-name                (default: gpt-3.5-turbo)
#   --tag NAME            Run label for outputs / logs          (default: timestamp)
#   --skip-serve          Skip vLLM readiness check
#
# SWE-bench Pro (--task swebp):
#   --instances KEY       full | 10test | /path/file.json       (default: full)
#   --slice S:E           Instance slice, e.g. 0:20
#   --workers N           Parallel agent workers                (default: 2)
#   --step-limit N        Max steps per instance                (default: 250)
#   --redo                Re-run already-completed instances
#
# SWE-bench Verified (--task swe-verified):
#   --num-tasks N         Limit to first N tasks
#   --slice S:E           Exact slice (alternative to --num-tasks)
#   --workers N           Parallel agent workers                (default: 2)
#   --step-limit N        Max steps per instance                (default: 250)
#
# MCP-Atlas (--task mcp-atlas):
#   --num-tasks N         Limit to first N tasks                (default: all 500)
#   --concurrency N       Parallel eval workers                 (default: 5)
#   --sandbox-port P      MCP sandbox Docker port               (default: 1984)
#   --harness-port P      Agent harness port                    (default: 3001)

# =============================================================================
# User-editable configuration
# =============================================================================
BENCHMARK_DIR="${BENCHMARK_DIR:-$PWD}"

# Default run parameters
TASK="swebp"
PORT=8888
MODEL_PATH=""
MAX_MODEL_LEN=262144
SERVED_MODEL_NAME="gpt-3.5-turbo"
WORKERS=2
STEP_LIMIT=250
CONCURRENCY=5
NUM_TASKS=""
INSTANCES_KEY="full"
SLICE_ARG=""
REDO_FLAG=""
SANDBOX_PORT=1984
HARNESS_PORT=3001
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SKIP_SERVE=false

# Fixed derived paths
# swebp: version pinned by SWE-bench_Pro-os submodule
AGENT_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os/mini-swe-agent"
# swe-verified: standalone mini-swe-agent on main
AGENT_DIR_VERIFIED="${BENCHMARK_DIR}/mini-swe-agent"
MCP_DIR="${BENCHMARK_DIR}/mcp-atlas"
LOG_DIR="${BENCHMARK_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Venvs live at BENCHMARK_DIR level
SWEBP_VENV="${BENCHMARK_DIR}/.venv-swebp"
SWE_VENV="${BENCHMARK_DIR}/.venv-swe"
MCP_VENV="${BENCHMARK_DIR}/.venv-mcp"

VLLM_PID=""
HARNESS_PID=""

# =============================================================================
# Helpers
# =============================================================================
usage() {
    sed -n '/^# Usage/,/^[^#]/p' "${BASH_SOURCE[0]}" | head -n -1 | sed 's/^# \?//'
    exit 0
}

die() { echo "[ERROR] $*" >&2; exit 1; }

cleanup() {
    [[ -n "${HARNESS_PID}" ]] && kill "${HARNESS_PID}" 2>/dev/null || true
    [[ -n "${VLLM_PID}" ]]   && kill "${VLLM_PID}"   2>/dev/null || true
    local z
    z=$(docker ps -a --filter name=minisweagent --filter status=created -q 2>/dev/null || true)
    [[ -n "${z}" ]] && docker rm -f ${z} 2>/dev/null || true
}
trap cleanup EXIT

model_id() {
    curl -sf "http://127.0.0.1:${PORT}/v1/models" \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])"
}

# =============================================================================
# Setup
# =============================================================================
setup_swebp() {
    local swebp_dir="${BENCHMARK_DIR}/SWE-bench_Pro-os"
    [[ -d "${swebp_dir}/.git" ]] || \
        git clone --depth 1 https://github.com/scaleapi/SWE-bench_Pro-os.git "${swebp_dir}"
    [[ -x "${SWEBP_VENV}/bin/vllm" ]] || \
        die "swebp venv not ready — run: bash setup_agent.sh swebp"
}

setup_swe_verified() {
    [[ -d "${AGENT_DIR_VERIFIED}/.git" ]] || \
        git clone --depth 1 https://github.com/SWE-agent/mini-swe-agent.git "${AGENT_DIR_VERIFIED}"
    [[ -x "${SWE_VENV}/bin/vllm" ]] || \
        die "swe-verified venv not ready — run: bash setup_agent.sh swe-verified"
}

setup_mcp() {
    echo "=== Setup: MCP-Atlas ==="
    if [[ ! -d "${MCP_DIR}/.git" ]]; then
        git clone --depth 1 https://github.com/scaleapi/mcp-atlas.git "${MCP_DIR}"
    fi
    [[ -x "${MCP_VENV}/bin/vllm" ]] || \
        die "mcp-atlas venv not ready — run: bash setup_agent.sh mcp-atlas"
    if [[ ! -d "${MCP_DIR}/services/agent-harness/node_modules" ]]; then
        export NVM_DIR="${HOME}/.nvm"
        [[ -s "${NVM_DIR}/nvm.sh" ]] && source "${NVM_DIR}/nvm.sh"
        npm install --prefix "${MCP_DIR}/services/agent-harness" --silent
    fi
    if ! docker image inspect agent-environment:latest &>/dev/null; then
        docker pull ghcr.io/scaleapi/mcp-atlas:1.2.5
        docker tag  ghcr.io/scaleapi/mcp-atlas:1.2.5 agent-environment:latest
    fi
    if [[ ! -f "${MCP_DIR}/.env" ]]; then
        cp "${MCP_DIR}/env.template" "${MCP_DIR}/.env"
        cat >> "${MCP_DIR}/.env" << ENVEOF

# auto-generated by run_agent_v2.sh
LLM_API_KEY=EMPTY
LLM_BASE_URL=http://localhost:${PORT}
MCP_SANDBOX_URL=http://localhost:${SANDBOX_PORT}
PORT=${HARNESS_PORT}
EVAL_LLM_API_KEY=EMPTY
EVAL_LLM_BASE_URL=http://localhost:${PORT}
EVAL_LLM_MODEL=${SERVED_MODEL_NAME}
LOG_LEVEL=info
ENVEOF
    fi
}

# =============================================================================
# vLLM
# =============================================================================
start_vllm_server() {
    [[ -z "${MODEL_PATH}" ]] && return
    local vllm_bin
    case "${TASK}" in
        swebp)        vllm_bin="${SWEBP_VENV}/bin/vllm" ;;
        swe-verified) vllm_bin="${SWE_VENV}/bin/vllm" ;;
        mcp-atlas)    vllm_bin="${MCP_VENV}/bin/vllm" ;;
    esac
    local venv_bin
    case "${TASK}" in
        swebp)        venv_bin="${SWEBP_VENV}/bin" ;;
        swe-verified) venv_bin="${SWE_VENV}/bin" ;;
        mcp-atlas)    venv_bin="${MCP_VENV}/bin" ;;
    esac
    echo "=== Starting vLLM (${MODEL_PATH##*/}, port=${PORT}) ==="
    local log="${LOG_DIR}/vllm_${RUN_TAG}.log"
    VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}" \
    PATH="${venv_bin}:${PATH}" nohup "${vllm_bin}" serve "${MODEL_PATH}" \
        --port              "${PORT}" \
        --max-model-len     "${MAX_MODEL_LEN}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL:-0.7}" \
        --enable-auto-tool-choice \
        --tool-call-parser  hermes \
        > "${log}" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID=${VLLM_PID}  log=${log}"
}

wait_for_server() {
    [[ "${SKIP_SERVE}" == true ]] && return
    local api="http://127.0.0.1:${PORT}/v1"
    local max_wait="${VLLM_WAIT_RETRIES:-180}"
    echo "Waiting for vLLM at ${api} (up to $((max_wait * 10))s) ..."
    for i in $(seq 1 "${max_wait}"); do
        curl -sf "${api}/models" -o /dev/null && break
        echo "  [${i}/${max_wait}] not ready, retrying in 10s..."
        sleep 10
    done
    curl -sf "http://127.0.0.1:${PORT}/v1/models" -o /dev/null \
        || die "vLLM not reachable at port ${PORT}"
    echo "vLLM ready. Model: $(model_id)"
}

# =============================================================================
# Benchmark runners
# =============================================================================
run_swebp() {
    local mid; mid=$(model_id)
    local out_dir="${AGENT_DIR}/results/run_${RUN_TAG}"
    local swebp_dir="${BENCHMARK_DIR}/SWE-bench_Pro-os"
    local patches="${swebp_dir}/patches_${RUN_TAG}.json"
    local inst_csv="${swebp_dir}/instances_${RUN_TAG}.csv"
    local eval_out="${swebp_dir}/eval_output_${RUN_TAG}"
    local py3="${SWEBP_VENV}/bin/python3"
    local cfg_base="${AGENT_DIR}/config/swebp_vllm.yaml"
    local cfg_run="${AGENT_DIR}/config/swebp_vllm_run.yaml"

    echo "=== SWE-bench Pro | model=${mid} workers=${WORKERS} step_limit=${STEP_LIMIT} ==="
    echo "    out=${out_dir}"

    [[ -f "${cfg_base}" ]] || die "Missing ${cfg_base} — run: bash setup_agent.sh swebp"
    [[ -f "${AGENT_DIR}/run_swebp.py" ]] || die "Missing run_swebp.py — run: bash setup_agent.sh swebp"

    # Generate runtime config: override model, api_base, step_limit
    _SWV_BASE="${cfg_base}" \
    _SWV_OUT="${cfg_run}" \
    _SWV_SERVED="${SERVED_MODEL_NAME}" \
    _SWV_PORT="${PORT}" \
    _SWV_STEP="${STEP_LIMIT}" \
    "${py3}" -c '
import yaml, os
cfg = yaml.safe_load(open(os.environ["_SWV_BASE"]))
cfg["agent"]["step_limit"] = int(os.environ["_SWV_STEP"])
cfg["model"] = {
    "model_name": "openai/" + os.environ["_SWV_SERVED"],
    "cost_tracking": "ignore_errors",
    "model_kwargs": {
        "api_base": "http://localhost:" + os.environ["_SWV_PORT"] + "/v1",
        "api_key": "dummy",
        "drop_params": True,
        "temperature": 0.0,
    }
}
os.makedirs(os.path.dirname(os.environ["_SWV_OUT"]), exist_ok=True)
open(os.environ["_SWV_OUT"], "w").write(yaml.dump(cfg, allow_unicode=True))
print("Config:", os.environ["_SWV_OUT"])
'

    # Instance file selection
    local inst_file
    case "${INSTANCES_KEY}" in
        full)   inst_file="${AGENT_DIR}/data/swebench_pro_instances.json" ;;
        10test) inst_file="${AGENT_DIR}/data/swebp_10instances.json" ;;
        *)
            [[ -f "${INSTANCES_KEY}" ]] || die "Instance file not found: ${INSTANCES_KEY}"
            inst_file="${INSTANCES_KEY}"
            ;;
    esac

    local slice_opt=()
    [[ -n "${SLICE_ARG}" ]] && slice_opt=(--slice "${SLICE_ARG}")

    local redo_opt=()
    [[ "${REDO_FLAG}" == "--redo-existing" ]] && redo_opt=(--redo)

    cd "${AGENT_DIR}"
    "${py3}" run_swebp.py \
        --instances "${inst_file}" \
        --config    "${cfg_run}" \
        --output    "${out_dir}" \
        --workers   "${WORKERS}" \
        "${slice_opt[@]}" \
        "${redo_opt[@]}"

    [[ -f "${out_dir}/preds.json" ]] || die "preds.json not generated"
    local n; n=$("${py3}" -c "import json; print(len(json.load(open('${out_dir}/preds.json'))))")
    echo "Generated ${n} patches -> ${out_dir}/preds.json"

    # Convert preds.json → patches format using convert_preds.py
    echo "=== Convert patches ==="
    cd "${swebp_dir}"
    "${py3}" convert_preds.py \
        --preds  "${out_dir}/preds.json" \
        --output "${patches}" \
        --prefix "${mid}_step${STEP_LIMIT}"

    # Prepare instance CSV from local JSON
    echo "=== Prepare instance CSV ==="
    SLICE_ARG="${SLICE_ARG}" \
    "${py3}" -c "
import json, os, pandas as pd
d = json.load(open('${inst_file}'))
sl = os.environ.get('SLICE_ARG', '')
if sl:
    parts = [int(x) if x else None for x in sl.split(':')]
    d = d[slice(*parts)]
pd.DataFrame(d).to_csv('${inst_csv}', index=False)
print(f'CSV: {len(d)} rows -> ${inst_csv}')
"

    echo "=== Evaluate (local Docker) ==="
    "${py3}" swe_bench_pro_eval.py \
        --raw_sample_path "${inst_csv}" \
        --patch_path      "${patches}" \
        --output_dir      "${eval_out}" \
        --scripts_dir     run_scripts \
        --dockerhub_username jefzda \
        --use_local_docker \
        --num_workers "${WORKERS}"

    echo ""
    "${py3}" -c "
import json
r = json.load(open('${eval_out}/eval_results.json'))
passed = sum(r.values()); total = len(r)
print(f'Score: {passed}/{total} = {passed/total*100:.1f}%')
for iid, ok in sorted(r.items(), key=lambda x: (not x[1], x[0])):
    print(f'  {\"PASS\" if ok else \"FAIL\"}  {iid[:70]}')
"
}

run_swe_verified() {
    local mid; mid=$(model_id)
    local out_dir="${AGENT_DIR_VERIFIED}/results/swe_verified_${RUN_TAG}"
    # mini-extra swebench writes preds.jsonl directly (swebench harness format)
    local preds_jsonl="${out_dir}/preds.jsonl"

    echo "=== SWE-bench Verified | model=${SERVED_MODEL_NAME} workers=${WORKERS} max_iterations=${STEP_LIMIT} ==="

    local slice_opt=()
    if   [[ -n "${SLICE_ARG}" ]]; then slice_opt=(--slice "${SLICE_ARG}")
    elif [[ -n "${NUM_TASKS}"  ]]; then slice_opt=(--slice "0:${NUM_TASKS}")
    fi

    cd "${AGENT_DIR_VERIFIED}"
    "${SWE_VENV}/bin/mini-extra" swebench \
        -m "${SERVED_MODEL_NAME}" \
        --subset  verified \
        --split   test \
        --workers "${WORKERS}" \
        --output  "${out_dir}" \
        -c "model.model_kwargs.base_url=http://localhost:${PORT}/v1" \
        -c "model.model_kwargs.api_key=EMPTY" \
        -c "agent.max_iterations=${STEP_LIMIT}" \
        -c swebench.yaml \
        "${slice_opt[@]}"

    echo ""
    # preds.jsonl may be in out_dir, or mini-extra may write preds.json — handle both
    if [[ ! -f "${preds_jsonl}" ]] && [[ -f "${out_dir}/preds.json" ]]; then
        echo "[INFO] Converting preds.json → preds.jsonl for swebench evaluator"
        "${SWE_VENV}/bin/python3" -c "
import json, sys
preds = json.load(open('${out_dir}/preds.json'))
with open('${preds_jsonl}', 'w') as f:
    for iid, v in preds.items():
        rec = v if isinstance(v, dict) else {'model_patch': v}
        rec.setdefault('instance_id', iid)
        rec.setdefault('model_name_or_path', '${SERVED_MODEL_NAME}')
        f.write(json.dumps(rec) + '\n')
print(f'Wrote {len(preds)} predictions to ${preds_jsonl}')
"
    fi

    if [[ -f "${preds_jsonl}" ]]; then
        local n; n=$(wc -l < "${preds_jsonl}")
        echo "Tasks run : ${n}"
        echo "Preds     : ${preds_jsonl}"

        local sb="${SWE_VENV}/bin/sb-cli"
        if [[ -f "${sb}" ]] && [[ -n "${SWEBENCH_API_KEY:-}" ]]; then
            echo "=== Submitting to SWE-bench cloud (sb-cli) ==="
            "${sb}" submit swe-bench_verified test \
                --predictions_path "${preds_jsonl}" \
                --run_id "${RUN_TAG}"
        else
            echo ""
            echo "[INFO] Set SWEBENCH_API_KEY to auto-submit via sb-cli:"
            echo "  ${SWE_VENV}/bin/sb-cli submit swe-bench_verified test \\"
            echo "    --predictions_path ${preds_jsonl} \\"
            echo "    --run_id ${RUN_TAG}"
        fi
    else
        echo "[WARNING] No predictions found at ${preds_jsonl}"
    fi
}

run_mcp() {
    local mid; mid=$(model_id)
    local out_dir="${MCP_DIR}/outputs/run_${RUN_TAG}"
    local eval_csv="${out_dir}/outputs.csv"
    local scored_dir="${out_dir}/scored"
    local gt_csv="${MCP_DIR}/outputs/groundtruth.csv"
    local harness_log="${out_dir}/harness.log"
    local mcp_python="${MCP_VENV}/bin/python"
    [[ -x "${mcp_python}" ]] || die "MCP venv not found at ${MCP_VENV} — run: bash setup_agent.sh mcp-atlas"

    mkdir -p "${out_dir}"
    echo "=== MCP-Atlas | model=${mid} concurrency=${CONCURRENCY} ==="

    echo "--- MCP sandbox (port ${SANDBOX_PORT}) ---"
    if docker ps --filter "publish=${SANDBOX_PORT}" --format '{{.Names}}' | grep -q .; then
        echo "Sandbox already running"
    else
        docker run -d --rm --name "mcp-sandbox-${RUN_TAG}" \
            -p "${SANDBOX_PORT}:1984" --env-file "${MCP_DIR}/.env" \
            agent-environment:latest
        for i in $(seq 1 18); do
            local online
            online=$(curl -sf "http://localhost:${SANDBOX_PORT}/enabled-servers" 2>/dev/null \
                | python3 -c "import json,sys; print(json.load(sys.stdin)['online'])" 2>/dev/null || echo 0)
            [[ "${online}" -gt 0 ]] && { echo "Sandbox ready: ${online} servers online"; break; }
            echo "  [${i}/18] waiting for sandbox..."; sleep 10
        done
    fi

    echo "--- Agent harness (port ${HARNESS_PORT}) ---"
    export NVM_DIR="${HOME}/.nvm"
    [[ -s "${NVM_DIR}/nvm.sh" ]] && source "${NVM_DIR}/nvm.sh"

    if ! curl -sf "http://localhost:${HARNESS_PORT}/health" -o /dev/null 2>/dev/null; then
        cd "${MCP_DIR}"
        local vllm_url="http://localhost:${PORT}"
        PORT="${HARNESS_PORT}" \
            LLM_BASE_URL="${vllm_url}" \
            LLM_API_KEY="EMPTY" \
            MCP_SANDBOX_URL="http://localhost:${SANDBOX_PORT}" \
            nohup npm run dev --prefix services/agent-harness > "${harness_log}" 2>&1 &
        HARNESS_PID=$!
        for i in $(seq 1 12); do
            curl -sf "http://localhost:${HARNESS_PORT}/health" -o /dev/null 2>/dev/null \
                && { echo "Harness ready (PID=${HARNESS_PID})"; break; }
            echo "  [${i}/12] waiting for harness..."; sleep 5
        done
    fi
    curl -sf "http://localhost:${HARNESS_PORT}/health" -o /dev/null \
        || die "Harness not ready at port ${HARNESS_PORT}"

    echo "--- Running eval ---"
    cd "${MCP_DIR}"
    local num_opt=()
    [[ -n "${NUM_TASKS}" ]] && num_opt=(--num-tasks "${NUM_TASKS}")

    HARNESS_URL="http://localhost:${HARNESS_PORT}" \
        "${mcp_python}" run_eval.py \
        --model       "${mid}" \
        --output      "${eval_csv}" \
        --concurrency "${CONCURRENCY}" \
        "${num_opt[@]}"

    [[ -f "${eval_csv}" ]] || die "${eval_csv} not generated"

    if [[ ! -f "${gt_csv}" ]]; then
        echo "Exporting ground truth from HuggingFace..."
        "${mcp_python}" -c "
from datasets import load_dataset; import pandas as pd, os
os.makedirs('$(dirname "${gt_csv}")', exist_ok=True)
ds = load_dataset('ScaleAI/MCP-Atlas', split='train')
ds.to_pandas().to_csv('${gt_csv}', index=False)
print(f'Ground truth: {len(ds)} tasks -> ${gt_csv}')
"
    fi

    echo "--- Scoring (LLM-as-judge) ---"
    EVAL_LLM_BASE_URL="http://localhost:${PORT}" \
    EVAL_LLM_API_KEY="EMPTY" \
    EVAL_LLM_MODEL="${mid}" \
        "${mcp_python}" "${MCP_DIR}/services/scoring/score_claims.py" \
        --groundtruth-file "${gt_csv}" \
        --model-file       "${eval_csv}" \
        --model-name       "${mid}" \
        --output-dir       "${scored_dir}"

    echo ""
    "${mcp_python}" -c "
import json, glob, sys
files = sorted(glob.glob('${scored_dir}/coverage_stats_*.json'))
if not files: print('[no stats found]'); sys.exit(0)
d = json.load(open(files[-1])); s = d.get('all', d)
print(f'Model     : ${mid}')
print(f'Tasks     : {s.get(\"total_tasks\",\"?\")}')
print(f'Pass@0.75 : {s.get(\"pass_rate_0.75\",\"?\")}%')
print(f'Coverage  : {float(s.get(\"mean_coverage\",0)):.4f}')
"
}

# =============================================================================
# Argument parsing
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          TASK="$2";              shift 2 ;;
        --port)          PORT="$2";              shift 2 ;;
        --model)         MODEL_PATH="$2";        shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2";     shift 2 ;;
        --served-name)   SERVED_MODEL_NAME="$2"; shift 2 ;;
        --instances)     INSTANCES_KEY="$2";     shift 2 ;;
        --slice)         SLICE_ARG="$2";         shift 2 ;;
        --workers)       WORKERS="$2";           shift 2 ;;
        --step-limit)    STEP_LIMIT="$2";        shift 2 ;;
        --redo)          REDO_FLAG="--redo";     shift ;;
        --num-tasks)     NUM_TASKS="$2";         shift 2 ;;
        --concurrency)   CONCURRENCY="$2";       shift 2 ;;
        --sandbox-port)  SANDBOX_PORT="$2";      shift 2 ;;
        --harness-port)  HARNESS_PORT="$2";      shift 2 ;;
        --tag)           RUN_TAG="$2";           shift 2 ;;
        --skip-serve)    SKIP_SERVE=true;        shift ;;
        --help|-h)       usage ;;
        *)               die "Unknown option: $1" ;;
    esac
done

case "${TASK}" in
    swebp|swe-verified|mcp-atlas) ;;
    *) die "--task must be swebp | swe-verified | mcp-atlas" ;;
esac

# =============================================================================
# Main
# =============================================================================
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/agent_${TASK}_${RUN_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo " Benchmark : ${TASK}"
echo " Tag       : ${RUN_TAG}"
echo " vLLM port : ${PORT}"
[[ -n "${MODEL_PATH}" ]] && echo " Model     : ${MODEL_PATH##*/}  (max_len=${MAX_MODEL_LEN})"
echo " Started   : $(date)"
echo "================================================================"
echo ""

if   [[ "${TASK}" == "mcp-atlas"    ]]; then setup_mcp
elif [[ "${TASK}" == "swe-verified" ]]; then setup_swe_verified
else                                          setup_swebp
fi

start_vllm_server
wait_for_server

if   [[ "${TASK}" == "swebp"        ]]; then run_swebp
elif [[ "${TASK}" == "swe-verified" ]]; then run_swe_verified
elif [[ "${TASK}" == "mcp-atlas"    ]]; then run_mcp
fi

echo ""
echo "================================================================"
echo " Benchmark [${TASK}] finished at $(date)"
echo " Log: ${LOG_FILE}"
echo "================================================================"
