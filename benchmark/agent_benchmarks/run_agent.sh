#!/bin/bash
set -euo pipefail

# Usage: bash run_agent.sh --task [swebp|swe-verified|mcp-atlas] [OPTIONS]
#
# Common:
#   --task TASK           swebp | swe-verified | mcp-atlas     (default: swebp)
#   --port N              vLLM API port                         (default: 8888)
#   --model PATH          Model path; starts vLLM when provided
#   --scheme NAME         vLLM serving scheme                    (default: FP8)
#   --max-model-len N     vLLM max_model_len                    (default: 262144)
#   --served-name NAME    vLLM served-model-name                (default: gpt-3.5-turbo)
#   --tag NAME            Run label for outputs / logs          (default: timestamp)
#   --skip-serve          Skip vLLM launch and readiness wait
#
# vLLM serving (env vars, set before calling the script):
#   KV_CACHE_DTYPE        kv-cache-dtype                        (default: fp8)
#   BLOCK_SIZE            block-size                            (default: 256)
#   TENSOR_PARALLEL_SIZE  tensor-parallel-size                  (default: 1)
#   GPU_MEM_UTIL          gpu-memory-utilization                (default: 0.9)
#
# Example vLLM command (equivalent to what --model triggers internally):
#   SAFETENSORS_FAST_GPU=1 CUDA_VISIBLE_DEVICES=0,1 \
#   TENSOR_PARALLEL_SIZE=2 bash run_agent.sh --task swebp \
#     --model /path/to/DeepSeek-V4-Flash-MXFP4 --port 8888 ...
#
# Note: DeepSeek-V4-Flash uses 2 GPUs and DeepSeek-V4-Pro uses 8 GPUs.
#   The selected --scheme chooses the specialised serve arguments for those models.
#
# SWE-bench Pro (--task swebp):
#   --num-tasks N         Limit to first N tasks
#   --slice S:E           Exact slice, e.g. 0:20
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
#   --workers N           Parallel eval workers                 (default: 5)
#   --sandbox-port P      MCP sandbox Docker port               (default: 1984)
#   --harness-port P      Agent harness port                    (default: 3001)
#   --skip-sandbox        Skip Docker sandbox startup (use already-running sandbox)

# =============================================================================
# User-editable configuration
# =============================================================================
BENCHMARK_DIR="${BENCHMARK_DIR:-$PWD}"

# Default run parameters
TASK="swebp"
PORT=8888
MODEL_PATH=""
SCHEME="${SCHEME:-FP8}"
MAX_MODEL_LEN=262144
SERVED_MODEL_NAME="gpt-3.5-turbo"
WORKERS=2
STEP_LIMIT=250
NUM_TASKS=""
SLICE_ARG=""
REDO_FLAG=""
SANDBOX_PORT=1984
HARNESS_PORT=3001
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SKIP_SERVE=false
SKIP_SANDBOX=false

# vLLM serving options
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# Fixed derived paths
# swebp: version pinned by SWE-bench_Pro-os submodule
AGENT_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os/mini-swe-agent"
# swe-verified: standalone mini-swe-agent on main
AGENT_DIR_VERIFIED="${BENCHMARK_DIR}/mini-swe-agent"
MCP_DIR="${BENCHMARK_DIR}/mcp-atlas"
LOG_DIR="${BENCHMARK_DIR}/logs"
mkdir -p "${LOG_DIR}"

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
    [[ -d "${swebp_dir}" ]] || die "SWE-bench_Pro-os not found — run: bash setup_agent.sh swebp"
    # Ensure swebench.py is patched for jefzda Docker images
    local patch_file="${BENCHMARK_DIR}/patches/swebench_pro_image.patch"
    local swebench_py="${AGENT_DIR}/src/minisweagent/run/extra/swebench.py"
    if ! grep -q 'dockerhub_tag' "${swebench_py}" 2>/dev/null; then
        [[ -f "${patch_file}" ]] || die "Patch file not found: ${patch_file}"
        git -C "${AGENT_DIR}" apply "${patch_file}" \
            && echo "[setup_swebp] swebench.py patched OK"
    fi
}

setup_swe_verified() {
    [[ -d "${AGENT_DIR_VERIFIED}" ]] || die "mini-swe-agent not found — run: bash setup_agent.sh swe-verified"
}

setup_mcp() {
    [[ -d "${MCP_DIR}" ]] || die "mcp-atlas not found — run: bash setup_agent.sh mcp-atlas"
    if [[ ! -f "${MCP_DIR}/.env" ]]; then
        cp "${MCP_DIR}/env.template" "${MCP_DIR}/.env"
        cat >> "${MCP_DIR}/.env" << ENVEOF

# auto-generated by run_agent.sh (static config only — LLM/port vars passed via shell env)
LOG_LEVEL=info
ENVEOF
    fi
}

# =============================================================================
# vLLM
# =============================================================================
start_vllm_server() {
    [[ -z "${MODEL_PATH}" ]] && return
    [[ "${SKIP_SERVE}" == true ]] && return

    local vllm_bin
    vllm_bin=$(command -v vllm 2>/dev/null || echo "")
    [[ -z "${vllm_bin}" ]] && die "vllm not found in PATH — activate a venv or pass full path"

    local log="${LOG_DIR}/vllm_${RUN_TAG}.log"
    local model_name="${MODEL_PATH%/}"
    model_name="${model_name##*/}"
    local normalized_model_name="${model_name//_/-}"
    local scheme_upper="${SCHEME^^}"
    local tool_call_parser="hermes"

    if [[ "${normalized_model_name}" == *"DeepSeek-V4-Flash"* || "${normalized_model_name}" == *"DeepSeek-V4-Pro"* ]]; then
        tool_call_parser="deepseek_v4"
    fi

    local common_args=(
        --port "${PORT}"
        --served-model-name "${SERVED_MODEL_NAME}"
        --trust-remote-code
        --enable-auto-tool-choice
        --tool-call-parser "${tool_call_parser}"
    )

    echo "=== Starting vLLM (${model_name}, scheme=${scheme_upper}, port=${PORT}) ==="

    if [[ "${normalized_model_name}" == *"DeepSeek-V4-Flash"* || "${normalized_model_name}" == *"DeepSeek-V4-Pro"* ]]; then
        local serve_env=()
        local serve_args=(
            --kv-cache-dtype fp8
            --block-size 256
            --tensor-parallel-size 2
            --attention_config.use_fp4_indexer_cache=True
            --max-model-len 1048576
            --gpu-memory-utilization 0.90
        )


        case "${scheme_upper}" in
            FP8)
                echo "=== Using DeepSeek-V4 ${scheme_upper} serve profile ==="
                serve_env+=(SAFETENSORS_FAST_GPU=1)
                serve_args+=(--moe-backend deep_gemm_mega_moe --enable-expert-parallel)
                ;;
            MXFP4)
                echo "=== Using DeepSeek-V4 ${scheme_upper} serve profile ==="
                serve_env+=(SAFETENSORS_FAST_GPU=1)
                serve_args+=(--moe-backend cutlass)
                ;;
            *)
                die "Unsupported --scheme for DeepSeek-V4 models: ${SCHEME} (expected FP8 or MXFP4)"
                ;;
        esac

        nohup env \
            VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}" \
            "${serve_env[@]}" \
            "${vllm_bin}" serve "${MODEL_PATH}" \
            "${common_args[@]}" \
            "${serve_args[@]}" \
            > "${log}" 2>&1 &
    else
        nohup env \
            SAFETENSORS_FAST_GPU=1 \
            VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}" \
            "${vllm_bin}" serve "${MODEL_PATH}" \
            "${common_args[@]}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --kv-cache-dtype "${KV_CACHE_DTYPE}" \
            --block-size "${BLOCK_SIZE}" \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}" \
            --attention_config.use_fp4_indexer_cache=True \
            --no-enable-flashinfer-autotune \
            > "${log}" 2>&1 &
    fi

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
    local py3="python3"

    echo "=== SWE-bench Pro | model=${mid} workers=${WORKERS} step_limit=${STEP_LIMIT} ==="
    echo "    out=${out_dir}"

    local slice_opt=()
    if   [[ -n "${SLICE_ARG}" ]];  then slice_opt=(--slice "${SLICE_ARG}")
    elif [[ -n "${NUM_TASKS}" ]];  then slice_opt=(--slice "0:${NUM_TASKS}")
    fi

    local redo_opt=()
    [[ -n "${REDO_FLAG}" ]] && redo_opt=(--redo-existing)

    # v1 -c only takes a single config file — generate runtime YAML with overrides
    local cfg_run="${out_dir}/swebp_run.yaml"
    mkdir -p "${out_dir}"
    "${py3}" -c "
import yaml
from minisweagent.config import builtin_config_dir
cfg = yaml.safe_load((builtin_config_dir / 'extra' / 'swebench.yaml').read_text())
cfg['agent']['step_limit'] = ${STEP_LIMIT}
cfg['model'] = {
    'model_name': '${SERVED_MODEL_NAME}',
    'cost_tracking': 'ignore_errors',
    'model_kwargs': {
        'api_base': 'http://localhost:${PORT}/v1',
        'api_key': 'dummy',
        'drop_params': True,
        'temperature': 0.0,
    },
}
open('${cfg_run}', 'w').write(yaml.dump(cfg, allow_unicode=True))
print('Runtime config:', '${cfg_run}')
"

    cd "${AGENT_DIR}"
    mini-extra swebench \
        --subset  ScaleAI/SWE-bench_Pro \
        --split   test \
        -m        "${SERVED_MODEL_NAME}" \
        --workers "${WORKERS}" \
        --output  "${out_dir}" \
        -c        "${cfg_run}" \
        "${slice_opt[@]}" \
        "${redo_opt[@]}"

    [[ -f "${out_dir}/preds.json" ]] || die "preds.json not generated"
    local n; n=$("${py3}" -c "import json; print(len(json.load(open('${out_dir}/preds.json'))))")
    echo "Generated ${n} patches -> ${out_dir}/preds.json"

    # Convert preds.json → patches.json for swe_bench_pro_eval.py
    echo "=== Convert patches ==="
    "${py3}" -c "
import json
preds = json.load(open('${out_dir}/preds.json'))
patches = [
    {'instance_id': v['instance_id'], 'patch': v.get('model_patch', ''), 'prefix': '${mid}_step${STEP_LIMIT}'}
    for v in preds.values()
]
json.dump(patches, open('${patches}', 'w'), indent=2)
print(f'Wrote {len(patches)} patches -> ${patches}')
"

    # Prepare instance CSV from HuggingFace dataset
    echo "=== Prepare instance CSV ==="
    "${py3}" -c "
import pandas as pd
from datasets import load_dataset
ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
data = list(ds)
sl = '${SLICE_ARG}' or ('0:${NUM_TASKS}' if '${NUM_TASKS}' else '')
if sl:
    parts = [int(x) if x else None for x in sl.split(':')]
    data = data[slice(*parts)]
pd.DataFrame(data).to_csv('${inst_csv}', index=False)
print(f'CSV: {len(data)} rows -> ${inst_csv}')
"

    echo "=== Evaluate (local Docker) ==="
    local swebp_dir
    swebp_dir="$(dirname "${AGENT_DIR}")"
    (cd "${swebp_dir}" && "${py3}" swe_bench_pro_eval.py \
        --raw_sample_path "${inst_csv}" \
        --patch_path      "${patches}" \
        --output_dir      "${eval_out}" \
        --scripts_dir     run_scripts \
        --dockerhub_username jefzda \
        --use_local_docker \
        --num_workers "${WORKERS}")

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
    mini-extra swebench \
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
        python3 -c "
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

        if command -v sb-cli &>/dev/null && [[ -n "${SWEBENCH_API_KEY:-}" ]]; then
            echo "=== Submitting to SWE-bench cloud (sb-cli) ==="
            sb-cli submit swe-bench_verified test \
                --predictions_path "${preds_jsonl}" \
                --run_id "${RUN_TAG}"
        else
            echo ""
            echo "[INFO] Set SWEBENCH_API_KEY to auto-submit via sb-cli:"
            echo "  sb-cli submit swe-bench_verified test \\"
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

    local sandbox_wait_retries=18   # 18 × 10s = 3 min max
    local harness_wait_retries=12   # 12 × 5s  = 1 min max

    mkdir -p "${out_dir}"
    echo "=== MCP-Atlas | model=${mid} workers=${WORKERS} ==="

    echo "--- MCP sandbox (port ${SANDBOX_PORT}) ---"
    if [[ "${SKIP_SANDBOX}" == true ]]; then
        echo "Sandbox skipped (--skip-sandbox)"
    elif curl -sf "http://localhost:${SANDBOX_PORT}/enabled-servers" -o /dev/null 2>/dev/null; then
        echo "Sandbox already running"
    else
        docker run -d --rm --name "mcp-sandbox-${RUN_TAG}" \
            -p "${SANDBOX_PORT}:1984" --env-file "${MCP_DIR}/.env" \
            agent-environment:latest
        for i in $(seq 1 ${sandbox_wait_retries}); do
            local online
            online=$(curl -sf "http://localhost:${SANDBOX_PORT}/enabled-servers" 2>/dev/null \
                | python3 -c "import json,sys; print(json.load(sys.stdin)['online'])" 2>/dev/null || echo 0)
            [[ "${online}" -gt 0 ]] && { echo "Sandbox ready: ${online} servers online"; break; }
            echo "  [${i}/${sandbox_wait_retries}] waiting for sandbox..."; sleep 10
        done
    fi

    echo "--- Agent harness (port ${HARNESS_PORT}) ---"
    export NVM_DIR="${HOME}/.nvm"
    [[ -s "${NVM_DIR}/nvm.sh" ]] && source "${NVM_DIR}/nvm.sh"

    # Kill any harness not belonging to our MCP_DIR or pointing to wrong vLLM port
    local h_pid
    h_pid=$(lsof -ti:"${HARNESS_PORT}" 2>/dev/null | head -1 || true)
    if [[ -n "${h_pid}" ]]; then
        local h_cwd h_llm_url
        h_cwd=$(readlink -f /proc/${h_pid}/cwd 2>/dev/null || true)
        h_llm_url=$(tr '\0' '\n' < /proc/${h_pid}/environ 2>/dev/null | grep '^LLM_BASE_URL=' | cut -d= -f2- || true)
        if ! echo "${h_cwd}" | grep -qF "${MCP_DIR}"; then
            echo "Killing stale harness (PID=${h_pid}, wrong dir: ${h_cwd})"
            kill "${h_pid}" 2>/dev/null || true; sleep 2
        elif [[ -n "${h_llm_url}" && "${h_llm_url}" != "http://localhost:${PORT}" ]]; then
            echo "Killing stale harness (PID=${h_pid}, wrong port: ${h_llm_url})"
            kill "${h_pid}" 2>/dev/null || true; sleep 2
        fi
    fi

    if ! curl -sf "http://localhost:${HARNESS_PORT}/health" -o /dev/null 2>/dev/null; then
        cd "${MCP_DIR}"
        local vllm_url="http://localhost:${PORT}"
        PORT="${HARNESS_PORT}" \
            LLM_BASE_URL="${vllm_url}" \
            LLM_API_KEY="EMPTY" \
            MCP_SANDBOX_URL="http://localhost:${SANDBOX_PORT}" \
            nohup npm run dev --prefix services/agent-harness > "${harness_log}" 2>&1 &
        HARNESS_PID=$!
        for i in $(seq 1 ${harness_wait_retries}); do
            curl -sf "http://localhost:${HARNESS_PORT}/health" -o /dev/null 2>/dev/null \
                && { echo "Harness ready (PID=${HARNESS_PID})"; break; }
            echo "  [${i}/${harness_wait_retries}] waiting for harness..."; sleep 5
        done
    fi
    curl -sf "http://localhost:${HARNESS_PORT}/health" -o /dev/null \
        || die "Harness not ready at port ${HARNESS_PORT}"

    echo "--- Running eval ---"
    cd "${MCP_DIR}"
    local num_opt=()
    [[ -n "${NUM_TASKS}" ]] && num_opt=(--num-tasks "${NUM_TASKS}")

    HARNESS_URL="http://localhost:${HARNESS_PORT}" \
        python run_eval.py \
        --model       "${mid}" \
        --output      "${eval_csv}" \
        --concurrency "${WORKERS}" \
        "${num_opt[@]}"

    [[ -f "${eval_csv}" ]] || die "${eval_csv} not generated"

    if [[ ! -f "${gt_csv}" ]]; then
        echo "Exporting ground truth from HuggingFace..."
        python -c "
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
        python "${MCP_DIR}/services/scoring/score_claims.py" \
        --groundtruth-file "${gt_csv}" \
        --model-file       "${eval_csv}" \
        --model-name       "${mid}" \
        --output-dir       "${scored_dir}"

    echo ""
    python -c "
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
        --scheme)        SCHEME="$2";            shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2";     shift 2 ;;
        --served-name)   SERVED_MODEL_NAME="$2"; shift 2 ;;
        --slice)         SLICE_ARG="$2";         shift 2 ;;
        --workers)       WORKERS="$2";           shift 2 ;;
        --step-limit)    STEP_LIMIT="$2";        shift 2 ;;
        --redo)          REDO_FLAG="--redo";     shift ;;
        --num-tasks)     NUM_TASKS="$2";         shift 2 ;;

        --sandbox-port)  SANDBOX_PORT="$2";      shift 2 ;;
        --harness-port)  HARNESS_PORT="$2";      shift 2 ;;
        --tag)           RUN_TAG="$2";           shift 2 ;;
        --skip-serve)    SKIP_SERVE=true;        shift ;;
        --skip-sandbox)  SKIP_SANDBOX=true;      shift ;;
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
[[ -n "${MODEL_PATH}" ]] && echo " Model     : ${MODEL_PATH##*/}  (scheme=${SCHEME}, max_len=${MAX_MODEL_LEN})"
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
