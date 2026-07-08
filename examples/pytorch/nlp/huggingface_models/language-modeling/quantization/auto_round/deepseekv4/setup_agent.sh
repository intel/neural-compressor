#!/bin/bash
# Usage: BENCHMARK_DIR=<path> bash setup_agent.sh [swebp|swe-verified|mcp-atlas|all]
#
# Venvs are created at BENCHMARK_DIR level:
#   .venv-swebp   — SWE-bench Pro
#   .venv-swe     — SWE-bench Verified
#   .venv-mcp     — MCP-Atlas
#
# BENCHMARK_DIR defaults to $PWD

set -euo pipefail

BENCHMARK_DIR="${BENCHMARK_DIR:-$PWD}"
UV=$(command -v uv 2>/dev/null || echo "${HOME}/.local/bin/uv")
TASK="${1:-all}"

SWEBP_VENV="${BENCHMARK_DIR}/.venv-swebp"
SWE_VENV="${BENCHMARK_DIR}/.venv-swe"
MCP_VENV="${BENCHMARK_DIR}/.venv-mcp"

AGENT_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os/mini-swe-agent"   # submodule
AGENT_DIR_VERIFIED="${BENCHMARK_DIR}/mini-swe-agent"
MCP_DIR="${BENCHMARK_DIR}/mcp-atlas"

die() { echo "[ERROR] $*" >&2; exit 1; }

# =============================================================================
setup_swebp() {
    local swebp_dir="${BENCHMARK_DIR}/SWE-bench_Pro-os"

    echo "=== [swebp] Clone SWE-bench_Pro-os ==="
    if [[ ! -d "${swebp_dir}/.git" ]]; then
        git clone --depth 1 https://github.com/scaleapi/SWE-bench_Pro-os.git "${swebp_dir}"
    else
        echo "  already cloned"
    fi

    echo "=== [swebp] Init mini-swe-agent submodule ==="
    if [[ ! -f "${AGENT_DIR}/pyproject.toml" ]]; then
        git -C "${swebp_dir}" submodule update --init --depth 1 mini-swe-agent
    else
        echo "  submodule already initialised"
    fi

    echo "=== [swebp] Write run_swebp.py ==="
    mkdir -p "${AGENT_DIR}/data" "${AGENT_DIR}/config"
    cat > "${AGENT_DIR}/run_swebp.py" << 'PYEOF'
#!/usr/bin/env python3
"""Run mini-swe-agent on SWE-bench Pro instances."""
import json, concurrent.futures, time, argparse
from pathlib import Path
import yaml
from minisweagent.run.extra.swebench import process_instance
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from rich.live import Live

BASE_DIR       = Path(__file__).parent
INSTANCES_FILE = BASE_DIR / "data" / "swebench_pro_instances.json"
DEFAULT_CONFIG = BASE_DIR / "config" / "swebp_vllm.yaml"
DEFAULT_OUTPUT = BASE_DIR / "results" / "swebp_vllm"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", default=str(INSTANCES_FILE))
    parser.add_argument("--config",    default=str(DEFAULT_CONFIG))
    parser.add_argument("--output",    default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers",   type=int, default=2)
    parser.add_argument("--slice",     default="")
    parser.add_argument("--redo",      action="store_true")
    args = parser.parse_args()

    with open(args.instances) as f:
        instances = json.load(f)

    # inject docker image name (SWE-bench Pro uses jefzda registry)
    for inst in instances:
        inst["image_name"] = f"jefzda/sweap-images:{inst['dockerhub_tag']}"

    if args.slice:
        parts = [int(x) if x else None for x in args.slice.split(":")]
        instances = instances[slice(*parts)]

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.redo and (output_path / "preds.json").exists():
        done = set(json.loads((output_path / "preds.json").read_text()).keys())
        instances = [i for i in instances if i["instance_id"] not in done]

    if not instances:
        print("All instances already completed. Use --redo to re-run.")
        return

    print(f"Running {len(instances)} instances with {args.workers} workers...")
    print(f"Output: {output_path}")

    progress = RunBatchProgressManager(
        len(instances),
        output_path / f"exit_statuses_{int(time.time())}.yaml"
    )
    with Live(progress.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(process_instance, inst, output_path, config, progress): inst["instance_id"]
                for inst in instances
            }
            for future in concurrent.futures.as_completed(futures):
                iid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\n[ERROR] {iid}: {e}")

    print(f"\nDone. Predictions saved to {output_path / 'preds.json'}")

if __name__ == "__main__":
    main()
PYEOF

    echo "=== [swebp] Write config/swebp_vllm.yaml ==="
    cat > "${AGENT_DIR}/config/swebp_vllm.yaml" << 'YAMEOF'
# SWE-bench Pro base config (model/api_base/step_limit overridden at runtime)
agent:
  system_template: |
    You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
    Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

    Include a THOUGHT section before your command where you explain your reasoning process.
    Format your response as shown in <format_example>.

    <format_example>
    THOUGHT: Your reasoning and analysis here

    ```bash
    your_command_here
    ```
    </format_example>

    Failure to follow these rules will cause your response to be rejected.
  instance_template: |
    <pr_description>
    Consider the following PR description:
    {{task}}
    </pr_description>

    <instructions>
    Your task is to make changes to non-test files in /app to fix the issue described above.
    Recommended workflow: explore → reproduce → fix → verify.
    When done, submit with:
    ```bash
    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
    ```
    </instructions>
  action_observation_template: |
    <returncode>{{output.returncode}}</returncode>
    {% if output.output | length < 10000 -%}
    <output>
    {{ output.output -}}
    </output>
    {%- else -%}
    <warning>Output too long. Use head/tail/grep to reduce.</warning>
    {%- set elided_chars = output.output | length - 10000 -%}
    <output_head>{{ output.output[:5000] }}</output_head>
    <elided_chars>{{ elided_chars }} characters elided</elided_chars>
    <output_tail>{{ output.output[-5000:] }}</output_tail>
    {%- endif -%}
  format_error_template: |
    Please provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.
  step_limit: 100
  cost_limit: 0.0

environment:
  cwd: /app
  timeout: 120
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: 'off'
    TQDM_DISABLE: '1'
  environment_class: docker

model:
  model_name: openai/gpt-3.5-turbo
  model_kwargs:
    api_base: http://localhost:8888/v1
    api_key: dummy
    drop_params: true
    temperature: 0.0
  cost_tracking: ignore_errors
YAMEOF

    echo "=== [swebp] Download instance data from HuggingFace ==="
    # 先装好venv再下载，否则datasets还没有
    echo "=== [swebp] Create venv at ${SWEBP_VENV} (Python 3.10) ==="
    [[ -f "${SWEBP_VENV}/bin/python" ]] || \
        "${UV}" venv "${SWEBP_VENV}" --python 3.10

    echo "=== [swebp] Install packages ==="
    "${UV}" pip install --python "${SWEBP_VENV}" \
        -e "${AGENT_DIR}" \
        vllm swebench sb-cli "swe-rex>=1.4.0"

    echo "=== [swebp] Download instance data from HuggingFace ==="
    local full_json="${AGENT_DIR}/data/swebench_pro_instances.json"
    local mini_json="${AGENT_DIR}/data/swebp_10instances.json"
    if [[ ! -f "${full_json}" ]]; then
        "${SWEBP_VENV}/bin/python3" -c "
import json
from datasets import load_dataset
print('Downloading ScaleAI/SWE-bench_Pro ...')
ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
data = [dict(r) for r in ds]
with open('${full_json}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Wrote {len(data)} instances -> ${full_json}')
with open('${mini_json}', 'w') as f:
    json.dump(data[:10], f, indent=2)
print(f'Wrote 10 instances  -> ${mini_json}')
"
    else
        echo "  instance data already present"
    fi

    echo "=== [swebp] Done — vllm: $("${SWEBP_VENV}/bin/vllm" --version) ==="
}

setup_swe_verified() {
    echo "=== [swe-verified] Clone mini-swe-agent (SWE-agent main) ==="
    if [[ ! -d "${AGENT_DIR_VERIFIED}/.git" ]]; then
        git clone --depth 1 https://github.com/SWE-agent/mini-swe-agent.git "${AGENT_DIR_VERIFIED}"
    else
        echo "  already cloned"
    fi

    echo "=== [swe-verified] Create venv at ${SWE_VENV} (Python 3.10) ==="
    [[ -f "${SWE_VENV}/bin/python" ]] || \
        "${UV}" venv "${SWE_VENV}" --python 3.10

    echo "=== [swe-verified] Install packages ==="
    "${UV}" pip install --python "${SWE_VENV}" \
        -e "${AGENT_DIR_VERIFIED}" \
        vllm sb-cli "datasets>=3.0.0"

    echo "=== [swe-verified] Done — vllm: $("${SWE_VENV}/bin/vllm" --version) ==="
}

setup_mcp() {
    echo "=== [mcp-atlas] Clone mcp-atlas ==="
    if [[ ! -d "${MCP_DIR}/.git" ]]; then
        git clone --depth 1 https://github.com/scaleapi/mcp-atlas.git "${MCP_DIR}"
    else
        echo "  already cloned"
    fi

    echo "=== [mcp-atlas] Create venv at ${MCP_VENV} (Python 3.10) ==="
    [[ -f "${MCP_VENV}/bin/python" ]] || \
        "${UV}" venv "${MCP_VENV}" --python 3.10

    echo "=== [mcp-atlas] Install packages ==="
    "${UV}" pip install --python "${MCP_VENV}" \
        -r "${MCP_DIR}/requirements.txt" \
        vllm

    echo "=== [mcp-atlas] npm install ==="
    export NVM_DIR="${HOME}/.nvm"
    [[ -s "${NVM_DIR}/nvm.sh" ]] && source "${NVM_DIR}/nvm.sh"
    npm install --prefix "${MCP_DIR}/services/agent-harness" --silent

    echo "=== [mcp-atlas] Docker image ==="
    if ! docker image inspect agent-environment:latest &>/dev/null; then
        docker pull ghcr.io/scaleapi/mcp-atlas:1.2.5
        docker tag  ghcr.io/scaleapi/mcp-atlas:1.2.5 agent-environment:latest
    else
        echo "  agent-environment:latest already present"
    fi

    echo "=== [mcp-atlas] Done — vllm: $("${MCP_VENV}/bin/vllm" --version) ==="
}

# =============================================================================
case "${TASK}" in
    swebp)        setup_swebp ;;
    swe-verified) setup_swe_verified ;;
    mcp-atlas)    setup_mcp ;;
    all)
        setup_swebp
        setup_swe_verified
        setup_mcp
        ;;
    *)
        echo "Usage: $0 [swebp|swe-verified|mcp-atlas|all]"
        exit 1
        ;;
esac

echo ""
echo "Setup complete for: ${TASK}"
