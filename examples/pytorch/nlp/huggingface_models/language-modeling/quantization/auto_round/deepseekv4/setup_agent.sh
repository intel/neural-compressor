#!/bin/bash
# Usage: BENCHMARK_DIR=<path> bash setup_agent.sh [swebp|swe-verified|mcp-atlas|all]
#
# Run from within an already-activated Python environment (conda/venv).
# BENCHMARK_DIR defaults to $PWD

set -euo pipefail

BENCHMARK_DIR="${BENCHMARK_DIR:-$PWD}"
TASK="${1:?Usage: bash setup_agent.sh [swebp|swe-verified|mcp-atlas]}"

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

    echo "=== [swebp] Patch swebench.py for SWE-bench Pro Docker images ==="
    local patch_file="${BENCHMARK_DIR}/patches/swebench_pro_image.patch"
    if grep -q 'dockerhub_tag' "${AGENT_DIR}/src/minisweagent/run/extra/swebench.py" 2>/dev/null; then
        echo "  already patched"
    elif [[ -f "${patch_file}" ]]; then
        git -C "${AGENT_DIR}" apply "${patch_file}" && echo "  Patched OK"
    else
        die "Patch file not found: ${patch_file} — place patches/swebench_pro_image.patch in BENCHMARK_DIR"
    fi

    mkdir -p "${AGENT_DIR}/results"

    echo "=== [swebp] Install packages ==="
    pip install -e "${AGENT_DIR}" swebench sb-cli "swe-rex>=1.4.0"
    VLLM_USE_PRECOMPILED=1 pip install \
        git+https://github.com/xin3he/vllm-fork.git@support_deepseekv4_mxfp \
        --no-build-isolation

    echo "=== [swebp] Done — vllm: $(vllm --version) ==="
}

setup_swe_verified() {
    echo "=== [swe-verified] Clone mini-swe-agent (SWE-agent main) ==="
    if [[ ! -d "${AGENT_DIR_VERIFIED}/.git" ]]; then
        git clone --depth 1 https://github.com/SWE-agent/mini-swe-agent.git "${AGENT_DIR_VERIFIED}"
    else
        echo "  already cloned"
    fi

    echo "=== [swe-verified] Install packages ==="
    pip install -e "${AGENT_DIR_VERIFIED}" sb-cli "datasets>=3.0.0"
    VLLM_USE_PRECOMPILED=1 pip install \
        git+https://github.com/xin3he/vllm-fork.git@support_deepseekv4_mxfp \
        --no-build-isolation

    echo "=== [swe-verified] Done — vllm: $(vllm --version) ==="
}

setup_mcp() {
    echo "=== [mcp-atlas] Clone mcp-atlas ==="
    if [[ ! -d "${MCP_DIR}/.git" ]]; then
        git clone --depth 1 https://github.com/scaleapi/mcp-atlas.git "${MCP_DIR}"
    else
        echo "  already cloned"
    fi

    echo "=== [mcp-atlas] Install packages ==="
    pip install -r "${MCP_DIR}/requirements.txt"
    VLLM_USE_PRECOMPILED=1 pip install \
        git+https://github.com/xin3he/vllm-fork.git@support_deepseekv4_mxfp \
        --no-build-isolation

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

    echo "=== [mcp-atlas] Kill stale harness and sandbox ==="
    # Kill any harness on port 3001 (regardless of which dir it came from)
    local h_pid
    h_pid=$(lsof -ti:3001 2>/dev/null || true)
    if [[ -n "${h_pid}" ]]; then
        echo "  Killing harness PID=${h_pid}"
        kill ${h_pid} 2>/dev/null || true
        sleep 1
    else
        echo "  No harness running on port 3001"
    fi

    # Stop all MCP sandbox containers
    local s_pids
    s_pids=$(docker ps -q --filter ancestor=agent-environment:latest 2>/dev/null || true)
    if [[ -n "${s_pids}" ]]; then
        echo "  Stopping sandbox container(s): ${s_pids}"
        docker stop ${s_pids} 2>/dev/null || true
    else
        echo "  No sandbox containers running"
    fi

    echo "=== [mcp-atlas] Done — vllm: $(vllm --version) ==="
}

# =============================================================================
case "${TASK}" in
    swebp)        setup_swebp ;;
    swe-verified) setup_swe_verified ;;
    mcp-atlas)    setup_mcp ;;
    all)
        echo "[ERROR] Run setup per task in separate environments:"
        echo "  bash setup_agent.sh swebp"
        echo "  bash setup_agent.sh swe-verified"
        echo "  bash setup_agent.sh mcp-atlas"
        exit 1 ;;
    *)
        echo "Usage: $0 [swebp|swe-verified|mcp-atlas|all]"
        exit 1
        ;;
esac

echo ""
echo "Setup complete for: ${TASK}"
