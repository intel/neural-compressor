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

    echo "=== [swebp] Create venv at ${SWEBP_VENV} (Python 3.10) ==="
    [[ -f "${SWEBP_VENV}/bin/python" ]] || \
        "${UV}" venv "${SWEBP_VENV}" --python 3.10

    echo "=== [swebp] Install packages ==="
    "${UV}" pip install --python "${SWEBP_VENV}" \
        -e "${AGENT_DIR}" \
        vllm swebench sb-cli "swe-rex>=1.4.0"

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
