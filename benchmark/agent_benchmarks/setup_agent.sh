#!/bin/bash
# Usage: BENCHMARK_DIR=<path> MODEL_NAME=<name> bash setup_agent.sh [swebp|swe-verified|mcp-atlas|all] [model_name]
#
# Run from within an already-activated Python environment (conda/venv).
# BENCHMARK_DIR defaults to $PWD
# MODEL_NAME may also be passed as the second positional argument.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

TASK="${1:?Usage: bash setup_agent.sh [swebp|swe-verified|mcp-atlas]}"
MODEL_NAME="${MODEL_NAME:-${2:-}}"

init_benchmark_paths

is_deepseek_v4_model() {
    [[ "${MODEL_NAME}" == *DeepSeek-V4* ]]
}

install_deepseek_v4_prereqs() {
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
    uv pip install -U pip setuptools_rust setuptools_scm
}

align_cuda_toolchain() {
    uv pip install -U         nvidia-cuda-nvcc==13.2.86         nvidia-nvvm==13.2.86         nvidia-cuda-runtime==13.2.86         nvidia-cuda-crt==13.2.86         nvidia-cuda-cccl==13.2.86
}

install_vllm() {
    if is_deepseek_v4_model; then
        echo "=== [vllm] Installing DeepSeek-V4-specific stack for MODEL_NAME=${MODEL_NAME} ==="
        install_deepseek_v4_prereqs
        align_cuda_toolchain
        VLLM_USE_PRECOMPILED=1 uv pip install \
            git+https://github.com/xin3he/vllm.git@support_deepseekv4_mxfp \
            --no-build-isolation
        align_cuda_toolchain
    else
        if [[ -n "${MODEL_NAME}" ]]; then
            echo "=== [vllm] Installing generic stack for MODEL_NAME=${MODEL_NAME} ==="
        else
            echo "=== [vllm] Installing generic stack (MODEL_NAME not set) ==="
        fi
        uv pip install -U torch torchvision torchaudio vllm
    fi
}

verify_cutlass_runtime() {
    python - <<"PYCUT"
import sys
try:
    import cutlass.cute.nvgpu.common as c
    if not hasattr(c, "normalize_field_to_ir_name"):
        raise RuntimeError("cutlass symbol normalize_field_to_ir_name missing")
    print("[cutlass] runtime check OK")
except Exception as e:
    print(f"[cutlass] runtime check FAILED: {e}")
    sys.exit(1)
PYCUT
}

repair_cutlass_runtime() {
    echo "=== [mcp-atlas] Repairing CUTLASS runtime ==="
    pip install --no-cache-dir --force-reinstall \
        nvidia-cutlass-dsl==4.5.2 \
        nvidia-cutlass-dsl-libs-base==4.5.2 \
        nvidia-cutlass-dsl-libs-cu13==4.5.2
    pip install --no-cache-dir numpy==2.3.5
}

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
        git -C "${swebp_dir}" submodule update --init --recursive --depth 1 mini-swe-agent
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

    echo "=== [swebp] Patch per-instance cleanup (container + image) ==="
    local cleanup_patch="${BENCHMARK_DIR}/patches/swebench_pro_per_instance_cleanup.patch"
    if grep -q 'remove_image_on_cleanup' "${AGENT_DIR}/src/minisweagent/environments/docker.py" 2>/dev/null; then
        echo "  already patched"
    elif [[ -f "${cleanup_patch}" ]]; then
        git -C "${AGENT_DIR}" apply "${cleanup_patch}" && echo "  Patched OK"
    else
        die "Patch file not found: ${cleanup_patch} — place patches/swebench_pro_per_instance_cleanup.patch in BENCHMARK_DIR"
    fi

    mkdir -p "${AGENT_DIR}/results"

    echo "=== [swebp] Install packages ==="
    uv pip install "${AGENT_DIR}" swebench sb-cli "swe-rex>=1.4.0"
    install_vllm

    echo "=== [swebp] Done — vllm: $(vllm --version) ==="
}

setup_swe_verified() {
    echo "=== [swe-verified] Clone mini-swe-agent (SWE-agent main) ==="
    if [[ ! -d "${AGENT_DIR_VERIFIED}/.git" ]]; then
        git clone --depth 1 https://github.com/SWE-agent/mini-swe-agent.git "${AGENT_DIR_VERIFIED}"
    else
        echo "  already cloned"
    fi

    echo "=== [swe-verified] Patch per-instance cleanup (container + image) ==="
    local cleanup_patch_verified="${BENCHMARK_DIR}/patches/swebench_verified_per_instance_cleanup.patch"
    if grep -q 'remove_image_on_cleanup' "${AGENT_DIR_VERIFIED}/src/minisweagent/environments/docker.py" 2>/dev/null; then
        echo "  already patched"
    elif [[ -f "${cleanup_patch_verified}" ]]; then
        git -C "${AGENT_DIR_VERIFIED}" apply "${cleanup_patch_verified}" && echo "  Patched OK"
    else
        die "Patch file not found: ${cleanup_patch_verified} — place patches/swebench_verified_per_instance_cleanup.patch in BENCHMARK_DIR"
    fi

    echo "=== [swe-verified] Install packages ==="
    uv pip install "${AGENT_DIR_VERIFIED}" sb-cli "datasets>=3.0.0"
    install_vllm

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
    uv pip install -r "${MCP_DIR}/requirements.txt"
    install_vllm

    if ! verify_cutlass_runtime; then
        repair_cutlass_runtime
        verify_cutlass_runtime
    fi

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
