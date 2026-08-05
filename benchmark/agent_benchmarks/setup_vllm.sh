#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
    printf '%s\n' \
        'Usage:' \
        '  bash setup_vllm.sh [OPTIONS]' \
        '' \
        'Install vLLM into the current uv/Conda/Docker environment. The environment' \
        'must already provide Python and support `uv pip install`. A model containing' \
        '"deepseek-v4" or "deepseek_v4" selects the pinned DeepSeek-V4 build;' \
        'otherwise the pinned standard PyPI release is installed.' \
        '' \
        'Options:' \
        '  --model MODEL       Model name or path used to select the installation stack.' \
        '  -h, --help          Show this help message.' \
        '' \
        'Examples:' \
        '  bash setup_vllm.sh' \
        '  bash setup_vllm.sh --model DeepSeek-V4-Flash'
}

install_standard_vllm() {
    log "Installing standard vLLM 0.26.0 from PyPI"
    uv pip install "vllm==0.26.0"
}

install_deepseek_v4_vllm() {
    log "Installing pinned DeepSeek-V4 vLLM stack"
    uv pip install -U pip setuptools_rust setuptools_scm
    uv pip install -U evalscope lm_eval[api] lm-eval["ruler"] transformers datasets
    uv pip install git+https://github.com/intel/auto-round.git@main
    uv pip install compressed-tensors --no-deps
    bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh)
    VLLM_USE_PRECOMPILED=1 uv pip install git+https://github.com/xin3he/vllm.git@support_deepseekv4_mxfp --no-build-isolation
}

MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            MODEL="$2"
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

require_command uv

MODEL_LOWER="${MODEL,,}"
if [[ "${MODEL_LOWER}" == *deepseek-v4* || "${MODEL_LOWER}" == *deepseek_v4* ]]; then
    require_command curl
    require_command git
    install_deepseek_v4_vllm
else
    install_standard_vllm
fi

log "Setup complete"
