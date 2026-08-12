#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=versions.env
source "${SCRIPT_DIR}/versions.env"

usage() {
    printf '%s\n' \
        'Usage:' \
        '  bash setup_vllm.sh [OPTIONS]' \
        '' \
        'Install vLLM into the current uv/Conda/Docker environment. The environment' \
        'must already provide Python and support `uv pip install`. A model containing' \
        '"deepseek-v4" or "deepseek_v4" selects the DeepSeek-V4-compatible fork;' \
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
    log "Installing standard vLLM ${VLLM_VERSION} from PyPI"
    uv pip install "vllm==${VLLM_VERSION}"
}

install_deepseek_v4_vllm() {
    log "Installing DeepSeek-V4 vLLM stack"
    uv pip install -U pip setuptools_rust setuptools_scm
    uv pip install git+https://github.com/intel/auto-round.git@main
    uv pip install compressed-tensors --no-deps
    pip install git+https://github.com/deepseek-ai/DeepGEMM.git@891d57b4db1071624b5c8fa0d1e51cb317fa709f \
        --no-build-isolation
    VLLM_USE_PRECOMPILED=1 uv pip install git+https://github.com/xin3he/vllm.git@support_deepseekv4_mxfp \
        --no-build-isolation
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

if is_deepseek_v4_model "${MODEL}"; then
    require_command curl
    require_command git
    install_deepseek_v4_vllm
else
    install_standard_vllm
fi

log "Setup complete"
