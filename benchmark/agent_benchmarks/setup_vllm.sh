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
        '  bash setup_vllm.sh' \
        '' \
        'Install vLLM into the current uv/Conda/Docker environment. The environment' \
        'must already provide Python and support `uv pip install`.' \
        '' \
        'Options:' \
        '  -h, --help          Show this help message.'
}

install_standard_vllm() {
    log "Installing standard vLLM ${VLLM_VERSION} from PyPI"
    uv pip install "vllm==${VLLM_VERSION}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
[[ $# -eq 0 ]] || die "Unknown argument: $1"

require_command uv
install_standard_vllm

log "Setup complete"
