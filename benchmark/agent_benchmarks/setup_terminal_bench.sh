#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=versions.env
source "${SCRIPT_DIR}/versions.env"

usage() {
	cat <<'EOF'
Usage:
	bash setup_terminal_bench.sh

Install the pinned Harbor release into the active uv/Conda/Docker environment.
The environment must already provide Python and support `uv pip install`.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi
[[ $# -eq 0 ]] || die "Unknown argument: $1"

require_command uv
log "Installing Harbor ${HARBOR_VERSION}"
uv pip install "harbor==${HARBOR_VERSION}"
log "Terminal-Bench setup complete"