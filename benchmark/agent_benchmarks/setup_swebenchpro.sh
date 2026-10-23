#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=versions.env
source "${SCRIPT_DIR}/versions.env"

usage() {
	cat <<'EOF'
Usage:
	bash setup_swebenchpro.sh

Clone the pinned SWE-bench Pro V2 task tree and locked-protocol tooling.
Install Harbor with Docker and optional Modal support into the activated
Python 3.12+ environment using uv. V1 checkouts must be moved aside first.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi
[[ $# -eq 0 ]] || die "Unknown argument: $1"

require_command git
require_command uv
init_benchmark_paths

readonly REPOSITORY="https://github.com/scaleapi/SWE-bench_Pro-os.git"
readonly SWEBENCH_PRO_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os"

if [[ ! -d "${SWEBENCH_PRO_DIR}/.git" ]]; then
	log "Cloning SWE-bench Pro at ${SWEBENCH_PRO_COMMIT}"
	git clone --filter=blob:none --no-checkout "${REPOSITORY}" "${SWEBENCH_PRO_DIR}"
	git -C "${SWEBENCH_PRO_DIR}" fetch --depth 1 origin "${SWEBENCH_PRO_COMMIT}"
	git -C "${SWEBENCH_PRO_DIR}" checkout --detach "${SWEBENCH_PRO_COMMIT}"
else
	current_commit="$(git -C "${SWEBENCH_PRO_DIR}" rev-parse HEAD)"
	[[ "${current_commit}" == "${SWEBENCH_PRO_COMMIT}" ]] || \
		die "Existing SWE-bench Pro checkout is not ${SWEBENCH_PRO_COMMIT}: ${SWEBENCH_PRO_DIR}"
	log "Using existing SWE-bench Pro checkout at ${SWEBENCH_PRO_COMMIT}"
fi

require_file "${SWEBENCH_PRO_DIR}/v2/tooling/locked_mini_swe.py"
require_file "${SWEBENCH_PRO_DIR}/v2/tooling/patch_replay.py"
log "Verifying the pinned V2 task tree and tooling"
(cd "${SWEBENCH_PRO_DIR}/v2" && sha256sum --check --quiet SHA256SUMS)

log "Installing Harbor (requires Python 3.12+)"
uv pip install --python "${PYTHON_EXECUTABLE:-python}" "harbor[modal]==${SWEBENCH_PRO_HARBOR_VERSION}" "PyYAML>=6"
log "SWE-bench Pro V2 setup complete; Docker Compose is required for local runs"
