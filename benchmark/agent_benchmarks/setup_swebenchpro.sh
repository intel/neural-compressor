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
	bash setup_swebenchpro.sh

Clone the pinned SWE-bench Pro repository and mini-SWE-agent submodule, apply
Docker image/cleanup support, and install generation and local-evaluation
dependencies into the current uv/Conda/Docker environment. The environment
must already provide Python and support `uv pip install`.
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
readonly IMAGE_PATCH="${BENCHMARK_DIR}/patches/swebench_pro_image.patch"

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

log "Initializing the pinned mini-SWE-agent submodule"
git -C "${SWEBENCH_PRO_DIR}" submodule update --init --depth 1 mini-swe-agent
submodule_commit="$(git -C "${AGENT_DIR}" rev-parse HEAD)"
[[ "${submodule_commit}" == "${SWEBENCH_PRO_MINI_SWE_AGENT_COMMIT}" ]] || \
	die "Unexpected mini-SWE-agent submodule commit: ${submodule_commit}"

require_file "${IMAGE_PATCH}"
if git -C "${AGENT_DIR}" apply --reverse --check "${IMAGE_PATCH}" 2>/dev/null; then
	log "SWE-bench Pro Docker image and cleanup patch is already applied"
elif git -C "${AGENT_DIR}" apply --check "${IMAGE_PATCH}" 2>/dev/null; then
	log "Applying SWE-bench Pro Docker image and cleanup patch"
	git -C "${AGENT_DIR}" apply "${IMAGE_PATCH}"
else
	die "Docker image patch does not match the mini-SWE-agent checkout: ${IMAGE_PATCH}"
fi

log "Installing mini-SWE-agent and SWE-bench Pro local-evaluation dependencies"
uv pip install "${AGENT_DIR}" -r "${SWEBENCH_PRO_DIR}/requirements.txt" "swe-rex>=1.4.0"

mkdir -p "${AGENT_DIR}/results"
log "SWE-bench Pro setup complete"
