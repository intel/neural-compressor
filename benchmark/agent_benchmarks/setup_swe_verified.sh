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
	bash setup_swe_verified.sh

Clone mini-SWE-agent, apply per-instance Docker cleanup, and install the
SWE-bench Verified dependencies into the current uv/Conda/Docker environment.
The environment must already provide Python and support `uv pip install`.
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

readonly REPOSITORY="https://github.com/SWE-agent/mini-swe-agent.git"
readonly PATCH_FILE="${BENCHMARK_DIR}/patches/swebench_verified_per_instance_cleanup.patch"

if [[ ! -d "${AGENT_DIR_VERIFIED}/.git" ]]; then
	log "Cloning mini-SWE-agent ${MINI_SWE_AGENT_VERSION}"
	git clone --depth 1 --branch "v${MINI_SWE_AGENT_VERSION}" \
		"${REPOSITORY}" "${AGENT_DIR_VERIFIED}"
else
	git -C "${AGENT_DIR_VERIFIED}" fetch --depth 1 origin \
		"refs/tags/v${MINI_SWE_AGENT_VERSION}:refs/tags/v${MINI_SWE_AGENT_VERSION}"
	expected_commit="$(git -C "${AGENT_DIR_VERIFIED}" rev-parse "v${MINI_SWE_AGENT_VERSION}^{commit}")"
	current_commit="$(git -C "${AGENT_DIR_VERIFIED}" rev-parse HEAD)"
	[[ "${current_commit}" == "${expected_commit}" ]] || \
		die "Existing mini-SWE-agent is not v${MINI_SWE_AGENT_VERSION}: ${AGENT_DIR_VERIFIED}"
	log "Using existing mini-SWE-agent ${MINI_SWE_AGENT_VERSION} checkout"
fi

require_file "${PATCH_FILE}"
if grep -q 'remove_image_on_cleanup' \
		"${AGENT_DIR_VERIFIED}/src/minisweagent/environments/docker.py"; then
	log "Per-instance Docker cleanup patch is already applied"
else
	log "Applying per-instance Docker cleanup patch"
	git -C "${AGENT_DIR_VERIFIED}" apply "${PATCH_FILE}"
fi

log "Installing mini-SWE-agent for inference and SWE-bench for local evaluation"
uv pip install "${AGENT_DIR_VERIFIED}" swebench "datasets>=3.0.0"

log "SWE-bench Verified setup complete"
