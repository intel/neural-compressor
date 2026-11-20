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
	bash setup_multimodal_bench.sh

Clone the pinned lmms-eval revision, apply the OpenAI-compatible API patch,
and install it into the active uv/Conda/Docker environment. The environment
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

readonly REPOSITORY="https://github.com/EvolvingLMMs-Lab/lmms-eval.git"
readonly LMMS_EVAL_DIR="${BENCHMARK_DIR}/lmms-eval"
readonly PATCH_FILE="${BENCHMARK_DIR}/patches/lmms_eval_qwen36_openai.patch"

if [[ ! -d "${LMMS_EVAL_DIR}/.git" ]]; then
	log "Cloning lmms-eval ${LMMS_EVAL_COMMIT}"
	git clone "${REPOSITORY}" "${LMMS_EVAL_DIR}"
	git -C "${LMMS_EVAL_DIR}" checkout "${LMMS_EVAL_COMMIT}"
else
	current_commit="$(git -C "${LMMS_EVAL_DIR}" rev-parse HEAD)"
	[[ "${current_commit}" == "${LMMS_EVAL_COMMIT}" ]] || \
		die "Existing lmms-eval checkout is not ${LMMS_EVAL_COMMIT}: ${LMMS_EVAL_DIR}"
	log "Using existing lmms-eval ${LMMS_EVAL_COMMIT} checkout"
fi

require_file "${PATCH_FILE}"
patch_options=(--unidiff-zero "${PATCH_FILE}")
if git -C "${LMMS_EVAL_DIR}" apply --reverse --check "${patch_options[@]}" 2>/dev/null; then
	log "OpenAI-compatible API patch is already applied"
elif git -C "${LMMS_EVAL_DIR}" apply --check "${patch_options[@]}" 2>/dev/null; then
	log "Applying OpenAI-compatible API patch"
	git -C "${LMMS_EVAL_DIR}" apply "${patch_options[@]}"
else
	die "Patch does not match the pinned lmms-eval checkout. Recreate it with: git -C ${LMMS_EVAL_DIR} diff --binary --unified=0 HEAD > ${PATCH_FILE}"
fi

log "Installing lmms-eval"
uv pip install -e "${LMMS_EVAL_DIR}"
log "Multimodal benchmark setup complete"
