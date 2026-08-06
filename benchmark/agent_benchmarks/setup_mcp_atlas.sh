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
	bash setup_mcp_atlas.sh

Clone the pinned MCP-Atlas repository, install its Python and TypeScript
dependencies, create .env from env.template when needed, and pull the pinned
prebuilt MCP sandbox image. Run this inside the intended Python environment.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi
[[ $# -eq 0 ]] || die "Unknown argument: $1"

require_command git
require_command uv
require_command docker
init_benchmark_paths

readonly REPOSITORY="https://github.com/scaleapi/mcp-atlas.git"
readonly NODE_RUNTIME_DIR="${BENCHMARK_DIR}/.tools/node-v${MCP_ATLAS_NODE_VERSION}"

node_major=0
if command -v node >/dev/null 2>&1; then
	node_major="$(node -p 'Number(process.versions.node.split(".")[0])' 2>/dev/null || printf '0')"
fi
if ((node_major < 20)); then
	require_command curl
	require_command tar
	case "$(uname -m)" in
		x86_64) node_arch="x64" ;;
		aarch64 | arm64) node_arch="arm64" ;;
		*) die "Unsupported architecture for bundled Node.js: $(uname -m)" ;;
	esac
	if [[ ! -x "${NODE_RUNTIME_DIR}/bin/node" ]]; then
		archive="$(mktemp --suffix=.tar.xz)"
		trap 'rm -f -- "${archive:-}"' EXIT
		node_url="https://nodejs.org/dist/v${MCP_ATLAS_NODE_VERSION}/node-v${MCP_ATLAS_NODE_VERSION}-linux-${node_arch}.tar.xz"
		log "Downloading Node.js ${MCP_ATLAS_NODE_VERSION} from ${node_url}"
		curl -fL --retry 3 -o "${archive}" "${node_url}"
		mkdir -p "${NODE_RUNTIME_DIR}"
		tar -xJf "${archive}" --strip-components=1 -C "${NODE_RUNTIME_DIR}"
		rm -f -- "${archive}"
		trap - EXIT
	fi
	export PATH="${NODE_RUNTIME_DIR}/bin:${PATH}"
fi
require_command node
require_command npm
node_major="$(node -p 'Number(process.versions.node.split(".")[0])')"
((node_major >= 20)) || die "MCP-Atlas requires Node.js 20 or newer; found $(node --version)"
log "Using Node.js $(node --version) from $(command -v node)"

if [[ ! -d "${MCP_DIR}/.git" ]]; then
	log "Cloning MCP-Atlas at ${MCP_ATLAS_COMMIT}"
	git clone --filter=blob:none --no-checkout "${REPOSITORY}" "${MCP_DIR}"
	git -C "${MCP_DIR}" fetch --depth 1 origin "${MCP_ATLAS_COMMIT}"
	git -C "${MCP_DIR}" checkout --detach "${MCP_ATLAS_COMMIT}"
else
	current_commit="$(git -C "${MCP_DIR}" rev-parse HEAD)"
	[[ "${current_commit}" == "${MCP_ATLAS_COMMIT}" ]] || \
		die "Existing MCP-Atlas checkout is not ${MCP_ATLAS_COMMIT}: ${MCP_DIR}"
	log "Using existing MCP-Atlas checkout at ${MCP_ATLAS_COMMIT}"
fi

if [[ ! -f "${MCP_DIR}/.env" ]]; then
	cp "${MCP_DIR}/env.template" "${MCP_DIR}/.env"
	log "Created ${MCP_DIR}/.env; add optional MCP-server API keys there"
else
	log "Preserving existing MCP-Atlas .env"
fi

log "Installing MCP-Atlas Python dependencies"
uv pip install -r "${MCP_DIR}/requirements.txt"

log "Installing and building the TypeScript agent harness"
npm install --prefix "${MCP_DIR}/services/agent-harness" --silent
npm run build --prefix "${MCP_DIR}/services/agent-harness"

if docker image inspect "${MCP_ATLAS_IMAGE}" >/dev/null 2>&1; then
	log "Using existing sandbox image ${MCP_ATLAS_IMAGE}"
else
	log "Pulling sandbox image ${MCP_ATLAS_IMAGE}"
	docker pull "${MCP_ATLAS_IMAGE}"
fi
docker tag "${MCP_ATLAS_IMAGE}" agent-environment:latest

log "MCP-Atlas setup complete"