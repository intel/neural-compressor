#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task="${TASKS:-}"

usage() {
  cat <<'USAGE'
Usage: bash setup.sh --task t2v|i2v|s2v
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      task="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$task" ]]; then
  echo "Error: --task is required"
  usage
  exit 1
fi

uv pip install -U pip setuptools

if [[ "$task" == "s2v" ]]; then
  req_file="${SCRIPT_DIR}/requirements_s2v.txt"
elif [[ "$task" == "t2v" || "$task" == "i2v" ]]; then
  req_file="${SCRIPT_DIR}/requirements_i2v_t2v.txt"
else
  echo "Error: unsupported task: $task"
  usage
  exit 1
fi

if [[ "$task" == "s2v" ]]; then
  uv pip install --no-cache-dir \
    torch \
    torchvision \
    transformers \
    accelerate \
    huggingface_hub \
    safetensors
  # flash-attn needs torch available at build time.
  uv pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3.post1
  uv pip install --no-cache-dir neural-compressor-pt auto-round
  uv pip install --no-cache-dir -r "$req_file"
else
  uv pip install --no-cache-dir -r "$req_file"
fi

uv pip install --no-cache-dir  opencv-python-headless==4.10.0.84

if [[ "$task" == "t2v" || "$task" == "i2v" ]]; then
  uv pip install --no-cache-dir VBench --no-deps
fi

echo "Setup completed for task: $task"
