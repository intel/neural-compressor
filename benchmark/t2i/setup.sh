#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

stage=${1:-all}
case "$stage" in
  -h|--help)
    echo "Usage: bash benchmark/t2i/setup.sh [all|generate|evaluate]"
    exit 0
    ;;
  all|generate|evaluate) ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: bash benchmark/t2i/setup.sh [all|generate|evaluate]" >&2
  exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

ensure_venv() {
  local directory=$1
  if [[ ! -e "$directory" ]]; then
    uv venv --seed --python 3.12 "$directory"
  fi
  if [[ ! -f "$directory/pyvenv.cfg" || ! -x "$directory/bin/python" ]]; then
    echo "Expected a Python virtual environment at $directory" >&2
    exit 1
  fi
  "$directory/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "T2I needs Python 3.12")'
}

if [[ "$stage" != evaluate ]]; then
  env_dir="$repo_dir/venv/t2i"
  ensure_venv "$env_dir"
  uv pip install --python "$env_dir/bin/python" --torch-backend=cu130 \
    vllm==0.32.0 vllm-omni==0.32.0 transformers==5.19.0 diffusers==0.40.0 \
    ninja nvidia-cutlass-dsl==4.8.0
fi

if [[ "$stage" != generate ]]; then
  env_dir="$repo_dir/venv/t2i-eval"
  ensure_venv "$env_dir"
  uv pip install --python "$env_dir/bin/python" --torch-backend=cu130 \
    torch==2.13.0 torchvision==0.28.0 transformers==4.44.2 diffusers==0.29.2 \
    huggingface-hub==0.34.4 datasets==3.6.0 accelerate==0.34.2 \
    image-reward==1.5 openai-clip 'setuptools<81'
fi
