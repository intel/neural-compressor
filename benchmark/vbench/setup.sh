#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

case "${1:-}" in
  -h|--help)
    echo "Usage: bash benchmark/vbench/setup.sh [--nvfp4_e5m3]"
    echo "Create or reuse venv/vbench. Add --nvfp4_e5m3 to install the NVFP4 E5M3 plugin."
    exit 0
    ;;
  ""|--nvfp4_e5m3) ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: bash benchmark/vbench/setup.sh [--nvfp4_e5m3]" >&2
  exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
env_dir="$repo_dir/venv/vbench"
python="$env_dir/bin/python"

if [[ ! -e "$env_dir" ]]; then
  uv venv --seed --python 3.12 "$env_dir"
elif [[ ! -f "$env_dir/pyvenv.cfg" || ! -x "$python" ]]; then
  echo "Expected a Python virtual environment at $env_dir" >&2
  exit 1
fi
"$python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "VBench needs Python 3.12")'

uv pip install --python "$python" --torch-backend=cu130 \
  vllm==0.32.0 vllm-omni==0.32.0 ninja nvidia-cutlass-dsl==4.8.0 \
  dreamsim==0.2.1 pyiqa==0.1.16 timm==1.0.12 \
  boto3 cython decord gdown==4.7.3 easydict fairscale fvcore lvis matplotlib openai-clip \
  opencv-python pycocoevalcap scikit-learn scikit-image tensorboard wheel
# VBench pins older Transformers and NumPy versions. Its metric dependencies are installed above.
uv pip install --python "$python" --no-deps vbench==0.1.5

if [[ "${1:-}" == --nvfp4_e5m3 ]]; then
  uv pip install --python "$python" --no-deps -e "$repo_dir/benchmark/vllm-qdq-plugin"
fi

printf 'Activate with: source %q\n' "$env_dir/bin/activate"
