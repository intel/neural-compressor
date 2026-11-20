#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$here/../.." && pwd)

case "${1:-}" in
  prepare|generate)
    python=${T2I_GENERATE_PYTHON:-"$repo_dir/venv/t2i/bin/python"}
    entry=benchmark.py
    ;;
  evaluate|summarize)
    python=${T2I_EVALUATE_PYTHON:-"$repo_dir/venv/t2i-eval/bin/python"}
    entry=evaluate.py
    ;;
  -h|--help)
    echo "Usage: bash benchmark/t2i/run_t2i.sh {prepare|generate|evaluate|summarize} [options]"
    exit 0
    ;;
  *)
    echo "Expected prepare, generate, evaluate, or summarize. Use --help for usage." >&2
    exit 2
    ;;
esac

if [[ ! -x "$python" ]]; then
  echo "Python interpreter $python is unavailable. Run bash benchmark/t2i/setup.sh first." >&2
  exit 1
fi

exec "$python" "$here/$entry" "$@"
