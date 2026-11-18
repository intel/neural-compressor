#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$SCRIPT_DIR/scripts/run.py" "$@"
