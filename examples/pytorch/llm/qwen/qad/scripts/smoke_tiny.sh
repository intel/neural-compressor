#!/usr/bin/env bash
# 1-GPU wiring smoke: tiny model, synthetic tokens, 2 steps.
# Validates quantize -> QADTrainer -> logging without touching the 27B recipe.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
export PYTHONPATH="${PYTHONPATH:-$PWD}"

OUT=${OUT:-runs/qad-smoke-tiny}
PYTHON=${PYTHON:-python}
CONTAINER=${CONTAINER:-}

CMD=("$PYTHON" examples/pytorch/llm/qwen/qad/main.py
     --recipe examples/pytorch/llm/qwen/qad/recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn.yaml
     --output "$OUT" --tiny --synthetic-batches 64
     --seq-length 128 --train-iters 2 --stop-after 2 --save-steps 0 --top-k 64)

GPU_ENV=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_ENV=(-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi
if [[ -n "$CONTAINER" ]]; then
  docker exec "${GPU_ENV[@]}" -e PYTHONPATH="$PWD" -w "$PWD" "$CONTAINER" "${CMD[@]}"
else
  "${CMD[@]}"
fi
