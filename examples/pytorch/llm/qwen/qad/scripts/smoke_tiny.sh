#!/usr/bin/env bash
# 1-GPU wiring smoke: tiny model, synthetic tokens, 2 steps.
# Validates quantize -> QADTrainer -> logging without touching the 27B recipe.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
export PYTHONPATH="${PYTHONPATH:-$PWD}"

OUT=${OUT:-runs/qad-smoke-tiny}
PYTHON=${PYTHON:-python}
CONTAINER=${CONTAINER:-}

CMD=("$PYTHON" examples/pytorch/llm/qwen/qad/train.py
     --config examples/pytorch/llm/qwen/qad/recipes/qwen3_5_27b_mxfp4_fp8_full_fsdp2.yaml
     --output_dir "$OUT" --tiny true --synthetic_batches 64 --calib_batches 2
     --seq_length 128 --max_steps 2 --save_strategy no --kd_top_k 64 --teacher_placement cuda)

GPU_ENV=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_ENV=(-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi
if [[ -n "$CONTAINER" ]]; then
  docker exec "${GPU_ENV[@]}" -e PYTHONPATH="$PWD" -w "$PWD" "$CONTAINER" "${CMD[@]}"
else
  "${CMD[@]}"
fi
