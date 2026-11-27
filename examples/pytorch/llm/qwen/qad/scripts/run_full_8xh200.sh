#!/usr/bin/env bash
# Qwen3.8-27B full QAD, 8xH200, seq8192/gbs16, 4000 steps.
#
# Hyper-parameters match the formal run
#   q38-27b__mxfp4__qad-full__hf__h200x8__s8192__gbs16__1108-0747
# and live in recipes/qwen3_5_27b_mxfp4_fp8_full_fsdp2.yaml:
#   lr 2e-6 -> 2e-7, warmup 50, wd 0.1, beta2 0.95, clip 1.0, seed 42,
#   save every 500 steps, exact fp32 amax from the PTQ summary or single-rank calibration.
#
# Usage:
#   DATA=/path/tokenized_prefix SUMMARY=/path/.quant_summary.txt bash scripts/run_full_8xh200.sh
#   CONTAINER=qad-quant-h200 bash scripts/run_full_8xh200.sh   # run in container
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."   # neural-compressor repo root
export NCCL_NVLS_ENABLE=0                # NVLS bind fails on this fabric
export PYTHONPATH="${PYTHONPATH:-$PWD}"

MODEL=${MODEL:-Qwen/Qwen3.8-27B}
DATA=${DATA:-}
SUMMARY=${SUMMARY:-}
RECIPE=${RECIPE:-examples/pytorch/llm/qwen/qad/recipes/qwen3_5_27b_mxfp4_fp8_full_fsdp2.yaml}
OUT=${OUT:-runs/qwen3_8-27b__mxfp4__qad-full__hf__h200x8__s8192__gbs16}
PYTHON=${PYTHON:-python}
NPROC=${NPROC:-8}
CONTAINER=${CONTAINER:-}
STOP_AFTER=${STOP_AFTER:-0}
SAVE_STEPS=${SAVE_STEPS:-}

EXTRA=()
if [[ "$STOP_AFTER" -gt 0 ]]; then
  EXTRA+=(--max_steps "$STOP_AFTER")
fi
if [[ -n "$SAVE_STEPS" ]]; then
  if [[ "$SAVE_STEPS" -le 0 ]]; then
    EXTRA+=(--save_strategy no)
  else
    EXTRA+=(--save_strategy steps --save_steps "$SAVE_STEPS")
  fi
fi
if [[ -n "$DATA" ]]; then
  EXTRA+=(--data_path "$DATA")
fi
if [[ -n "$SUMMARY" ]]; then
  EXTRA+=(--amax_source "$SUMMARY")
fi

CMD=("$PYTHON" -m torch.distributed.run --nproc_per_node "$NPROC"
     examples/pytorch/llm/qwen/qad/train.py
     --config "$RECIPE"
     --model_name_or_path "$MODEL" --output_dir "$OUT" "${EXTRA[@]}")

GPU_ENV=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_ENV=(-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi
if [[ -n "$CONTAINER" ]]; then
  docker exec "${GPU_ENV[@]}" -e NCCL_NVLS_ENABLE=0 -e PYTHONPATH="$PWD" -w "$PWD" "$CONTAINER" "${CMD[@]}"
else
  "${CMD[@]}"
fi
