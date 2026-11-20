#!/usr/bin/env bash
# Qwen3.8-27B full QAD, 8xH200, seq8192/gbs16, 4000 steps.
#
# Hyper-parameters match the formal run
#   q38-27b__mxfp4__qad-full__hf__h200x8__s8192__gbs16__1108-0747
# and live in recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn.yaml:
#   lr 2e-6 -> 2e-7, warmup 50, wd 0.1, beta2 0.95, clip 1.0, seed 42,
#   save every 500 steps, exact fp32 amax from the PTQ summary.
#
# Usage:
#   bash scripts/run_full_8xh200.sh
#   CONTAINER=qad-quant-h200 bash scripts/run_full_8xh200.sh   # run in container
# DATA is required for 27B runs. SUMMARY optionally supplies PTQ amax values.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."   # neural-compressor repo root
export NCCL_NVLS_ENABLE=0                # NVLS bind fails on this fabric
export PYTHONPATH="${PYTHONPATH:-$PWD}"

MODEL=${MODEL:-Qwen/Qwen3.8-27B}
DATA=${DATA:-}
SUMMARY=${SUMMARY:-}
OUT=${OUT:-runs/qwen3_8-27b__mxfp4__qad-full__hf__h200x8__s8192__gbs16}
PYTHON=${PYTHON:-python}
NPROC=${NPROC:-8}
CONTAINER=${CONTAINER:-}
STOP_AFTER=${STOP_AFTER:-0}
SAVE_STEPS=${SAVE_STEPS:-}

EXTRA=()
if [[ "$STOP_AFTER" -gt 0 ]]; then
  EXTRA+=(--stop-after "$STOP_AFTER")
fi
if [[ -n "$SAVE_STEPS" ]]; then
  EXTRA+=(--save-steps "$SAVE_STEPS")
fi
if [[ -n "$DATA" ]]; then
  EXTRA+=(--data "$DATA")
fi
if [[ -n "$SUMMARY" ]]; then
  EXTRA+=(--amax-source "$SUMMARY")
fi

CMD=("$PYTHON" -m torch.distributed.run --nproc_per_node "$NPROC"
     examples/pytorch/llm/qwen/qad/main.py
     --recipe examples/pytorch/llm/qwen/qad/recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn.yaml
     --model "$MODEL" --output "$OUT" "${EXTRA[@]}")

GPU_ENV=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_ENV=(-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi
if [[ -n "$CONTAINER" ]]; then
  docker exec "${GPU_ENV[@]}" -e NCCL_NVLS_ENABLE=0 -e PYTHONPATH="$PWD" -w "$PWD" "$CONTAINER" "${CMD[@]}"
else
  "${CMD[@]}"
fi
