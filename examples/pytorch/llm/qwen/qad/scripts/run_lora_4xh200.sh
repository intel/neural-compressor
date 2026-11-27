#!/usr/bin/env bash
# Qwen3.8-27B LoRA QAD (path B: merge-before-quant + self-teacher), 4xH200, seq8192/gbs16.
#
# Hyper-parameters live in recipes/qwen3_5_27b_mxfp4_fp8_lora_path_b.yaml:
#   lr 2e-4 -> 2e-5, warmup 0, clip 1.0, rank/alpha 64, 2000 steps.
# DATA is required for 27B runs. Static FP8 amax calibrates online via `calib_batches`
# unless QUANTIZER_STATE (native JSON) or SUMMARY (legacy ModelOpt summary) is supplied.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."   # neural-compressor repo root
export NCCL_NVLS_ENABLE=0
export PYTHONPATH="${PYTHONPATH:-$PWD}"

MODEL=${MODEL:-Qwen/Qwen3.8-27B}
DATA=${DATA:-}
QUANTIZER_STATE=${QUANTIZER_STATE:-}
SUMMARY=${SUMMARY:-}
RECIPE=${RECIPE:-examples/pytorch/llm/qwen/qad/recipes/qwen3_5_27b_mxfp4_fp8_lora_path_b.yaml}
OUT=${OUT:-runs/qwen3_8-27b__mxfp4__lora-qad__hf__h200x4__s8192__gbs16}
PYTHON=${PYTHON:-python}
NPROC=${NPROC:-4}
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
  EXTRA+=(--data_prefix "$DATA")
fi
if [[ -n "$QUANTIZER_STATE" ]]; then
  EXTRA+=(--quantizer_state_path "$QUANTIZER_STATE")
elif [[ -n "$SUMMARY" ]]; then
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
