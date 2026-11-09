#!/usr/bin/env bash
# Qwen3.8-27B LoRA QAD (path B: merge-before-quant), 4xH200, seq2048/gbs4.
#
# Hyper-parameters match the reference LoRA run (see
# recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn_lora.yaml):
#   lr 2e-4 -> 2e-5, warmup 0, clip 1.0, rank/alpha 32, 2000 steps.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."   # neural-compressor repo root
export NCCL_NVLS_ENABLE=0
export PYTHONPATH="${PYTHONPATH:-$PWD}"

MODEL=${MODEL:-/home/kaokaolv/models/Qwen3.8-27B}
DATA=${DATA:-/home/kaokaolv/qad/data/tokenized_cascade2_full/nvidia--Nemotron-Cascade-2-SFT-Data_chat_train_messages_max150000}
OUT=${OUT:-runs/qwen3_8-27b__mxfp4__lora-qad__hf__h200x4__s2048__gbs4}
PYTHON=${PYTHON:-python}
NPROC=${NPROC:-4}
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

CMD=("$PYTHON" -m torch.distributed.run --nproc_per_node "$NPROC"
     examples/pytorch/llm/qwen/qad/main.py
     --recipe examples/pytorch/llm/qwen/qad/recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn_lora.yaml
     --model "$MODEL" --data "$DATA" --output "$OUT" "${EXTRA[@]}")

GPU_ENV=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_ENV=(-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi
if [[ -n "$CONTAINER" ]]; then
  docker exec "${GPU_ENV[@]}" -e NCCL_NVLS_ENABLE=0 -e PYTHONPATH="$PWD" -w "$PWD" "$CONTAINER" "${CMD[@]}"
else
  "${CMD[@]}"
fi
