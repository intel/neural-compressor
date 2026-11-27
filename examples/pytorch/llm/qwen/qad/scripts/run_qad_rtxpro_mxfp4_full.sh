#!/usr/bin/env bash
# Formal Qwen3.8-27B MXFP4+FP8 Full-Parameters QAD on 4x RTX Pro 6000 / H200 GPUs (single NUMA domain).
#
# Self-contained Intel Neural Compressor counterpart to `qad_full_rtxpro_mxfp4.sh`:
#   - Mixed precision: MXFP4 dynamic W4A4 MLP + static FP8 E4M3 W8A8 Attention/GatedDeltaNet
#   - Native static FP8 amax calibration (`QAD_CALIB_BATCHES`, default 32 -> 128 sequences across 4 GPUs,
#     or pre-calibrated native `quantizer_state.json` via `QAD_QUANTIZER_STATE`)
#   - Full-Parameters FSDP2 (`cpu_offload=true` with FP32 master weights + FSDP2 activation checkpointing)
#   - Sharded online BF16 Teacher (`teacher_placement=shard`) with Top-4096 forward KL (`kd_temperature=1.0`)
#   - Sequence length 8192, micro-batch 1, grad_accum 4 -> global_batch_size 16 (131,072 tokens/step)
#   - AdamW (`lr=2e-6 -> 2e-7` cosine, `warmup_steps=50`, `beta2=0.95`, `weight_decay=0.1`, `clip=1.0`)
#
# Example (200-step formal alignment run matching 4000-step cosine schedule on GPUs 0,1,2,3):
#   QAD_GPUS=0,1,2,3 QAD_ITERS=4000 QAD_STOP_AFTER_STEPS=200 QAD_SAVE_INTERVAL=0 \
#   QAD_MODEL=/path/to/Qwen3.8-27B QAD_DATA=/path/to/tokenized_prefix \
#   QAD_OUT=runs/qwen38-27b-mxfp4-qad-full-200 \
#   bash examples/pytorch/llm/qwen/qad/scripts/run_qad_rtxpro_mxfp4_full.sh
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."   # neural-compressor repo root

export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES="${QAD_GPUS:-0,1,2,3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29557}"
RECIPE="${RECIPE:-examples/pytorch/llm/qwen/qad/recipes/qwen3_5_27b_mxfp4_fp8_full_cpu_offload.yaml}"

MODEL="${QAD_MODEL:-${MODEL:-Qwen/Qwen3.8-27B}}"
DATA_INPUT="${QAD_DATA:-${DATA:-}}"
OUT="${QAD_OUT:-${OUT:-runs/qwen38-27b-mxfp4-qad-full-rtxpro}}"
ITERS="${QAD_ITERS:-4000}"
STOP_AFTER_STEPS="${QAD_STOP_AFTER_STEPS:-0}"
SAVE_INTERVAL="${QAD_SAVE_INTERVAL:-500}"
CALIB_BATCHES="${QAD_CALIB_BATCHES:-32}"
QUANTIZER_STATE="${QAD_QUANTIZER_STATE:-}"
EXPORT_DIR="${QAD_EXPORT_DIR:-}"

DATA_PREFIX=""
if [[ -n "$DATA_INPUT" ]]; then
  if [[ -f "${DATA_INPUT}.bin" && -f "${DATA_INPUT}.idx" ]]; then
    DATA_PREFIX="$DATA_INPUT"
  elif [[ -d "$DATA_INPUT" ]]; then
    DATA_PREFIX=$(ls "$DATA_INPUT"/*.bin 2>/dev/null | head -n 1 | sed 's/\.bin$//' || true)
    if [[ -z "$DATA_PREFIX" && -d "$DATA_INPUT/tokenized_cascade2_full" ]]; then
      DATA_PREFIX=$(ls "$DATA_INPUT/tokenized_cascade2_full"/*.bin 2>/dev/null | head -n 1 | sed 's/\.bin$//' || true)
    fi
  else
    DATA_PREFIX="$DATA_INPUT"
  fi
fi
if [[ -z "$DATA_PREFIX" ]]; then
  echo "Provide QAD_DATA=<prefix_or_dir> pointing to Megatron .bin/.idx tokenized data." >&2
  exit 1
fi

mkdir -p "$OUT"

EXTRA=()
if [[ "$SAVE_INTERVAL" -le 0 || ( "$STOP_AFTER_STEPS" -gt 0 && "$SAVE_INTERVAL" -gt "$STOP_AFTER_STEPS" ) || "$SAVE_INTERVAL" -gt "$ITERS" ]]; then
  EXTRA+=(--save_strategy no)
else
  EXTRA+=(--save_strategy steps --save_steps "$SAVE_INTERVAL")
fi

if [[ "$STOP_AFTER_STEPS" -gt 0 ]]; then
  EXTRA+=(--stop_after_steps "$STOP_AFTER_STEPS")
fi

if [[ -n "$QUANTIZER_STATE" ]]; then
  EXTRA+=(--quantizer_state_path "$QUANTIZER_STATE" --calib_batches 0)
else
  EXTRA+=(--calib_batches "$CALIB_BATCHES")
fi

if [[ -n "$EXPORT_DIR" ]]; then
  EXTRA+=(--export_dir "$EXPORT_DIR")
fi

exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node "$NPROC" \
  --master_port "$MASTER_PORT" \
  examples/pytorch/llm/qwen/qad/train.py \
  --config "$RECIPE" \
  --model_name_or_path "$MODEL" \
  --data_prefix "$DATA_PREFIX" \
  --max_steps "$ITERS" \
  --output_dir "$OUT" \
  "${EXTRA[@]}"
