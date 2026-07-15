#!/usr/bin/env bash
# Wan2.2 T2V using SpargeAttn (block-sparse + quantized attention) as the backend.
#
# SpargeAttn replaces self-attention via the plugin's SpargeAttnImpl. The plugin
# overrides the SAGE_ATTN backend slot when VLLM_SPARGE_ATTN=1, so the host's
# DIFFUSION_ATTENTION_BACKEND stays "SAGE_ATTN".
#
# SPARGE_TOPK=1.0 keeps ALL KV blocks (dense). This isolates SpargeAttn's
# quantization error for a correctness-first run (no sparsity speedup yet).
# To enable real block-sparsity later, lower SPARGE_TOPK (e.g. 0.8, 0.5) — but
# note SpargeAttn is UNtuned on Wan2.2, so video quality at topk<1.0 is unverified.
set -euo pipefail
cd /home/yiliu7/workspace/vllm-omni

source /home/yiliu7/workspace/venvs/omni/bin/activate
export C_INCLUDE_PATH=$HOME/.local/include:$HOME/.local/include/python3.12${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}

# The omni venv has a stale vllm 0.20.0 wheel, but the editable vllm-omni (rebased
# to v0.22.0) needs symbols only present in the newer vllm source checkout at
# /home/yiliu7/workspace/vllm (0.21.1rc1, already compiled). Prepend it on
# PYTHONPATH so `import vllm` resolves to that tree. Non-invasive: no venv mutation.
# export PYTHONPATH=/home/yiliu7/workspace/vllm${PYTHONPATH:+:$PYTHONPATH}

PY=/home/yiliu7/workspace/venvs/omni/bin/python
MODEL=/storage/yiliu7/Wan-AI/Wan2.2-T2V-A14B-Diffusers

# height=420
# width=280
# num_frames=81
# num_inference_steps=40
height=720
width=1280
num_frames=81
num_inference_steps=40
seed=42
SPARGE_TOPK=0.5

out_dir=/home/yiliu7/workspace/wan-res
out_file=t2v_sparge_h${height}_w${width}_f${num_frames}_s${num_inference_steps}_topk${SPARGE_TOPK}_2
mkdir -p "${out_dir}"

# VLLM_SPARGE_ATTN=1 activates the plugin's SpargeAttn backend (overriding SAGE_ATTN).
# SPARGE_ATTN_REPO is injected on sys.path so the prebuilt spas_sage_attn package
# (built in the `sparge` venv) is importable from the `omni` venv — identical
# torch/CUDA/python ABI, so no rebuild is needed.
# NOTE: linear/MoE quantization is intentionally OFF (bf16). The host's
# --quantization mxfp8 path is gated to NPU/XPU only (mxfp8_config.py:144) and
# raises on CUDA — and it is orthogonal to attention. Leaving it off isolates the
# SpargeAttn attention integration, which is what this demo exercises.
VLLM_SPARGE_ATTN=1 \
SPARGE_ATTN_REPO=/home/yiliu7/workspace/SpargeAttn \
SPARGE_MODE=topk \
SPARGE_TOPK=${SPARGE_TOPK} \
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN \
CUDA_VISIBLE_DEVICES=6,7 \
PYTHONUNBUFFERED=1 \
  ${PY} -u examples/offline_inference/text_to_video/text_to_video.py \
    --model ${MODEL} \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --negative-prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --height ${height} \
    --width ${width} \
    --num-frames ${num_frames} \
    --guidance-scale 4.0 \
    --guidance-scale-high 3.0 \
    --num-inference-steps ${num_inference_steps} \
    --fps 16 \
    --tensor-parallel-size 2 \
    --output ${out_dir}/${out_file}.mp4 2>&1 | tee ${out_file}.log
