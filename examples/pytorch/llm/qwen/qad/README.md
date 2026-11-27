# QAD (Quantization-Aware Distillation) for Qwen3.5 / Qwen3.8-27B

Reference example for Intel Neural Compressor's composable PyTorch QAT + KD + QAD stack:

- `neural_compressor.torch.quantization`: `FakeQuantConfig`, `QuantSchemeRule`, `prepare_qat`, `QATTrainer`, `QuantizationTrainingArguments`, fused Triton QDQ kernels, and PEFT `QuantLoraLinear`.
- `neural_compressor.torch.distillation`: `DistillationConfig`, `DistillationTrainingArguments`, `KDTrainer`, chunked `TopKLogitsKLLoss` / `LogitsKLLoss`, FSDP2 teacher sharding, and zero-VRAM `teacher_model="self"`.
- `neural_compressor.torch.qad`: `QADTrainingArguments`, `QADTrainer`, and `prepare_qad_model`.
- `neural_compressor.torch.export`: `export_hf_checkpoint` (`export_format="bf16"` and `export_format="compressed_tensors"` real packed `MXFP4` + `FP8` / `MXFP8` / `NVFP4` for direct **vLLM** serving, including multimodal VLM text-tower key remapping).

What the shipped recipes train:

- **Mixed fake quantization** — MXFP4 (`E2M1`, block 32, `E8M0` power-of-two scales) or NVFP4 (`E2M1`, block 16, `E4M3` + global `amax`) for MLP weights and activations, paired with static FP8 `E4M3` (`amax` calibrated natively or loaded from `quantizer_state.json`) for attention & GatedDeltaNet projections;
- **Fused Triton QDQ kernels** — automatically dispatched on GPU (`quant_backend="auto"`), achieving **0-ULP bitwise parity** against PyTorch / ModelOpt CUDA references with **8.9x–21.6x kernel speedups** and **0 B** temporary workspace;
- **Chunked Top-$k$ / Full-vocab KL distillation** — FP32 forward/reverse KL computed over token chunks with `lm_head` chunking (`logits_chunk_size`) so full `[B, S, V]` logits never materialize simultaneously;
- **Full fine-tuning and PEFT LoRA**:
  - **Path B (`lora_merge_before_quant: true`, default)**: $y = Q_w(W + s \cdot BA)\,Q_a(x)$ — deployment-exact adapter merging inside quantization;
  - **Path A (`lora_merge_before_quant: false`)**: $y = Q_w(W)\,Q_a(x) + s \cdot BA\,x$ — compatible with optional `unsloth` models;
  - **Zero-VRAM Self-Teacher (`teacher_placement: self`)**: dynamically disables student LoRA adapters and fake quantizers during the `torch.no_grad()` teacher pass — eliminating the second 54 GiB 27B teacher model entirely;
- **PyTorch FSDP2** (`fsdp_version: 2`) with activation checkpointing, automatic teacher layer sharding (`teacher_placement: shard`), and optional FSDP2 CPU offload (`fsdp_config.cpu_offload: true`) for `< 82 GiB/GPU` 4-card full-parameter 27B QAD;
- **Real packed `compressed-tensors` & BF16 HF export** (`export.py --format compressed_tensors`), preserving multimodal VLM vision weights & processors (`keep_vision=True`) for immediate **vLLM** and `lm_eval` deployment.

## Directory Layout

| Path | Purpose |
|---|---|
| `train.py` | Standard `HfArgumentParser((ModelDataArguments, QADTrainingArguments))` + `--config <yaml>` training script |
| `modeling/qwen35_text_loader.py` | Streams `model.language_model.*` + `lm_head.*` from a Qwen3.5/3.8 VLM checkpoint into standard `Qwen3_5ForCausalLM` (plus optional `unsloth` loader) |
| `data/megatron_indexed_dataset.py` | Memory-mapped Megatron `.bin/.idx` reader, `PackedDataset` (loss-masked padding), and `SyntheticDataset` |
| `recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn_lora.yaml` | Formal 4×GPU RTX Pro 6000 / H200 Path-B LoRA QAD recipe (`seq8192`, `gbs16`, `rank=32`, `warmup=50`, `wd=0.1`) |
| `recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn.yaml` | Formal 8×GPU FSDP2 full-parameter MXFP4+FP8 QAD recipe (`seq8192`, `gbs16`, 4000 steps) |
| `recipes/qwen3_5_27b_mxfp4_fp8_full_fsdp2.yaml` | Full-parameter MXFP4+FP8 QAD recipe (8×GPU FSDP2, `seq8192`, `gbs16`, 4000 steps) |
| `recipes/qwen3_5_27b_mxfp4_fp8_full_cpu_offload.yaml` | Low-memory 4×GPU (`< 82 GiB/GPU`) full-parameter MXFP4+FP8 QAD recipe (FSDP2 `cpu_offload: true` + sharded teacher) |
| `recipes/qwen3_5_27b_mxfp4_fp8_lora_path_b.yaml` | Zero-VRAM `self`-teacher Path-B LoRA QAD recipe (`< 45 GiB/GPU` on 4 GPUs, `seq8192`, `rank=64`, 2000 steps) |
| `recipes/qwen3_5_27b_nvfp4_fp8_full_fsdp2.yaml` | Full-parameter NVFP4+FP8 QAD recipe (8×GPU FSDP2, `seq8192`, `gbs16`, 4000 steps) |
| `prepare_data.py` | Tokenizes an HF chat/text dataset into Megatron `.bin/.idx` shards |
| `calibrate_amax.py` | Single-rank native static `amax` calibration utility writing `quantizer_state.json` |
| `tools/convert_modelopt_amax.py` | Optional offline converter from ModelOpt `.quant_summary.txt` or `.pt` state into `quantizer_state.json` |
| `export.py` | Streaming checkpoint exporter (`--format compressed_tensors` or `--format bf16`) with VLM vision preservation |
| `scripts/smoke_tiny.sh` | 1-GPU tiny-model end-to-end smoke test (calibration + QAD step + metrics) |
| `scripts/run_qad_rtxpro_mxfp4_lora_b.sh` | Formal 4-GPU RTX Pro 6000 / H200 MXFP4+FP8 Path-B LoRA QAD launcher |
| `scripts/run_full_8xh200.sh` | 8-GPU full-parameter QAD launcher |
| `scripts/run_lora_4xh200.sh` | 4-GPU Path-B LoRA QAD launcher |

Library entry points used in `train.py` and `export.py`:

```python
from neural_compressor.torch.quantization import FakeQuantConfig, prepare_qat
from neural_compressor.torch.qad import QADTrainer, QADTrainingArguments, prepare_qad_model
from neural_compressor.torch.export import export_hf_checkpoint
```

## Quick Start

```bash
# 0) Install Neural Compressor (editable) and example dependencies
pip install -e . && pip install -r examples/pytorch/llm/qwen/qad/requirements.txt

# 1) 1-GPU wiring smoke test (tiny Qwen3.5, synthetic tokens, 2 steps)
CUDA_VISIBLE_DEVICES=0 bash examples/pytorch/llm/qwen/qad/scripts/smoke_tiny.sh

# 2) Optional: prepare Megatron .bin/.idx data & single-rank static FP8 amax JSON
python examples/pytorch/llm/qwen/qad/prepare_data.py \
  --model /path/Qwen3.8-27B \
  --dataset nvidia/Nemotron-Cascade-2-SFT-Data --subset chat \
  --out /path/tokenized_cascade2

CUDA_VISIBLE_DEVICES=0 python examples/pytorch/llm/qwen/qad/calibrate_amax.py \
  --model /path/Qwen3.8-27B \
  --recipe examples/pytorch/llm/qwen/qad/recipes/qwen3_5_27b_mxfp4_fp8_full_fsdp2.yaml \
  --data /path/tokenized_cascade2 --samples 32 --out /path/quantizer_state.json

# 3) Formal 4×GPU RTX Pro 6000 / H200 MXFP4+FP8 LoRA Path-B QAD (seq8192, gbs16, native online calibration)
QAD_GPUS=0,1,2,3 QAD_MODEL=/path/Qwen3.8-27B QAD_DATA=/path/tokenized_cascade2 \
  bash examples/pytorch/llm/qwen/qad/scripts/run_qad_rtxpro_mxfp4_lora_b.sh

# 4) Full QAD on 8×H200 (seq8192, gbs16, 4000 steps)
DATA=/path/tokenized_cascade2 QUANTIZER_STATE=/path/quantizer_state.json \
  bash examples/pytorch/llm/qwen/qad/scripts/run_full_8xh200.sh

# 5) Low-Memory Full QAD on 4×80GB/96GB GPUs (FSDP2 CPU offload, < 82 GiB/GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  examples/pytorch/llm/qwen/qad/train.py \
  --config examples/pytorch/llm/qwen/qad/recipes/qwen3_5_27b_mxfp4_fp8_full_cpu_offload.yaml \
  --model_name_or_path /path/Qwen3.8-27B \
  --data_prefix /path/tokenized_cascade2 \
  --quantizer_state_path /path/quantizer_state.json \
  --output_dir runs/qwen35_27b_lowmem
```

Every field in `--config <yaml>` maps directly onto `ModelDataArguments` or `QADTrainingArguments` (`transformers.TrainingArguments` subclass), so any parameter (`--learning_rate`, `--max_steps`, `--quant_backend`, `--kd_temperature`, `--kd_top_k`, `--save_steps`, etc.) can be overridden cleanly on the command line.

## YAML Recipe Format

```yaml
model_name_or_path: Qwen/Qwen3.8-27B
model_dtype: bfloat16
seq_length: 8192

# Quantization (FakeQuantConfig inline dict or preset path)
quant:
  backend: auto              # auto -> fused Triton QDQ on GPU, PyTorch ATen fallback
  config_groups:
    - targets: ['re:.*\.mlp\.(gate_proj|up_proj|down_proj)$']
      weights: {format: mxfp4, dynamic: true}
      input_activations: {format: mxfp4, dynamic: true}
    - targets: ['re:.*\.(self_attn|linear_attn)\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj)$']
      weights: {format: fp8_e4m3, dynamic: false}
      input_activations: {format: fp8_e4m3, dynamic: false}
  ignore: ['lm_head', 're:.*mlp\.gate$']

# Knowledge Distillation (DistillationTrainingArguments)
teacher_placement: shard     # shard (FSDP2) | self (zero-VRAM LoRA) | cuda | cpu
kd_loss_type: topk_kl        # topk_kl | kl | ce
kd_top_k: 4096
kd_temperature: 1.0
kd_reverse_kl: false
kd_loss_weight: 1.0
ce_loss_weight: 0.0
logits_chunk_size: 1024      # bounds peak logit memory on 248k vocabulary

# Standard HuggingFace TrainingArguments + FSDP2
bf16: true
fsdp: "full_shard auto_wrap"
fsdp_config:
  fsdp_version: 2
  reshard_after_forward: true
  activation_checkpointing: true
  cpu_offload: false
```

### Static FP8 `amax` Sources

Static quantizers (`dynamic: false`, such as `fp8_e4m3`) calibrate FP32 `amax` buffers natively without external toolkits:

1. **Native online FSDP2 calibration (`calib_batches`, default)**: `QADTrainer` automatically bypasses dynamic quantizers during calibration, observes static activations across distributed ranks, and all-reduces exact `MAX` `amax` buffers before step 1 (`0.0`-ULP identical to ModelOpt `mtq.quantize`).
2. **Standalone pre-calibration (`calibrate_amax.py`)**: Runs `calibrate_static_amax` over `N` sequences on 1 GPU and writes lossless `quantizer_state.json` (`amax` + IEEE-754 `amax_hex`), reusable via `--quantizer_state_path`.
3. **Optional ModelOpt import (`tools/convert_modelopt_amax.py`)**: Converts legacy ModelOpt `.quant_summary.txt` or `.pt` state files into `quantizer_state.json` for cross-framework parity experiments.

## Training Metrics & Checkpoints

Each `QADTrainer` log step emits:

```text
loss, kd_loss, ce_loss, kd_tokens, learning_rate, grad_norm,
tokens, tokens_per_s, step_s, world_size, recompute,
params_total, params_trainable,
tflops_student, tflops_teacher, tflops_total, tflops_*_per_gpu,
mfu_student, mfu_teacher, mfu_total, peak_basis, peak_gib, alloc_gib
```

Every saved checkpoint writes `quantizer_state.json` (exact IEEE-754 `amax_hex` buffers) and `quant_config.json` alongside `model.safetensors` (or PEFT `adapter_model.safetensors`), enabling deterministic resume and one-command `compressed-tensors` export.

## Export & vLLM Deployment

Export runs in a standalone process with bounded host RAM (~1 shard in memory at a time) and supports both **real packed `compressed-tensors`** (`MXFP4` packed `uint8` + `E8M0` scales and `FP8_E4M3` tensors + scales) and **BF16** (`format="bf16"`), automatically transplanting quantized text weights back into the full multimodal VLM directory (`keep_vision=True`):

```bash
# 1) Real packed compressed-tensors MXFP4 + FP8 export (ready for vLLM V1!)
python examples/pytorch/llm/qwen/qad/export.py \
  --model /path/Qwen3.8-27B \
  --format compressed_tensors \
  --checkpoint runs/<run>/checkpoint-1000 \
  --out runs/export-mxfp4-fp8

# 2) Path-B LoRA adapter merge + real compressed-tensors export
python examples/pytorch/llm/qwen/qad/export.py \
  --model /path/Qwen3.8-27B \
  --format compressed_tensors \
  --checkpoint runs/<run>/checkpoint-1000 \
  --out runs/export-lora-mxfp4-fp8

# 3) Serve directly with vLLM or evaluate with lm_eval
lm_eval --model vllm \
  --model_args pretrained=runs/export-mxfp4-fp8,tensor_parallel_size=4,gpu_memory_utilization=0.85,max_model_len=8192 \
  --tasks mmlu,gsm8k,piqa,hellaswag --batch_size auto
```

## Numerical Parity & Performance Highlights

- **L1–L4 Numerical Parity vs. ModelOpt / Megatron**:
  - **L1 Operator QDQ**: **0-ULP bitwise identity** across `MXFP4`, `MXFP8`, `NVFP4`, and `FP8_E4M3` (both PyTorch ATen and fused Triton kernels).
  - **L2 Checkpoint / Scale Import**: **0.0 diff** FP32 `amax` preservation across Megatron $\leftrightarrow$ HuggingFace projection splits.
  - **L3 Single-Step & Multi-Step Training**: Steps 1–2 match the formal 8×H200 run (`...1108-0747`) to `< 1e-8` (`0.08816621`, `0.09347378`), and controlled single-step/2-step FSDP2 probes match loss, grad norms, and parameter updates bitwise (`0.0` diff).
  - **L4 Export & Serving**: Packed `compressed-tensors` `MXFP4` + `FP8` checkpoints decompress with **0-ULP exact match** against fake-quantized `Q_w(W)` tensors and load natively in **vLLM V1**.
- **Memory & Kernel Efficiency**:
  - **Fused Triton QDQ**: **8.95x–21.64x** faster than eager PyTorch QDQ on `Qwen3.8-27B` MLP shapes (`[5120, 17408]`, `[17408, 5120]`) with **0 B** temporary workspace.
  - **4-GPU Low-Memory Full QAD**: `< 82 GiB/GPU` at `seq8192` via FSDP2 `cpu_offload: true` + layer-sharded teacher (`13.5 GiB/GPU`).
  - **Zero-VRAM Self-Teacher LoRA QAD**: `< 45 GiB/GPU` on 1–4 GPUs by reusing the student's frozen base weights for teacher inference.
