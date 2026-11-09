# QAD (Quantization-Aware Distillation) for Qwen3.8-27B

Reference example for `neural_compressor.torch.qad`. The library provides the
mechanism (fake-quant formats, quantizers, distillation loss, trainer, export);
this directory only contains the Qwen3.8-27B recipe, data conversion and
launch scripts.

What the recipe trains:

- **mixed fake quantization** — MXFP4 (E2M1, block 32, E8M0) for MLP weights
  and activations; FP8 E4M3 with a calibrated per-tensor `amax` for
  attention/GDN projections;
- **top-k KL distillation** from a frozen bf16 teacher (teacher top-k logits
  only, fp32 loss, chunked over tokens);
- **full fine-tuning and LoRA**:
  - path A (`quant_then_lora`): `y = Q_w(W)·Q_a(x) + BA·x`
  - path B (`lora_then_quant`, default): `y = Q_w(W + BA)·Q_a(x)`
- **FSDP2** training with activation checkpointing and optional teacher layer
  sharding; metrics for loss/throughput/TFLOPs/peak memory;
- **HF export** of the full VLM (vision tower + processor kept), ready for
  vLLM and `lm_eval`.

> The export is **bf16** (fake-quant training keeps float weights). Real
> MXFP4/FP8 export is a separate follow-up that reuses the calibrated `amax`.

## Layout

| Path | Purpose |
|---|---|
| `main.py` | thin CLI: recipe → `quantize()` → `QADTrainer` → optional export |
| `recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn.yaml` | full QAD recipe (4000 steps, 8×H200) |
| `recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn_lora.yaml` | LoRA path-B recipe (2000 steps, 4×H200) |
| `prepare_data.py` | HF dataset → Megatron `.bin/.idx` |
| `calibrate_amax.py` | self-calibrate FP8 `amax` when no PTQ summary exists |
| `export.py` | export a saved checkpoint to a HF directory |
| `scripts/run_full_8xh200.sh` | full QAD reference run |
| `scripts/run_lora_4xh200.sh` | LoRA path-B reference run |
| `scripts/smoke_tiny.sh` | 1-GPU wiring smoke (tiny model, 2 steps) |

Library entry points used here:

```python
from neural_compressor.torch.quantization import FakeQuantConfig, QuantRule, quantize
from neural_compressor.torch.qad import QADConfig, QADTrainer, export, load_model
```

## Quick start

```bash
# 0) install this repo (editable) and the example extras
pip install -e . && pip install -r examples/pytorch/llm/qwen/qad/requirements.txt

# 1) wiring smoke: tiny model, seq128, 2 steps, 1 GPU
bash examples/pytorch/llm/qwen/qad/scripts/smoke_tiny.sh

# 2) full QAD: 8×H200, seq8192/gbs16, 4000 steps
bash examples/pytorch/llm/qwen/qad/scripts/run_full_8xh200.sh

# 3) LoRA path B: 4×H200, seq2048/gbs4, 2000 steps
bash examples/pytorch/llm/qwen/qad/scripts/run_lora_4xh200.sh

# 6-step parity run with the formal 4000-step LR schedule
STOP_AFTER=6 SAVE_STEPS=0 bash examples/pytorch/llm/qwen/qad/scripts/run_full_8xh200.sh
```

The scripts read their hyper-parameters from the recipes, so they reproduce the
runs used for the validation experiments:

| Run | Recipe | Key parameters |
|---|---|---|
| full QAD, 8×H200 (`...1108-0747`) | `qwen3_8_27b_mxfp4_mlp_fp8_attn.yaml` | seq8192, mbs1, gbs16 (accum 2), 4000 steps, lr 2e-6→2e-7, warmup 50, wd 0.1, β2 0.95, clip 1.0, save/500, seed 42 |
| LoRA path B, 4×H200 | `..._lora.yaml` | seq2048, gbs4, 2000 steps, lr 2e-4→2e-5, warmup 0, rank/alpha 32, clip 1.0, save/500, seed 42 |

Set `CONTAINER=qad-quant-h200` to launch inside the training container (the
scripts then `docker exec` with `PYTHONPATH`/`NCCL_NVLS_ENABLE=0`, matching the
formal-run launcher).

Data preparation (if you do not already have the tokenized prefix):

```bash
python examples/pytorch/llm/qwen/qad/prepare_data.py \
  --model /path/Qwen3.8-27B \
  --dataset nvidia/Nemotron-Cascade-2-SFT-Data --subset chat \
  --out /path/tokenized_cascade2_full
```

## The recipe

```yaml
quant:
  algorithm: max
  rules:
    - module: '.*\.mlp\.(gate_proj|up_proj|down_proj)$'
      weight: {format: mxfp4}
      input:  {format: mxfp4}
    - module: '.*\.(self_attn|linear_attn)\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj)$'
      weight: {format: fp8_e4m3, scheme: static}
      input:  {format: fp8_e4m3, scheme: static}
  amax_source:
    file: /path/.quant_summary.txt
    translate: modelopt_megatron
```

- `algorithm: max` is the **calibration algorithm** for static quantizers:
  each `amax` is the per-tensor maximum absolute value over the calibration
  batches (ModelOpt's `"max"`). It is only used when no `amax_source` is given;
  this recipe reads the PTQ summary instead, so calibration never runs. v1
  implements only `max` — other values raise `NotImplementedError`.
  The estimator is the same as ModelOpt's, but the **values are not guaranteed
  to match**: ModelOpt calibrates fused Megatron tensors (`linear_qkv` covers
  q/k/v together, `in_proj` covers qkv+z) on `cnn_dailymail` (128×8192), while
  the fallback calibrates each HF projection separately on the data you pass;
- rules are regex matches on **module names**, applied in order (later wins);
- `default: null` means everything is off unless a rule enables it;
- `^lm_head$` and `*embed_tokens*` are excluded by default;
- `scheme: static` needs an `amax`; the recipe reads the exact fp32 values from
  ModelOpt's `.quant_summary.txt` (the bf16-rounded HF `weight_scale` export is
  a fallback and drifts grad norm by ~2.3× over 64 FP8 layers).

Without a PTQ summary, calibrate first (single rank) and point
`amax_source.file` at the JSON:

```bash
python examples/pytorch/llm/qwen/qad/calibrate_amax.py \
  --recipe examples/pytorch/llm/qwen/qad/recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn.yaml \
  --data /path/tokenized --samples 32 --out /path/amax.json
```

## Training and logs

`QADTrainer` extends `transformers.Trainer`; each log line adds:

```text
loss, kd_loss, kd_tokens, learning_rate, grad_norm,
tokens, tokens_per_s, step_s, backend, world_size, recompute,
params_total, params_trainable,
tflops_student, tflops_teacher, tflops_total, tflops_*_per_gpu,
mfu_student, mfu_teacher, mfu_total, peak_basis, peak_gib, alloc_gib
```

- `peak_gib` starts at the first training step (excludes loading/FSDP wrapping);
- MFU uses global tokens and global student/teacher FLOPs divided by the
  aggregate device peak (`world_size × per-device BF16 peak`). The
  `tflops_*_per_gpu` values divide the same global rates by `world_size`;
  activation recomputation adds the corresponding student forward FLOPs;
- `mfu_*` is `None` on devices missing from the BF16 peak table instead of a
  invented denominator;
- the trainer writes `quantization_config.json` next to every checkpoint
  (plan + static `amax`), so quantizers can be rebuilt exactly.

## Memory footprint

All numbers are `peak_gib` reported by `QADTrainer`: the torch allocator peak
measured from the first training step, so model loading and FSDP2 wrapping are
excluded. `nvidia-smi` shows ~2–4 GiB more per GPU for the CUDA context/NCCL.

| Configuration (GPUs) | Peak GiB/GPU | Source run |
|---|---|---|
| full QAD, **8×H200**, seq8192, gbs16 | **118.9** | `...1108-0747` (max over 1000 steps) |
| full QAD, **4×H200**, seq1024, gbs4 | **126.4** | `...20261106-0237` |
| full QAD, 4×H200, seq8192 | not measured | activation memory grows with seq; the formal run uses 8 cards |
| LoRA path B, **4×H200, separate teacher** (layer-sharded), seq2048/1024, gbs4 | **50.3** | `...20261106-0025` / `...0405` |
| LoRA path B, **2×H200, shared teacher** (student weights reused, DDP), seq2048, gbs2 | **57.9** | `...20261106-0338` |
| LoRA path B, **1×H200, shared teacher** (DDP), seq2048, gbs1 | **57.5** | `...20261105-2352` |

Minimum GPU memory (measured peak + ~8–10% headroom for context/NCCL/fragmentation):

| Scenario | Measured peak | Measured on | Minimum per GPU |
|---|---|---|---|
| full QAD 27B, seq8192 | 118.9 | 8×H200 | **≥ 128 GiB** (H200 141 GiB; an 80 GiB card is not enough without optimizer offload / smaller seq) |
| full QAD 27B, seq1024 | 126.4 | 4×H200 | **≥ 136 GiB** (4×H200 is tight; 8 cards is the practical setting) |
| LoRA path B 27B, seq2048, separate teacher | 50.3 | 4×H200 | **≥ 56 GiB** |
| LoRA path B 27B, seq2048, shared teacher | 57.5 / 57.9 | 1× / 2×H200 | **≥ 64 GiB** |

Notes:

- **GPU counts matter**: the 50.3 GiB number is 4 cards with a separate
  (layer-sharded) teacher; the 57.5/57.9 numbers are 1–2 cards with a *shared*
  teacher, i.e. no second 27B model. A separate teacher on 1–2 GPUs is not
  feasible.
- fewer GPUs do not necessarily mean less memory per GPU: 4-card full QAD
  peaks higher (126.4) than 8-card (118.9) because parameters, gradients and
  Adam states are sharded over fewer ranks;
- the teacher is layer-sharded whenever FSDP2 is enabled (before `.to(device)`),
  which is what keeps the 4-card LoRA run at ~50 GiB/GPU;
- full QAD keeps fp32 Adam states on GPU; CPU optimizer offload is not part of
  this v1 (it would cut the full-QAD peak substantially);
- we have **no direct 80 GiB-card measurement** for LoRA path B. The 96 GB
  RTX PRO 6000 (91.5 GiB) fits by the 4-card number (50.3 GiB + headroom);
  48 GiB cards do not fit at seq2048.

### Reference-run parity (8×H200, 6 steps)

The full recipe was re-run with this library against the formal run
`...1108-0747` (same data, seed 42, 8 GPUs, 4000-step LR schedule, stop after 6):

| step | formal (`qad_impl`) | this library (A) | this library (B) |
|---|---|---|---|
| 1 | 0.08816621 | 0.08816621 | 0.08816621 |
| 2 | 0.09347378 | 0.09347378 | 0.09347378 |
| 3 | 0.10996736 | 0.11290175 | 0.10370222 |
| 4 | 0.11598256 | 0.10702234 | 0.10954103 |
| 5 | 0.08385994 | 0.08536845 | 0.08090645 |
| 6 | 0.06918009 | 0.06746635 | 0.06789998 |
| mean | 0.093438 | 0.092400 | 0.090615 |

Steps 1–2 match to 1e-8, i.e. the forward, the fake-quant scales and the first
optimizer update are identical. From step 3 the trajectories fluctuate because
the backward pass is not bitwise deterministic: two runs of the **same** code
already give step-1 grad norms 6.1018 vs 6.1593 (0.9%) and 6-step means 0.0906
vs 0.0924. Fake-quant turns tiny reduction-order differences into FP8 code
flips, which move the loss by a few percent.

Practical rule: use the single-step/tiny-model parity script when comparing
implementations (bitwise); use the loss curve only with a few-percent
tolerance.

`seed` only fixes RNG (data order, init, dropout). It does not control cuBLAS
algorithm/workspace selection (`CUBLAS_WORKSPACE_CONFIG`), attention backward
atomics, or NCCL collective internals, which is why the same code and seed
still differ at the 0.1–1% gradient level.

### Controlled bitwise parity after the QAD fixes

Separate short-sequence probes compared this library with the reference
`qad_impl` implementation using identical initial weights, FP32 amax values,
samples and AdamW settings. The probes compare gradients and update tensors,
not only scalar loss curves:

| Scenario | Configuration | Result |
|---|---|---|
| HF full QAD | 8×H200 FSDP2, seq256, 2 steps | Loss, grad norm, all 66 hidden dumps, gradient norms and update probes bitwise equal |
| unsloth full QAD | 4×H200 FSDP2, seq256, 1 step | Metadata, loss, grad norm and gradient/update probes bitwise equal |
| unsloth LoRA path A | 1×H200, seq256, 1 step | Same adapter initialization and FP32 scale hash; numerical checks bitwise equal |
| HF LoRA path B | 1×H200, seq256, 1 step | 400 merge-before-quant modules; numerical checks bitwise equal |

The HF path-B run gave loss `0.5040409564971924`, gradient norm
`3.4116947160122804` and update L2 `2.229990999862231` in both
implementations. LoRA A/B gradients and base/adapter updates for the layer-0
GDN output projection, layer-3 query projection and layer-63 MLP down
projection were bitwise equal. Requesting unsloth path B was also verified to
raise `ValueError: unsloth backend supports only quant_then_lora (path A)`.
These checks compare two implementations of the **same** backend; they do not
claim that unsloth is bitwise equal to HF or to ModelOpt/Megatron.

To smoke the shipped HF path-B recipe for two steps:

```bash
STOP_AFTER=2 SAVE_STEPS=0 \
  bash examples/pytorch/llm/qwen/qad/scripts/run_lora_4xh200.sh
```

## Export and evaluation

FSDP2 keeps parameters as DTensors, so export runs in a separate process from
a saved checkpoint (full-VLM export imports the Qwen2VL image processor, which
needs `pillow` and `torchvision` — both listed in `requirements.txt`):

```bash
# full fine-tuning
python examples/pytorch/llm/qwen/qad/export.py \
  --model /path/Qwen3.8-27B --format full \
  --state-dict runs/<run>/checkpoint-1000/model.safetensors --out runs/export

# LoRA (adapter is merged before saving)
python examples/pytorch/llm/qwen/qad/export.py \
  --model /path/Qwen3.8-27B --format full \
  --adapter runs/<run>/checkpoint-500 --out runs/export-lora
```

Evaluation follows the ModelOpt protocol (vLLM TP4 + lm-eval 0.4.13, full
datasets; see `scripts/eval_export_lm_eval.sh` in the workspace docs). The
reference numbers for the bf16 export of a QAD@1000 checkpoint are
mmlu 0.8376 / piqa 0.8145 / hellaswag 0.6338 / gsm8k 0.9757; the same weights
served in real MXFP4+FP8 are ≈0.8070 mmlu (ModelOpt measurement).

## Constraints

- **Static FP8 + multi-rank**: calibration must come from a file; the CLI
  refuses implicit dynamic fallback. Calibrate on one rank first.
- **Export under FSDP2**: not supported in-process; export from the saved
  checkpoint in a separate process (the CLI above).
- **Real quantized export** (`mxfp4_fp8`) is not implemented yet; the
  `amax`/plan metadata needed for it is already written next to checkpoints.
- **Backends and LoRA quant order**: `train.backend: hf` supports both path A
  (`quant_then_lora`) and path B (`lora_then_quant`). The optional `unsloth`
  backend supports path A only; requesting path B raises `ValueError` instead
  of silently using the wrong forward. The shipped LoRA recipe and
  `scripts/run_lora_4xh200.sh` use HF path B.
- **Teacher placement**: `distill.teacher_device` accepts `shard`, `cuda`, or
  `cpu`. With `cpu`, input IDs and attention masks run on the CPU teacher and
  its logits are copied back to the student device for the KD loss.
- **Export roundtrip**: model weights and quantizer buffers are saved
  separately. Static FP32 amax values live in `quantization_config.json` and
  are restored by `load_state_dict_into_model`. `ExportConfig.lora` accepts
  `merge` (default) or `adapter`; sharded text exports can be reloaded with
  `AutoModelForCausalLM.from_pretrained`.
- **unsloth dependency**: install unsloth separately only when selecting
  `train.backend: unsloth`; it is not a dependency of the default HF path.
