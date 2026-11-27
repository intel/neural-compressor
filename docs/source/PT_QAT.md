# PyTorch Quantization-Aware Training (QAT)

This document describes how to use Quantization-Aware Training (QAT) in Intel Neural Compressor (INC) to achieve high-accuracy, hardware-friendly quantization for large language models (LLMs). QAT explicitly simulates quantization during training, significantly narrowing the accuracy gap compared to pure Post-Training Quantization (PTQ), especially under aggressive compression (e.g., 4-bit).

---

## Contents

1. [Overview](#overview)  
2. [QAT Workflow and Benefits](#qat-workflow-and-benefits)  
3. [Quick Start](#quick-start)  
4. [Core Components](#core-components)  
5. [Configuration and Key Parameters](#configuration-and-key-parameters)  
6. [Unified Fake-Quant QAT, Distillation (KD), QAD, and `compressed-tensors` Export](#unified-fake-quant-qat-distillation-kd-qad-and-compressed-tensors-export)  
7. [Best Practices and Troubleshooting](#best-practices-and-troubleshooting)  
8. [References](#references)

---

## Overview

Quantization-Aware Training (QAT) is a training strategy that inserts “fake quantization” operators into the model during training. The forward pass mimics the quantized inference behavior, while gradients are still propagated in floating point. This allows the model to adapt to quantization noise and recover most of the accuracy that may be lost with PTQ-only workflows.

In this repository, QAT is integrated with:

- **PyTorch** and **Hugging Face Transformers**  
- **AutoRound** for quantization schemes and kernels 
- **Microscaling (MX) formats** (e.g., MXFP4, MXFP8) for efficient deployment

Although this document focuses on LLMs (e.g., Llama 3.x), the same concepts can be extended to other architectures.

---

## QAT Workflow and Benefits

### High-Level Workflow

A typical QAT pipeline consists of the following stages:

1. **Train or Fine-Tune a Baseline Model**  
   - Train or fine-tune the model in FP32/BF16 to obtain a strong baseline.
   - Example: fine-tune `meta-llama/Llama-3.1-8B` in BF16.

2. **Quantization-Aware Fine-Tuning (QAT)**  
   - Insert QAT modules into the model using `prepare_qat`.
   - Optionally load the PTQ model weights as initialization.
   - Fine-tune the quantized model with a small learning rate.

3. **Export and Deployment**  
   - Save the QAT model as a standard Hugging Face model directory.
   - Deploy with compatible inference engines (e.g., vLLM), or export using INC/AutoRound export utilities for specific runtimes.

### Why QAT?

Compared with PTQ, QAT offers:

1. **Higher Accuracy Under Aggressive Compression**  
   - QAT significantly reduces accuracy degradation for low-bit formats (e.g., MXFP4).

2. **Realistic Simulation of Inference Behavior**  
   - Fake-quant modules simulate the exact quantization scheme that will be used at inference, including MX formats.

3. **Better Robustness on LLMs**  
   - LLMs are often highly sensitive to weight perturbations; QAT helps the model adapt to such changes.

4. **Flexible Integration**  
   - QAT is implemented via modular PyTorch components (`QuantLinear`, `TensorQuantizer`) that can be inserted into standard Transformer architectures.

---

## Quick Start

This section walks through an end-to-end [example](/examples/pytorch/llm/others/qat).

### 1. Setup Environment

From the `qat` directory:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes (among others):

- `auto-round==0.9.3`
- `neural-compressor-pt==3.7`
- `transformers==4.53.0`
- `accelerate`
- `datasets`
- `lm-eval`


---

### 2. Baseline Fine-Tuning (BF16)

You can fine-tune a BF16 model using FSDP for Llama 3.1 as follows:

```bash
accelerate launch --config-file accelerate_config/fsdp1.yaml \
  main.py \
  --model_name_or_path meta-llama/Llama-3.1-8B \
  --model_max_length 4096 \
  --dataloader_drop_last True \
  --do_train True \
  --do_eval True \
  --output_dir ./llama3.1-finetuned \
  --dataset Daring-Anteater \
  --num_train_epochs 2.0 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 3000 \
  --eval_strategy steps \
  --eval_steps 3000 \
  --load_best_model_at_end True \
  --save_total_limit 2 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type linear \
  --logging_steps 1 \
  --report_to tensorboard
```

Key points (see `main.py`):

- `TrainingArguments` are extended with `model_max_length`, `bf16`, etc.
- Dataset is created via `make_supervised_data_module` (see `utils.py`).
- FSDP configuration is controlled by `accelerate_config/fsdp1.yaml` or `fsdp2.yaml`.

---


### 3. QAT Fine-Tuning

The core QAT logic is driven from `main.py` using a `QuantizationArguments` dataclass:

```python
@dataclass
class QuantizationArguments:
    quant_scheme: str | None = field(
        default=None,
        metadata={
            "help": "Specify the quantization format for PTQ/QAT. If specified, PTQ/QAT will be enabled.",
            "choices": ["MXFP8", "MXFP4"],
        },
    )
```

When `--quant_scheme` is provided, the model is prepared for QAT:

```python
if quant_args.quant_scheme is not None:
    from neural_compressor.torch.quantization.quantize import prepare_qat

    model.train()
    if quant_args.quant_scheme == "MXFP8":
        # Default MXFP8 scheme
        prepare_qat(model)
    if quant_args.quant_scheme == "MXFP4":
        mappings = {torch.nn.Linear: "MXFP4"}
        prepare_qat(model, mappings)

    logger.info("Finish model preparation for QAT.")
```

#### Example: QAT with MXFP4

```bash
accelerate launch --config-file accelerate_config/fsdp1.yaml \
  main.py \
  --model_name_or_path ./llama3.1-finetuned \
  --model_max_length 4096 \
  --dataloader_drop_last True \
  --do_train True \
  --do_eval True \
  --quant_scheme MXFP4 \
  --output_dir ./llama3.1-finetuned-qat \
  --dataset Daring-Anteater \
  --max_steps 1000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 3000 \
  --eval_strategy steps \
  --eval_steps 3000 \
  --load_best_model_at_end True \
  --save_total_limit 2 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type linear \
  --logging_steps 1 \
  --report_to tensorboard
```

---

### 4. Evaluation and Deployment

Once QAT training finishes, the model is saved in Hugging Face format via `QATTrainer.save_model`. You can evaluate it with vLLM and `lm_eval`:

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=./llama3.1-finetuned-qat,\
tensor_parallel_size=1,data_parallel_size=1,\
gpu_memory_utilization=0.8,max_model_len=32768 \
  --tasks gsm8k \
  --batch_size 8
```

`QATTrainer` also ensures the correct `dtype` is written back to `config.json` after FSDP training, which helps downstream inference libraries choose the right precision.

---

## Core Components

This section describes the key building blocks used by QAT under the hood.

### 1. `prepare_qat` and Module Replacement

The QAT preparation uses utility functions in `neural_compressor/torch/algorithms/qat/quant_utils.py`:

```python
def replace_with_quant_linear(model, quant_cfg=None):
    """Recursively replace modules with quantized modules."""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if "lm_head" in name:
                continue
            # Replace on the parent
            quantized = convert(child, quant_cfg, QuantLinear)
            setattr(model, name, quantized)
        replace_with_quant_linear(getattr(model, name), quant_cfg=quant_cfg)
    return model
```

- All `nn.Linear` layers (except `lm_head` by default) are recursively replaced with `QuantLinear`.
- The quantization configuration (`quant_cfg`) is retrieved from AutoRound schemes (`preset_name_to_scheme`).

### 2. `QuantLinear`

Defined in `neural_compressor/torch/algorithms/qat/quant_linear.py`:

```python
class QuantLinear(nn.Module):
    """Quantized version of nn.Linear."""

    def forward(self, input: torch.Tensor):
        qw = self.weight_quantizer(self.weight)
        qi = self.input_quantizer(input)
        out = F.linear(qi, qw, self.bias)
        out = self.output_quantizer(out)
        return out

    def _setup(self, quant_cfg):
        self.weight_quantizer = TensorQuantizer(
            data_type=quant_cfg.data_type,
            block_size=quant_cfg.group_size,
            bits=quant_cfg.bits,
            sym=quant_cfg.sym,
            if_quant=True,
            learn_exponent=False,
        )
        self.input_quantizer = TensorQuantizer(
            data_type=quant_cfg.act_data_type,
            block_size=quant_cfg.act_group_size,
            bits=quant_cfg.act_bits,
            sym=quant_cfg.act_sym,
            if_quant=True,
            learn_exponent=False,
        )
        self.output_quantizer = TensorQuantizer(
            data_type=quant_cfg.act_data_type,
            block_size=quant_cfg.act_group_size,
            bits=quant_cfg.act_bits,
            sym=quant_cfg.act_sym,
            if_quant=False,
        )
        # Disable output quantization for now
        self.output_quantizer.disable()
```

Key points:

- **Weight quantizer**: usually MXFP4 or MXFP8 with block-wise scaling.
- **Input quantizer**: activation quantization; configurable via AutoRound scheme.
- **Output quantizer**: currently disabled (acts as a pure passthrough). This can be extended later if full activation quantization is needed.

### 3. `TensorQuantizer`

Defined in `neural_compressor/torch/algorithms/qat/tensor_quantizer.py`, this module encapsulates the fake-quant logic:

```python
class TensorQuantizer(nn.Module):
    def __init__(
        self,
        data_type="mx_fp8",
        bits=8,
        block_size=32,
        sym=True,
        if_quant=True,
        learn_exponent=False,
        amax=None,
        scale_shape=None,
        device=None,
    ):
        ...
        assert get_quant_func is not None, "The quantization function is imported from AutoRound, please install it."

        self.quant_func, self.data_type = get_quant_func(self.data_type, self.num_bits, self.sym)
        ...
```

In the forward pass:

```python
def forward(self, inputs: torch.Tensor):
    if self._disabled or (not self._if_quant):
        self._input_dtype = inputs.dtype
        return inputs

    x = inputs.contiguous()
    if self.fake_quant:
        q = self._fake_quantize(x)[0]
    else:
        q = self._real_quantize(x)

    return q.to(inputs.dtype)
```

- Uses AutoRound’s `get_quant_func` to obtain the proper quantizer for `mx_fp4`, `mx_fp8`, etc.
- Supports block-wise exponent sharing (`block_size`) and optional saving of scales.
- For MX formats, `weight_pack` can pack weights and E8M0 scales to efficient storage formats.

### 4. QAT-Specific Trainer (`QATTrainer`)

Defined in `examples/.../qat/utils.py`:

```python
class QATTrainer(Trainer):
    def save_model(self, *args, **kwargs):
        # Handle FSDP state-dict types and dtype rewriting
        ...
        if (not self.is_in_train) and self.args.should_save:
            out_dir = args[0]
            self._update_config_json_dtype(out_dir, str(self._original_dtype).split(".")[1])
```

- Extends Hugging Face’s `Trainer` to handle:
  - FSDP full-state-dict export at the final checkpoint.
  - Ensuring that `config.json` contains the original model dtype (`dtype` or `torch_dtype`), which is important for downstream inference.

---

## Configuration and Key Parameters

### Command-Line Arguments (from `main.py`)

1. **ModelArguments**

```python
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.1-8B")
```

2. **TrainingArguments** (extends `transformers.TrainingArguments`)

Key fields:

- `model_max_length`: maximum sequence length (e.g., 2048, 4096).
- `dataloader_drop_last`: whether to drop the last batch (useful for distributed training).
- `bf16`: enable BF16 for training.

3. **DataArguments**

```python
@dataclass
class DataArguments:
    dataset: str = field(
        default="Daring-Anteater",
        metadata={"choices": ["Daring-Anteater", "cnn_dailymail"]},
    )
    train_size: int = 0  # 0 = default / automatic
    eval_size: int = 0
```


### AutoRound Quantization Schemes

The mapping from scheme names to quantization configs is handled by AutoRound:

```python
from auto_round.schemes import preset_name_to_scheme

quant_cfg = preset_name_to_scheme(scheme)
```

This `quant_cfg` is then used to initialize `TensorQuantizer` instances inside `QuantLinear`.

### Detecting Quantization Format

`get_quantization_format` inspects layers to determine the applied MX format:

```python
if weight_quantizer.num_bits == 8 and weight_quantizer.data_type == "mx_fp8":
    return "MXFP8"
if weight_quantizer.num_bits == 4 and weight_quantizer.data_type == "mx_fp4":
    return "MXFP4"
```

---

## Unified Fake-Quant QAT, Distillation (KD), QAD, and `compressed-tensors` Export

Intel Neural Compressor also provides a unified, self-contained fake-quantization, knowledge distillation, and Quantization-Aware Distillation (QAD) stack that does not require external quantization toolkits at training time and exports directly to **vLLM**-ready `compressed-tensors` checkpoints.

### Supported Quantization Formats & Triton QDQ Kernels

`FakeQuantConfig` (`neural_compressor.torch.quantization`) supports mixed per-layer / regex-targeted schemes across 5 native formats:

| Format | Block Size | Scale Representation | Dynamic / Static |
|---|---|---|---|
| `mxfp4` | 32 | `E8M0` power-of-two (`ceil(log2(max / 6.0))`) | Dynamic |
| `mxfp8` | 32 | `E8M0` power-of-two (`ceil(log2(max / 448.0))`) | Dynamic |
| `nvfp4` | 16 | Two-level FP8 `E4M3` block scale + global FP32 `amax` | Dynamic block (+ optional static global `amax`) |
| `fp8_e4m3` | Per-tensor | FP32 `amax / 448.0` | Static (`amax`) or Dynamic |
| `bf16` | — | No-op passthrough | Explicit layer exemption |

On GPU, `backend="auto"` dispatches to fused single-pass Triton QDQ kernels (`MXFP4`, `MXFP8`, `NVFP4`, `FP8_E4M3`) with **0-ULP bitwise parity** against reference PyTorch/CUDA implementations and **0 B** temporary workspace.

### Composable Python APIs

#### 1. Mixed-Precision QAT (`FakeQuantConfig` + `prepare_qat` + `QATTrainer`)

```python
from neural_compressor.torch.quantization import (
    FakeQuantConfig,
    QATTrainer,
    QuantSchemeRule,
    QuantizationTrainingArguments,
    prepare_qat,
)

quant_cfg = FakeQuantConfig(
    backend="auto",
    config_groups=[
        QuantSchemeRule(
            targets=[r"re:.*\.mlp\.(gate_proj|up_proj|down_proj)$"],
            weights="mxfp4",
            input_activations="mxfp4",
        ),
        QuantSchemeRule(
            targets=[r"re:.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"],
            weights={"format": "fp8_e4m3", "dynamic": False},
            input_activations={"format": "fp8_e4m3", "dynamic": False},
        ),
    ],
)

model = prepare_qat(model, quant_cfg)
trainer = QATTrainer(
    model=model,
    args=QuantizationTrainingArguments(output_dir="runs/qat", calib_batches=32, bf16=True),
    train_dataset=train_dataset,
)
trainer.train()
```

#### 2. Standalone Knowledge Distillation (`KDTrainer`)

`neural_compressor.torch.distillation` provides `KDTrainer` with chunked `lm_head` evaluation (`logits_chunk_size`), forward/reverse Top-$k$ or full-vocab KL (`TopKLogitsKLLoss`, `LogitsKLLoss`), automatic FSDP2 teacher layer sharding (`teacher_placement="shard"`), and zero-VRAM `teacher_model="self"` for LoRA self-distillation.

#### 3. Quantization-Aware Distillation (`QADTrainer` & PEFT `QuantLoraLinear`)

`neural_compressor.torch.qad` composes `KDTrainer` and `QATTrainer` via cooperative multiple inheritance, plus `prepare_qad_model` for PEFT LoRA integration (`lora_merge_before_quant=True` $\implies y = Q_w(W + s \cdot BA)\,Q_a(x)$):

```python
from neural_compressor.torch.qad import QADTrainer, QADTrainingArguments, prepare_qad_model

student = prepare_qad_model(
    student,
    quant_cfg,
    lora_config={
        "r": 32,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    lora_merge_before_quant=True,
)

trainer = QADTrainer(
    model=student,
    teacher_model="self",  # Zero extra teacher VRAM!
    args=QADTrainingArguments(
        output_dir="runs/qad_lora",
        calib_batches=32,
        kd_loss_type="topk_kl",
        kd_top_k=4096,
        kd_loss_weight=1.0,
        ce_loss_weight=0.0,
        logits_chunk_size=1024,
        bf16=True,
    ),
    train_dataset=train_dataset,
)
trainer.train()
```

#### 4. Real Packed `compressed-tensors` Export for vLLM

`export_hf_checkpoint` streams safetensors shards in bounded host RAM, merges optional PEFT LoRA adapters, packs real `MXFP4` (`uint8` nibbles + `E8M0` scales), `MXFP8`, `NVFP4`, or `FP8_E4M3` weights with a `compressed-tensors` `quantization_config` header, and preserves multimodal VLM vision towers (`keep_vision=True`):

```python
from neural_compressor.torch.export import export_hf_checkpoint

export_hf_checkpoint(
    "runs/qad_lora/checkpoint-500",
    "runs/export-mxfp4-fp8",
    export_format="compressed_tensors",
    quant_config=quant_cfg,
    quant_state_path="runs/qad_lora/checkpoint-500/quantizer_state.json",
    base_model_path="/path/to/Qwen3.8-27B",
    keep_vision=True,
)
```

See the complete end-to-end 27B recipes and launch scripts in [`examples/pytorch/llm/qwen/qad`](/examples/pytorch/llm/qwen/qad).

---

## Best Practices and Troubleshooting

### Recommended Workflow

1. **Start from a good FP32/BF16 baseline.**  
   Poor baselines are harder to recover with QAT.

2. **Optionally pre-quantize with AutoRound.**  
   - Use PTQ to get an initial quantized model (especially for MXFP4).
   - Then run QAT starting from this model to refine accuracy.

3. **Use small learning rates.**  
   - For QAT, start with `1e-5` or lower; higher learning rates can destabilize training.

4. **Monitor perplexity and accuracy during QAT.**  
   - `get_metrics_with_perplexity` adds perplexity to evaluation metrics for easy monitoring.

### Memory / Performance Issues

**Issue: Out-of-Memory (OOM) during QAT**

- Reduce `per_device_train_batch_size` and/or `model_max_length`.
- Enable gradient checkpointing (`--gradient_checkpointing True` in HF args, and ensure `gradient_checkpointing_kwargs` is set when needed).
- Use FSDP configuration (`fsdp1.yaml` or `fsdp2.yaml`) for sharded training:
  - `fsdp_activation_checkpointing: true`
  - `fsdp_cpu_ram_efficient_loading: true`

**Issue: QAT Training Is Too Slow**

- Reduce `max_steps` or number of epochs for initial experiments.
- Use fewer training samples (`train_size`).
- Ensure BF16 mixed precision is enabled on supported hardware.

### Accuracy Issues

**Issue: Large Accuracy Drop After QAT**

- Increase the number of training steps or epochs.
- Use a slightly higher precision scheme first (e.g., MXFP8), then transition to MXFP4 if needed.
- Ensure the dataset used for QAT matches your target task distribution as much as possible.

**Issue: Model Loading / Inference Errors**

- Verify the model directory contains:
  - `config.json`
  - `pytorch_model.bin` / `model.safetensors`
  - tokenizer files  
- Ensure your inference stack (e.g., vLLM) supports the quantization format produced by AutoRound/INC. Some runtimes may require additional export steps.

---

## References

- **Intel Neural Compressor (INC)**  
  [intel/neural-compressor](https://github.com/intel/neural-compressor)

- **AutoRound**  
  [intel/auto-round](https://github.com/intel/auto-round)

- **QAT LLM Example**  
  [qat example directory](/examples/pytorch/llm/others/qat)
