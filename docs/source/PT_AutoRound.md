
# PyTorch AutoRound

## Overview
AutoRound is an advanced model quantization algorithm integrated into Neural Compressor for low-bit LLM. As a key algorithm component of INC, AutoRound enables efficient quantization across a wide range of models and features while consistently achieving superior accuracy. While requiring additional tuning time, it provides a robust foundation for INC's comprehensive quantization capabilities.

## Supported Features

- **Weight-Only Quantization (WoQ)** - Quantize model weights while keeping activations in full precision. See [Weight-Only Quantization](./PT_WeightOnlyQuant.md) for details.

- **Microscaling (MX) Quantization** - Neural Compressor seamlessly applies the MX data type to post-training quantization, offering meticulously crafted recipes to empower users to quantize LLMs without sacrificing accuracy. Refer to [MX Quantization](./PT_MXQuant.md).

- **NVFP4 Quantization** - NVFP4 is a specialized 4-bit floating-point format (FP4) developed by NVIDIA for deep learning workloads. See [NVFP4 Quantization](./PT_NVFP4Quant.md).

- **Quantization-Aware Training (QAT)** - Fine-tune models during quantization to achieve better accuracy. See [Quantization-Aware Training](./PT_QAT.md) for details.

- **FP8 KV Cache and Attention Static Quantization (Experimental)** - The support for the FP8 data type enhances inference performance by quantizing key-value cache and attention computations to FP8 precision.

## Getting Started

### Basic Usage

```python
from neural_compressor.torch.quantization import prepare, convert, AutoRoundConfig

quant_config = AutoRoundConfig(tokenizer=tokenizer)  # tokenizer used for calibration
model = prepare(model, quant_config)
model = convert(model)

# For more detailed usage, please refer to the [Supported Features] documentation.
```
### FP8 KV Cache and FP8 Attention support
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    convert,
    prepare,
)

fp32_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)

output_dir = "./saved_inc"
quant_config = AutoRoundConfig(
    tokenizer=tokenizer,
    scheme="MXFP4",  # MXFP4, MXFP8, NVFP4
    iters=0,  # rtn mode
    seqlen=2,
    static_kv_dtype="fp8",  # None, fp8, float16
    static_attention_dtype=None,  # None, fp8
    export_format="auto_round",
    output_dir=output_dir,
)

model = prepare(model=fp32_model, quant_config=quant_config)
model = convert(model)
```

## Reference

[1]. Cheng, Wenhua, et al. "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs" arXiv preprint arXiv:2309.05516 (2023).

[2]: NVIDIA, Introducing NVFP4 for efficient and accurate low-precision inference,NVIDIA Developer Blog, Jun. 2025. [Online]. Available: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

[3]. Intel AutoRound, https://github.com/intel/auto-round
