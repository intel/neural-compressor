
# PyTorch AutoRound

## Overview
AutoRound is an advanced weight-only quantization algorithm for low-bits LLM inference. It's tailored for a wide range of models and consistently delivers noticeable improvements, often significantly outperforming SignRound with the cost of more tuning time for quantization.

## Supported Features

- **Weight-Only Quantization (WoQ)** - Quantize model weights while keeping activations in full precision. See [Weight-Only Quantization](./PT_WeightOnlyQuantization.md) for details.

- **MX Quantization** - Neural Compressor seamlessly applies the MX data type to post-training quantization, offering meticulously crafted recipes to empower users to quantize LLMs without sacrificing accuracy. Refer to [MX Quantization](./PT_MXQuant.md).

- **NVFP4 Quantization** - NVFP4 is a specialized 4-bit floating-point format (FP4) developed by NVIDIA for deep learning workloads. See [NVFP4 Quantization](./PT_NVFloat4Quant.md).

- **Quantization-Aware Training (QAT)** - Fine-tune models during quantization to achieve better accuracy. See [Quantization-Aware Training](./PT_QuantizationAwareTraining.md) for details.

## Getting Started

```python
from neural_compressor.torch.quantization import prepare, convert, AutoRoundConfig

quant_config = AutoRoundConfig(tokenizer=tokenizer)  # tokenizer used for calibration
model = prepare(model, quant_config)
model = convert(model)

# For more detailed usage, please refer to the [Supported Features] documentation.
```

## Reference

[1]. Cheng, Wenhua, et al. "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs" arXiv preprint arXiv:2309.05516 (2023).

[2]: NVIDIA, Introducing NVFP4 for efficient and accurate low-precision inference,NVIDIA Developer Blog, Jun. 2025. [Online]. Available: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

[3]. Intel AutoRound, https://github.com/intel/auto-round
