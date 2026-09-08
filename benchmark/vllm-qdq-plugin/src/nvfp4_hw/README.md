# vLLM INC NVFP4 Support

The `vllm_nvfp4_support` folder provides an out-of-tree vLLM plugin for running
AutoRound NVFP4 checkpoints through vLLM's INC quantization path. It supports
4-bit W4A4 weights with `nv_fp` metadata and group size 16, without modifying
the vLLM installation.

The plugin is loaded through the `vllm.general_plugins` entry-point mechanism.
After installation, users can load a supported NVFP4 checkpoint with vLLM or
with an evaluation tool that uses the vLLM backend.

## Supported functionality

- AutoRound NVFP4 packed E2M1 weights.
- FP8 E4M3 per-group weight scales.
- Weight and input global scales.
- Dense Linear layers.
- MoE expert layers, including fused `w13` and `w2` weights.
- vLLM's native NVFP4 linear and MoE kernels.

## Code structure

The folder contains the following implementation components:

- `patch.py`: Registers the plugin and adds NVFP4 metadata support to the INC
  quantization path.
- `inc_nvfp4_scheme.py`: Selects the appropriate dense or MoE NVFP4 method for
  each layer.
- `inc_nvfp4_linear.py`: Loads NVFP4 weights and runs dense Linear layers.
- `inc_nvfp4_moe.py`: Loads MoE expert weights and runs the fused MoE path.
- `__init__.py`: Exposes the NVFP4 support components as a Python package.

## Installation

Install the plugin in editable mode:

```bash
python -m pip install -e /path/to/vllm-qdq-plugin --no-deps
```

The package declares the following vLLM general plugin entry point:

```text
inc_nvfp4 = vllm_nvfp4_support.patch:register
```

## Usage

After installation, the plugin is loaded automatically by vLLM through the
`vllm.general_plugins` entry point. The following example loads an AutoRound
NVFP4 checkpoint through vLLM:

```python
from vllm import LLM

llm = LLM(
    model="/path/to/autoround-nvfp4-checkpoint",
    dtype="bfloat16",
)
```

No changes to the vLLM source code or model code are required. The plugin
recognizes the NVFP4 metadata, selects the dense or MoE implementation for
each layer, and passes the packed weights and scales to the corresponding
vLLM NVFP4 kernel.

## Evaluation results

The following results were obtained with the vLLM backend. PIQA uses zero-shot
evaluation and GSM8K uses five-shot evaluation.

| Model | Task | Metric | Score |
|---|---|---|---:|
| Qwen3-8B-NVFP4 | PIQA | `acc` | 76.50% |
| Qwen3-8B-NVFP4 | PIQA | `acc_norm` | 77.20% |
| Qwen3-8B-NVFP4 | GSM8K | `exact_match` (strict) | 86.73% |
| Qwen3-8B-NVFP4 | GSM8K | `exact_match` (flexible) | 87.26% |
| Qwen3-30B-A3B-NVFP4 | PIQA | `acc` | 79.16% |
| Qwen3-30B-A3B-NVFP4 | PIQA | `acc_norm` | 80.14% |
| Qwen3-30B-A3B-NVFP4 | GSM8K | `exact_match` (strict) | 88.48% |
| Qwen3-30B-A3B-NVFP4 | GSM8K | `exact_match` (flexible) | 89.23% |
