# vllm-qdq-plugin

Out-of-tree [vLLM](https://github.com/vllm-project/vllm) plugins for activation quant-dequant (QDQ) simulation and Sage3 Triton diffusion attention. The QDQ plugin makes it possible to study the accuracy impact of real quantized compute compared with weight-only dequantization.

## Contents

- [Quick Start](#quick-start)
- [QDQ Plugin](#qdq-plugin)
  - [Supported Formats](#supported-formats)
  - [NVFP4](#nvfp4)
  - [Performance and Accuracy](#performance-and-accuracy)
  - [Implementation Notes](#implementation-notes)
- [Sage3 Triton Attention](#sage3-triton-attention)
- [Adding New Dtypes](#adding-new-dtypes)

## Quick Start

Install the plugin into the same environment that runs vLLM:

```bash
# From the neural-compressor repository root
pip install -e benchmark/vllm-qdq-plugin/

# Or, when already inside benchmark/vllm-qdq-plugin
pip install -e .
```

NVIDIA CUTLASS DSL is optional. Install it only when using the CuTe backend (VLLM_QDQ_CUTE=1):

```bash
pip install 'nvidia-cutlass-dsl>=4.6.0'
```

Enable QDQ for a script, a vLLM server, or a local evaluation. CuTe is the recommended backend on supported NVIDIA GPUs:

```bash
# Script
VLLM_QDQ=1 python my_script.py

# vLLM server
VLLM_QDQ=1 VLLM_QDQ_CUTE=1 vllm serve /path/to/model --tensor-parallel-size 2

# Evaluation
VLLM_QDQ=1 VLLM_QDQ_CUTE=1 CUDA_VISIBLE_DEVICES=<idle-gpu> \
  lm_eval --model vllm --model_args pretrained=/path/to/model,...
```

## QDQ Plugin

### What It Does

The plugin registers as a `vllm.general_plugins` entry point. vLLM loads it in all processes, including workers. It monkey-patches low-level wrappers in `vllm._custom_ops` to inject activation QDQ before the quantized GEMM call.

- Zero vLLM source modifications
- Works with the `LLM()` Python API and `vllm serve`
- Covers dense linear, MoE gate-and-up, and MoE down call sites

### Runtime Controls

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_QDQ` | `0` | Set to `1` to enable QDQ. |
| `VLLM_QDQ_TRACE` | `0` | Set to `1` to print up to 200 QDQ shape and dtype trace lines. |
| `VLLM_QDQ_CUTE` | automatic | CuTe is selected automatically for MXFP4, MXFP8, and NVFP4_E5M3 when the input is on an NVIDIA CUDA GPU with SM80+, NVIDIA CUTLASS DSL is installed, and the format-specific shape requirements are met. In automatic mode, a missing CUTLASS DSL installation warns with an install command before falling back to the reference implementation. Set to `0` to force the reference implementation or `1` to explicitly require CuTe; an explicit request raises an error when CUTLASS DSL is missing. Other unsupported CuTe conditions warn before using the slower reference implementation. |
| `VLLM_MARLIN_MOE_QDQ_MODE` | `0` | Set to `FORCE_MXFP4` to apply MXFP4 QDQ in `moe_wna16_marlin_gemm` when dtype-based routing is not sufficient. Matching is case-insensitive. |

For diagnostics, add `VLLM_QDQ_TRACE=1` to print up to 200 QDQ shape and dtype trace lines. To force MXFP4 QDQ for Marlin MoE when dtype detection is insufficient, add `VLLM_MARLIN_MOE_QDQ_MODE=FORCE_MXFP4`.

### Supported Formats

| Dtype | Op | Status | Notes |
| --- | --- | --- | --- |
| **MXFP4** (E2M1 + E8M0 scales) | `marlin_gemm` | ✅ Supported | Dense quantized linear (MXFP4 via Marlin) |
| **MXFP4** (E2M1 + E8M0 scales) | `moe_wna16_marlin_gemm` | ✅ Supported | MoE quantized linear (MXFP4 via Marlin) |
| **NVFP4_E5M3** (E2M1 + UE5M3 scales) | dense linear | ✅ Supported | AutoRound `nvfp4_v2`, group size 16, vLLM FP4 Marlin |
| **NVFP4_E5M3** (E2M1 + UE5M3 scales) | fused MoE | ✅ Supported | AutoRound `nvfp4_v2`, group size 16, vLLM FP4 Marlin experts |

### NVFP4

The plugin supports both vLLM-native NVFP4 checkpoints and AutoRound
NVFP4_E5M3 (`nvfp4_v2`) checkpoints for dense and MoE layers. See the
[NVFP4 implementation guide](src/nvfp4_hw/README.md) for format details,
runtime selection, commands, limitations, validation results, and source layout.

### Performance and Accuracy

> **Disclaimer:** All accuracy and performance data in this section are internal observations from one local environment and are included only to demonstrate the plugin's behavior. They are not official published results, independently verified benchmarks, performance guarantees, or claims about results on other hardware or software environments.

#### CuTe QDQ Microbenchmark

The CuTe backend is selected automatically on NVIDIA CUDA devices with SM80 or newer when NVIDIA CUTLASS DSL is installed. If the GPU requirements are met but CUTLASS DSL is missing, automatic mode warns with the installation command and uses the slower reference implementation. Setting `VLLM_QDQ_CUTE=1` makes the dependency mandatory and raises an error when it is missing. MXFP4/MXFP8 additionally require group size 32 and a `K` dimension divisible by 32. Non-contiguous inputs are materialized as contiguous tensors before the CuTe kernel runs; set `VLLM_QDQ_TRACE=1` to print their shape, stride, dtype, device, and storage offset. When another requirement is not met, the plugin warns with the reason and uses the reference implementation. Set `VLLM_QDQ_CUTE=0` to force reference QDQ without a warning. NVFP4-specific requirements are documented in the [NVFP4 implementation guide](src/nvfp4_hw/README.md).

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> python scripts/verify_cute_dsl.py
CUDA_VISIBLE_DEVICES=<idle-gpu> python scripts/bench_qdq_cute.py --shape 1024 4096
```

The first call includes CuTe JIT compilation. The following warmed-up results use 20 warmup and 100 measured iterations on an NVIDIA A100 with `[1024, 4096]` BF16 input; the fused outputs exactly matched the reference.

| Format | Reference | CuTe | Speedup |
| --- | ---: | ---: | ---: |
| MXFP4 | 1.437 ms | 0.088 ms | 16.32x |
| MXFP8 | 0.859 ms | 0.087 ms | 9.83x |

#### vLLM Throughput

Model: Qwen3.6-35B-A3B MXFP4. Hardware: 1x A100. Workload: 200 random prompts, 512 input tokens, and 128 output tokens per prompt.

```bash
MODEL_PATH=/path/to/your/model
CUDA_VISIBLE_DEVICES=<idle-gpu> VLLM_QDQ=1 VLLM_QDQ_CUTE=1 vllm bench throughput \
  --model "$MODEL_PATH" \
  --dataset-name random \
  --num-prompts 200 \
  --random-input-len 512 \
  --random-output-len 128
```

| QDQ backend | Requests/s | Total tokens/s | Output tokens/s | Prompt tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference (`VLLM_QDQ_CUTE=0`) | 7.46 | 4,773.13 | 954.63 | 102,400 | 25,600 |
| CuTe (`VLLM_QDQ_CUTE=1`) | 16.76 | 10,723.57 | 2,144.71 | 102,400 | 25,600 |

### Implementation Notes

#### MXFP4 QDQ Semantics

For MXFP4, the QDQ simulates:

1. **Quantize**: Scale activations per group of 32 using E8M0 (power-of-2) scales, then round to nearest FP4 E2M1 value `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`
2. **Dequantize**: Multiply back by the scale to restore the original dtype

This introduces the same quantization noise that a "real" MXFP4 GEMM would produce on the input side, while keeping the actual computation in bf16 via Marlin's weight-only dequant kernel.

#### CUDA Graphs

The fused QDQ kernels support CUDA Graph capture and replay, including the graph path used by vLLM. Each `(format, dtype, shape, device)` specialization must execute once in eager mode before capture so CuTe JIT compilation stays outside the graph. vLLM's normal warmup satisfies this requirement. A cache miss during capture raises an actionable error instead of attempting an unsafe JIT compilation.

Compiled specializations are cached per CUDA device. Kernel launches use PyTorch's current CUDA stream, so capture records the QDQ kernel in the same graph as the following Marlin operation.

The CuTe launchers are registered as `torch.library.custom_op` operators with FakeTensor implementations. This keeps Python capability checks and the CUTLASS runtime outside `torch.compile(fullgraph=True)` while allowing vLLM AOT compilation to retain QDQ as an opaque graph node.

## Adding New Dtypes

1. Create a new QDQ implementation in `src/vllm_qdq_plugin/qdq/` (e.g., `fp8.py`)
2. Add an `elif` branch in `patch.py` where the dtype check happens
3. The QDQ function signature: `(x: Tensor, **config) -> Tensor` — same shape and dtype in/out

## Sage3 Triton Attention

This plugin also provides an **out-of-tree diffusion attention backend** for [vllm-omni](https://github.com/vllm-project/vllm-omni), using the [SageAttention3](https://github.com/thu-ml/SageAttention) standalone Triton kernel.

### Overview

Registers via the `vllm_omni.general_plugins` entry_point. When `VLLM_SAGE3_TRITON=1`, overrides the `SAGE_ATTN` diffusion attention backend with the sage3 Triton implementation. When disabled (default), the original in-tree backend is used unchanged.

- Zero vllm-omni source modifications
- Conditional activation - does not affect normal operation when off
- Falls back to torch SDPA for cross-attention (different Q/K sequence lengths)

### Usage

```bash
# Enable sage3 Triton attention for diffusion models
VLLM_SAGE3_TRITON=1 \
SAGE3_QUANT_FORMAT=mxfp4 \
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN \
python examples/offline_inference/text_to_image/text_to_image.py \
  --model /path/to/model ...

# Use the original in-tree sage_attn (sageattention v2) - default
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN python ...

# Use torch SDPA (no sage at all)
DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA python ...
```

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_SAGE3_TRITON` | `0` | Set to `1` to enable sage3 Triton backend override |
| `SAGE3_QUANT_FORMAT` | `mxfp4` | Quantization config for K/V (`mxfp4`, `nvfp4`, `mxfp8_s1`, `mxfp4_s1`) |
| `SAGE3_ACC_DTYPE` | `fp32` | Accumulator dtype (`fp32`, `bf16_both_dot`, `bf16_pv_only`, etc.) |

### Notes

- **Shared memory requirement**: The sage3 fp32 kernel needs ~192KB shared memory per SM. On GPUs with less (e.g., RTX 6000D with 100KB), use `SAGE3_ACC_DTYPE=bf16_both_dot` or switch to TORCH_SDPA.
- **Cross-attention**: sage3 requires Q and K to have the same sequence length. Cross-attention calls automatically fall back to torch SDPA.

## License

Apache-2.0
