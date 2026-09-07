# vllm-qdq-plugin

Out-of-tree [vLLM](https://github.com/vllm-project/vllm) plugins for activation quant-dequant (QDQ) simulation and Sage3 Triton diffusion attention. The QDQ plugin makes it possible to study the accuracy impact of real quantized compute compared with weight-only dequantization.

## Contents

- [Quick Start](#quick-start)
- [QDQ Plugin](#qdq-plugin)
  - [Supported Formats](#supported-formats)
  - [NVFP4_E5M3](#format-and-runtime-behavior)
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
| `VLLM_QDQ_CUTE` | `0` | Enable fused CuTe MXFP4/MXFP8 QDQ kernels. Requires CUDA, SM80+, NVIDIA CUTLASS DSL, contiguous input, group size 32, and `K` divisible by 32. Unsupported inputs fall back to the reference implementation. |
| `VLLM_MARLIN_MOE_QDQ_MODE` | `0` | Set to `FORCE_MXFP4` to apply MXFP4 QDQ in `moe_wna16_marlin_gemm` when dtype-based routing is not sufficient. Matching is case-insensitive. |

For diagnostics, add `VLLM_QDQ_TRACE=1` to print up to 200 QDQ shape and dtype trace lines. To force MXFP4 QDQ for Marlin MoE when dtype detection is insufficient, add `VLLM_MARLIN_MOE_QDQ_MODE=FORCE_MXFP4`.

### Supported Formats

| Dtype | Op | Status | Notes |
| --- | --- | --- | --- |
| **MXFP4** (E2M1 + E8M0 scales) | `marlin_gemm` | ✅ Supported | Dense quantized linear (MXFP4 via Marlin) |
| **MXFP4** (E2M1 + E8M0 scales) | `moe_wna16_marlin_gemm` | ✅ Supported | MoE quantized linear (MXFP4 via Marlin) |
| **NVFP4_E5M3** (E2M1 + UE5M3 scales) | dense linear | ✅ Supported | AutoRound `nvfp4_v2`, group size 16, vLLM FP4 Marlin |
| **NVFP4_E5M3** (E2M1 + UE5M3 scales) | fused MoE | ✅ Supported | AutoRound `nvfp4_v2`, group size 16, vLLM FP4 Marlin experts |

### NVFP4_E5M3

#### Format and Runtime Behavior

AutoRound checkpoints can use `data_type: nvfp4_v2` globally or override selected
layers such as `mlp.experts` in `extra_config`. The plugin accepts both
`auto_round:llm_compressor` and
`auto_round:llm_compressor_nvfp4_e5m3` packing formats. It preserves the
checkpoint's raw `uint8` E2M1 payload and UE5M3 block scales. Dense and MoE layers
decode the raw UE5M3 scale bytes once during loading, then reuse vLLM's FP4 Marlin
weight repacking, scale processing, workspace management, GEMMs, expert routing,
and top-k reduction. Activations retain the checkpoint's UE5M3 + E2M1 QDQ semantics
before dense GEMMs and before both MoE expert GEMMs. This path requires group size 16.

#### Local Model Commands

```bash
source /path/to/venv/bin/activate
MODEL_PATH=/path/to/nvfp4_e5m3_model
CUDA_VISIBLE_DEVICES=<idle-gpu> vllm serve "$MODEL_PATH" \
  --dtype bfloat16 --trust-remote-code

# Or run the spawn-safe one-prompt verification:
CUDA_VISIBLE_DEVICES=<idle-gpu> python scripts/test_nvfp4_ue5m3_model.py \
  "$MODEL_PATH"

# Run with the default vLLM TorchDynamo/AOT and CUDA Graph configuration:
CUDA_VISIBLE_DEVICES=<idle-gpu> VLLM_QDQ=1 VLLM_QDQ_CUTE=1 vllm bench throughput \
  --model "$MODEL_PATH" \
  --dataset-name random --num-prompts 200 \
  --random-input-len 512 --random-output-len 128
```

#### Limitations

- Dense linear and fused MoE require BF16 activations.
- MoE does not support `apply_router_weight_on_input` or EPLB.
- The fused path supports vLLM TorchDynamo/AOT compilation and CUDA Graph capture.

#### Validation Status

Validation was performed in a local virtual environment:

- Target checkpoint tensors were confirmed as `uint8 [N, K/2]` weights and `uint8 [N, K/16]` UE5M3 scales.
- Configuration routing tests: `3 passed`.
- CuTe activation QDQ and weight dequantization have pure-Torch parity references and fake-aware custom-op boundaries for `torch.compile`.
- The batched Triton MoE matched the former per-expert reference for group size 16 and 32 with cosine similarity above 0.9999 and relative L2 error below 1%.
- On one A100 with the target 256-expert, hidden-size 2048, intermediate-size 512, top-k 8 shape, warmed MoE latency improved from 22.404 ms to 1.068 ms at `M=1` (20.97x) and from 255.390 ms to 2.471 ms at `M=64` (103.37x).
- CuTe parity, fullgraph, CUDA Graph, and real-model validation for the new dense path are pending an idle GPU.

UE5M3 activation scales use bit-level round-to-nearest-even conversion. Stored checkpoint scales are decoded directly from their existing bytes.

### Performance and Accuracy

All results in this section are internal observations from one local environment. They are included for development and reproducibility only; they are not official benchmark results, independently verified results, performance guarantees, or claims about other hardware or software environments.

#### CuTe QDQ Microbenchmark

The optional CuTe backend accepts CUDA devices with SM80 or newer. It requires CUDA, NVIDIA CUTLASS DSL, contiguous input, group size 32, and a `K` dimension divisible by 32. Unsupported inputs use the reference implementation.

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

#### GSM8K: MXFP4 vs NVFP4_E5M3

Model: Qwen3.6-35B-A3B. Task: GSM8K v3, 5-shot. The runs use the vLLM backend, automatic batch size, chat template enabled, thinking disabled, tensor parallel size 1, data parallel size 1, maximum model length 8192, and expert parallelism enabled. Quantized runs use the CuTe QDQ backend.

| Format | Evaluated checkpoint | Flexible exact match | Strict exact match | Prompts/s | Input tokens/s | Output tokens/s | Elapsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | Original BF16 model | 0.8863 +/- 0.0087 | 0.8772 +/- 0.0090 | / | / | / | / |
| MXFP4 | AutoRound MXFP4 checkpoint | 0.8666 +/- 0.0094 | 0.8499 +/- 0.0098 | 9.71 | 10,346.38 | 1,511.55 | 2m 15s |
| NVFP4_E5M3 | AutoRound NVFP4_E5M3 checkpoint | 0.8726 +/- 0.0092 | 0.8560 +/- 0.0097 | 9.22 | 9,824.72 | 1,459.98 | 2m 23s |

| Comparison | Flexible exact match | Strict exact match | Prompts/s | Input tokens/s | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXFP4 vs BF16 | -1.97 pp | -2.73 pp | N/A | N/A | N/A |
| NVFP4_E5M3 vs BF16 | -1.37 pp | -2.12 pp | N/A | N/A | N/A |
| NVFP4_E5M3 vs MXFP4 | **+0.60 pp** | **+0.60 pp** | -5.0% | -5.0% | -3.4% |

NVFP4_E5M3 recovers 0.60 percentage points on both accuracy metrics relative to MXFP4 in this run. MXFP4 is about 5% faster for prompt and input-token throughput. The BF16 run has no retained timing data and is included only as an accuracy baseline. GPU differences, nondeterminism, runtime state, and dependency versions can affect these measurements.

##### Reproduce Quantization and Evaluation

Use the same commands for MXFP4 and NVFP4_E5M3; change only `SCHEME`.

```bash
MODEL_PATH=/path/to/Qwen3.6-35B-A3B/
SCHEME=nvfp4_e5m3  # Use mxfp4 for MXFP4.
auto-round "$MODEL_PATH" \
  --format auto_round \
  --scheme "$SCHEME" \
  --output_dir "/path/to/output/qwen3.6-moe-${SCHEME}" \
  --ignore_layers shared_expert_gate \
  --model_free
```

```bash
SCHEME=nvfp4_e5m3  # Use the same SCHEME value as quantization.
MODEL_PATH=/path/to/output/qwen3.6-moe-${SCHEME}
VLLM_QDQ=1 VLLM_QDQ_CUTE=1 CUDA_VISIBLE_DEVICES=<idle-gpu> \
  lm_eval --model vllm \
  --model_args pretrained="$MODEL_PATH",tensor_parallel_size=1,data_parallel_size=1,max_model_len=8192,enable_expert_parallel=True,trust_remote_code=True,enable_thinking=False \
  --tasks gsm8k \
  --batch_size auto \
  --apply_chat_template
```

For the BF16 baseline, point `MODEL_PATH` to the original checkpoint and omit `VLLM_QDQ=1 VLLM_QDQ_CUTE=1`.

### Implementation Notes

#### NVFP4_E5M3 Components

| File | Purpose |
| --- | --- |
| `src/nvfp4_hw/patch.py` | Register the NVFP4_E5M3 packing format, preserve per-layer `extra_config.data_type`, and route `nvfp4_v2` layers without changing vLLM. |
| `src/nvfp4_hw/inc_nvfp4_ue5m3_scheme.py` | Select the NVFP4_E5M3 dense linear or fused MoE implementation. |
| `src/nvfp4_hw/inc_nvfp4_ue5m3_linear.py` | Load TP-aware packed weights, dequantize them to BF16 with CuTe, apply activation QDQ, and dispatch vLLM's selected BF16 GEMM. |
| `src/nvfp4_hw/inc_nvfp4_ue5m3_moe.py` | Load raw packed expert tensors and dispatch top-k MoE execution to the batched kernel. |
| `src/nvfp4_hw/fused_moe_ue5m3.py` | Align routed tokens and run gate/up and down projections as two batched Triton expert launches with in-kernel E2M1 and UE5M3 decoding. |
| `src/vllm_qdq_plugin/qdq/nvfp4_e5m3.py` | Apply E2M1 activation QDQ with groupwise UE5M3 scales, using CuTe on supported CUDA tensors and a pure-Torch correctness reference otherwise. |
| `src/vllm_qdq_plugin/qdq/nvfp4_e5m3_cute.py` | Register fake-aware CuTe activation-QDQ and weight-dequant custom ops for TorchDynamo. |
| `src/vllm_qdq_plugin/qdq/cute_kernels.py` | Implement NVFP4_E5M3 activation QDQ and packed-weight dequantization in CuTe DSL. |
| `tests/test_nvfp4_ue5m3.py` | Verify global and mixed `nvfp4_v2` routing and NVFP4 QDQ fullgraph compilation. |
| `tests/test_fused_moe_ue5m3.py` | Verify fused MoE parity for group size 16/32, Dynamo fullgraph capture, and CUDA Graph replay. |
| `scripts/test_nvfp4_ue5m3_model.py` | Reproducibly load a local model and run one prompt under vLLM's spawn worker mode. |
| `pyproject.toml` | Declare the NVIDIA CUTLASS DSL runtime dependency. |

The previous native NVFP4/Marlin path was replaced. Dense GEMM runs on dequantized BF16 weights through vLLM's unquantized dispatcher.

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
