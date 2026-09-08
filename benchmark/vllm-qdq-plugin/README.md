# vllm-qdq-plugin

Out-of-tree [vLLM](https://github.com/vllm-project/vllm) plugin that simulates activation quant-dequant (QDQ) before quantized GEMM kernels. Useful for studying the accuracy impact of "real" quantized compute vs weight-only dequant approaches.

## How It Works

The plugin registers as a `vllm.general_plugins` entry point, which vLLM loads automatically in **all processes** (main + workers). It monkey-patches the low-level op wrappers in `vllm._custom_ops` to inject QDQ on input activations before the actual kernel call. This means:

- Zero vLLM source modifications
- Works with both `LLM()` Python API and `vllm serve`
- Covers all call sites automatically (dense linear, MoE gate+up, MoE down)

## Installation

```bash
pip install git+https://github.com/yiliu30/vllm-qdq-plugin.git

# Or install from the neural-compressor repository root for development:
pip install -e benchmark/vllm-qdq-plugin/

# When already inside benchmark/vllm-qdq-plugin:
pip install -e .
```

## Usage

```bash
# Enable QDQ
VLLM_QDQ=1 python my_script.py

# With vllm serve
VLLM_QDQ=1 vllm serve /path/to/model --tensor-parallel-size 2

# Enable trace logging (prints shape/dtype for each QDQ call)
VLLM_QDQ=1 VLLM_QDQ_TRACE=1 vllm serve /path/to/model

# Request the CuTe QDQ backend on SM80+ GPUs
VLLM_QDQ=1 VLLM_QDQ_CUTE=1 vllm serve /path/to/model

# Force MXFP4 QDQ on Marlin MoE when dtype-based detection is not enough
VLLM_QDQ=1 VLLM_MARLIN_MOE_QDQ_MODE=FORCE_MXFP4 vllm serve /path/to/model
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_QDQ` | `0` | Set to `1` to enable QDQ |
| `VLLM_QDQ_TRACE` | `0` | Set to `1` to print trace lines (up to 200) |
| `VLLM_QDQ_CUTE` | `0` | Enable fused CuTe MXFP4/MXFP8 QDQ kernels. Requires CUDA, SM80+, NVIDIA CUTLASS DSL, contiguous input, group size 32, and `K` divisible by 32. Unsupported inputs fall back to the reference implementation. |
| `VLLM_MARLIN_MOE_QDQ_MODE` | `0` | Set to `FORCE_MXFP4` to apply MXFP4 QDQ in `moe_wna16_marlin_gemm` when dtype-based routing is not sufficient. Matching is case-insensitive. |

## Support Status

| Dtype | Op | Status | Notes |
|---|---|---|---|
| **MXFP4** (E2M1 + E8M0 scales) | `marlin_gemm` | ✅ Supported | Dense quantized linear (MXFP4 via Marlin) |
| **MXFP4** (E2M1 + E8M0 scales) | `moe_wna16_marlin_gemm` | ✅ Supported | MoE quantized linear (MXFP4 via Marlin) |

### How QDQ Works

For MXFP4, the QDQ simulates:
1. **Quantize**: Scale activations per group of 32 using E8M0 (power-of-2) scales, then round to nearest FP4 E2M1 value `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`
2. **Dequantize**: Multiply back by the scale to restore the original dtype

This introduces the same quantization noise that a "real" MXFP4 GEMM would produce on the input side, while keeping the actual computation in bf16 via Marlin's weight-only dequant kernel.

### CuTe QDQ Validation

The optional CuTe backend checks CUDA capability at runtime and only accepts SM80 or newer GPUs. Verify the installed CuTe DSL first, then compare the CuTe and reference paths:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> python scripts/verify_cute_dsl.py
CUDA_VISIBLE_DEVICES=<idle-gpu> python scripts/bench_qdq_cute.py --shape 1024 4096
```

The benchmark reports exact output equality, maximum absolute error, latency, and speedup for MXFP4 and MXFP8. On an NVIDIA A100 with a `[1024, 4096]` bf16 input, 20 warmup iterations, and 100 measured iterations, the fused kernels produced exact reference output and measured:

| Format | Reference | CuTe | Speedup |
|---|---:|---:|---:|
| MXFP4 | 1.437 ms | 0.088 ms | 16.32x |
| MXFP8 | 0.859 ms | 0.087 ms | 9.83x |

The first call includes CuTe JIT compilation. The table measures warmed-up steady-state execution.

#### Internal Test Results

The following measurements are internal test observations from one local environment. They are included only to aid development and reproducibility. They are not released benchmark results, independently verified results, official model evaluations, performance guarantees, or claims about results in other hardware and software environments.

##### vLLM Throughput

Model example: (Qwen3.6-35B-A3B MXFP4); GPU: 1x A100; workload: 200 random prompts, 512 input tokens and 128 output tokens per prompt.

Reproduce (set your own model path):

```bash
MODEL_PATH=/path/to/your/model
CUDA_VISIBLE_DEVICES=2 VLLM_QDQ=1 VLLM_QDQ_CUTE=1 vllm bench throughput \
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

##### GSM8K Evaluation

Model example: (Qwen3.6-35B-A3B MXFP4); task: GSM8K v3, 5-shot; `lm_eval` with the vLLM backend, automatic batch size, tensor parallel size 1, data parallel size 1, maximum model length 8192, and expert parallelism enabled.

Reproduce (set your own model path):

```bash
MODEL_PATH=/path/to/your/model
CUDA_VISIBLE_DEVICES=1 VLLM_QDQ=1 VLLM_QDQ_CUTE=1 lm_eval --model vllm \
  --model_args pretrained="$MODEL_PATH",tensor_parallel_size=1,data_parallel_size=1,max_model_len=8192,enable_expert_parallel=True,trust_remote_code=True \
  --tasks gsm8k \
  --batch_size auto
```

| QDQ backend | GPU | Flexible-extract exact match | Stderr | Strict-match exact match | Stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference (`VLLM_QDQ_CUTE=0`) | 1 | 0.7589 | 0.0118 | 0.7392 | 0.0121 |
| CuTe (`VLLM_QDQ_CUTE=1`) | 2 | 0.7635 | 0.0117 | 0.7407 | 0.0121 |

The GSM8K runs used different GPU devices and may also be affected by nondeterminism, runtime state, and dependency versions. The small score differences are within the reported uncertainty and should not be interpreted as evidence that either backend improves model accuracy. Reproduce the measurements under a controlled environment before drawing conclusions or publishing comparisons.

### CUDA Graphs

The fused QDQ kernels support CUDA Graph capture and replay, including the graph path used by vLLM. Each `(format, dtype, shape, device)` specialization must execute once in eager mode before capture so CuTe JIT compilation stays outside the graph. vLLM's normal warmup satisfies this requirement. A cache miss during capture raises an actionable error instead of attempting an unsafe JIT compilation.

Compiled specializations are cached per CUDA device. Kernel launches use PyTorch's current CUDA stream, so capture records the QDQ kernel in the same graph as the following Marlin operation.

The CuTe launchers are registered as `torch.library.custom_op` operators with FakeTensor implementations. This keeps Python capability checks and the CUTLASS runtime outside `torch.compile(fullgraph=True)` while allowing vLLM AOT compilation to retain QDQ as an opaque graph node.

After updating the plugin, reinstall it in the same virtual environment that runs vLLM:

```bash
uv pip install -e /path/to/neural-compressor/benchmark/vllm-qdq-plugin
```

## Adding New Dtypes

1. Create a new QDQ implementation in `src/vllm_qdq_plugin/qdq/` (e.g., `fp8.py`)
2. Add an `elif` branch in `patch.py` where the dtype check happens
3. The QDQ function signature: `(x: Tensor, **config) -> Tensor` — same shape and dtype in/out

## License

Apache-2.0

---

## Sage3 Triton Attention Backend (vllm-omni)

This plugin also provides an **out-of-tree diffusion attention backend** for [vllm-omni](https://github.com/vllm-project/vllm-omni), using the [SageAttention3](https://github.com/thu-ml/SageAttention) standalone Triton kernel.

### How It Works

Registers via the `vllm_omni.general_plugins` entry_point. When `VLLM_SAGE3_TRITON=1`, overrides the `SAGE_ATTN` diffusion attention backend with the sage3 Triton implementation. When disabled (default), the original in-tree backend is used unchanged.

- Zero vllm-omni source modifications
- Conditional activation — doesn't affect normal operation when off
- Falls back to torch SDPA for cross-attention (different Q/K sequence lengths)

### Usage

```bash
# Enable sage3 Triton attention for diffusion models
VLLM_SAGE3_TRITON=1 \
SAGE3_QUANT_FORMAT=mxfp4 \
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN \
python examples/offline_inference/text_to_image/text_to_image.py \
  --model /path/to/model ...

# Use original in-tree sage_attn (sageattention v2) — default
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN python ...

# Use torch SDPA (no sage at all)
DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA python ...
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_SAGE3_TRITON` | `0` | Set to `1` to enable sage3 Triton backend override |
| `SAGE3_QUANT_FORMAT` | `mxfp4` | Quantization config for K/V (`mxfp4`, `nvfp4`, `mxfp8_s1`, `mxfp4_s1`) |
| `SAGE3_ACC_DTYPE` | `fp32` | Accumulator dtype (`fp32`, `bf16_both_dot`, `bf16_pv_only`, etc.) |

### Notes

- **Shared memory requirement**: The sage3 fp32 kernel needs ~192KB shared memory per SM. On GPUs with less (e.g., RTX 6000D with 100KB), use `SAGE3_ACC_DTYPE=bf16_both_dot` or switch to TORCH_SDPA.
- **Cross-attention**: sage3 requires Q and K to have the same sequence length. Cross-attention calls automatically fall back to torch SDPA.
