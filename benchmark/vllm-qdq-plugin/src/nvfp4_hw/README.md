# Native NVFP4 Hardware Support

The `nvfp4_hw` package is the out-of-tree adapter for vLLM's native NVFP4
hardware kernels. It is intentionally separate from the activation QDQ
implementations in `vllm_qdq_plugin.quantization`.

## Scope

- AutoRound `data_type: nv_fp` checkpoints.
- Packed E2M1 weights and FP8 E4M3 per-group weight scales.
- Weight and input global scales.
- Dense linear and fused MoE layers.
- vLLM native NVFP4 linear and MoE kernels selected by its hardware oracle.

This package does not implement NVFP4_E5M3 and does not simulate activation
quantization. When `VLLM_QDQ=1`, its plugin entry point skips registration so
the QDQ package can select its Marlin-based standard NVFP4 implementation.

## Entry Point

```text
inc_nvfp4 = nvfp4_hw.patch:register
```

With QDQ disabled, loading an AutoRound standard NVFP4 checkpoint selects this
hardware path automatically:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> vllm serve /path/to/nvfp4-model \
  --dtype bfloat16 --trust-remote-code
```

## Source Layout

| File | Purpose |
| --- | --- |
| `patch.py` | Register AutoRound `nv_fp` metadata and native scheme routing. |
| `inc_nvfp4_scheme.py` | Select native dense or MoE implementations. |
| `inc_nvfp4_linear.py` | Load weights and invoke vLLM's native NVFP4 linear kernel. |
| `inc_nvfp4_moe.py` | Load experts and invoke the vLLM-selected native NVFP4 MoE backend. |
