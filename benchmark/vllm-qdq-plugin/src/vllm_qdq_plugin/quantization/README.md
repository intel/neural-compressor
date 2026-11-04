# AutoRound NVFP4 QDQ Support

This package implements activation quant-dequant simulation for AutoRound
standard NVFP4, NVFP4_E5M3, and NVFP4_E5M3_GS checkpoints. All paths use vLLM FP4 Marlin for
packed E2M1 weights and apply QDQ to activations before dense and MoE GEMMs.
They do not invoke the native NVFP4 hardware kernels in `nvfp4_hw`.

## Formats

| Format | AutoRound data type | Block scale | Group size |
| --- | --- | --- | ---: |
| Standard NVFP4 | `nv_fp` | FP8 E4M3 plus per-layer global scale | 16 |
| NVFP4_E5M3 | `nvfp4_v2` | Unsigned E5M3 | 16 |
| NVFP4_E5M3_GS | `nvfp4_v2_gs` | Unsigned E5M3 plus per-layer FP32 global scale | 16 |

Standard NVFP4 QDQ follows vLLM's reference semantics: each activation group
computes `FP8_E4M3(global_scale * max(abs(x)) / 6)`, rounds normalized values
to E2M1, and dequantizes using the block scale divided by the global scale.

NVFP4_E5M3 quantizes each activation block scale to unsigned E5M3 and applies
the same E2M1 value grid. It supports the CuTe backend and pure-Torch fallback.

NVFP4_E5M3_GS (packing format `auto_round:llm_compressor_nvfp4_e5m3_gs`) adds
AutoRound's per-layer FP32 global scales (`gs = 114688 * 6 / amax`):

- Weights dequantize as `E2M1 * UE5M3(scale) / weight_global_scale`. The global
  scale is folded into Marlin's global multiplier together with an exact
  power-of-two shift that keeps the block scales inside Marlin's scale range.
  Fused projections or gate/up shards with different global scales are rescaled to
  one shared multiplier, which may add a small rounding error.
- Activations compute `UE5M3(input_global_scale * max(abs(x)) / 6)` and
  dequantize with the block scale divided by the global scale. A dedicated CuTe
  kernel (group size 16) handles this, with a pure-Torch fallback.
- Model-free checkpoints store `input_global_scale = 1`; the plugin detects this
  at load time and reuses the global-scale-free NVFP4_E5M3 activation kernel.

## Usage

Install the plugin into the vLLM environment and set `VLLM_QDQ=1`:

```bash
VLLM_QDQ=1 CUDA_VISIBLE_DEVICES=<idle-gpu> \
  vllm serve /path/to/autoround-nvfp4-model \
  --dtype bfloat16 --trust-remote-code
```

The `inc_nvfp4_qdq` entry point registers both QDQ schemes. The separate
`nvfp4_hw` entry point observes `VLLM_QDQ=1` and does not register its hardware
scheme, preventing either implementation from taking over the other's model.

## Limitations

- Checkpoint-backed dense and MoE execution requires group size 16 and BF16 activations.
- Standard NVFP4 currently uses the correctness-first reference activation QDQ implementation.
- Marlin MoE requires one shared input global scale for each expert projection.
  For NVFP4_E5M3_GS, mismatched values fall back to the smallest one with a warning.
- MoE does not support `apply_router_weight_on_input` or EPLB.

## Source Layout

| File | Purpose |
| --- | --- |
| `patch.py` | Register standard NVFP4, NVFP4_E5M3, and NVFP4_E5M3_GS QDQ schemes. |
| `inc_nvfp4_{scheme,linear,moe}.py` | Standard NVFP4 QDQ plus FP4 Marlin execution. |
| `inc_nvfp4_e5m3_{scheme,linear,moe}.py` | NVFP4_E5M3(_GS) QDQ plus FP4 Marlin execution. |
| `fused_moe_e5m3.py` | Standalone Triton parity and benchmark implementation. |
| `../qdq/nvfp4.py` | Standard NVFP4 reference activation QDQ. |
| `../qdq/nvfp4_e5m3.py` | NVFP4_E5M3 backend selection and reference QDQ. |
| `../qdq/nvfp4_e5m3_cute.py` | Fake-aware CuTe custom operators. |
