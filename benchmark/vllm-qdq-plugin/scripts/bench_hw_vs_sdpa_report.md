# SAGE3 HW Kernel vs BF16 SDPA Benchmark Report

## Setup

| Parameter | Value |
|-----------|-------|
| Device | NVIDIA B200 |
| Shape | `[1, 40, 75600, 128]` (B, H, N, D) |
| dtype | bfloat16 |
| Warmup | 3 iterations |
| Measured | 10 iterations |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu130 |
| Total FLOPs/call | 116.96 TFLOP (4×B×H×N²×D) |

## Results

| Config | Latency (ms) | TFLOPS | Speedup vs SDPA |
|--------|-------------|--------|-----------------|
| **bf16_sdpa** (baseline) | 84.56 | 1384.2 | 1.00× |
| mxfp8_hw | 258.42 | 452.9 | 0.33× |
| mxfp4_hw | 261.81 | 447.1 | 0.32× |
| ~~mixed_mxfp8qk_mxfp4pv_hw~~ | 204.49 | 572.4 | 0.41× |
| ~~mixed_mxfp4qk_mxfp8pv_hw~~ | 210.28 | 556.6 | 0.40× |

## Analysis

All SAGE3 hardware kernels are **slower than cuDNN BF16 SDPA** on this shape:

- **BF16 SDPA** achieves ~1384 TFLOPS — close to B200's BF16 Tensor Core peak, thanks to highly optimized cuDNN flash attention.
- **MXFP8/MXFP4 HW kernels** achieve only 450–570 TFLOPS (0.32–0.41× of SDPA). The overhead likely comes from:
  1. **Host-side quantization** (`quantize_to_mxfp4`/`quantize_to_mxfp8`) adds significant latency before the kernel even launches
  2. **Triton vs cuDNN**: cuDNN's flash attention is heavily tuned for B200; the Triton `tl.dot_scaled` path is still maturing
  3. **Delta_s correction** (per-block-mean smoothing) adds extra compute in the mxfp8/mxfp4 paths
- **Mixed precision** (mxfp8 QK + mxfp4 PV, or vice versa) is ~20% faster than pure mxfp4/mxfp8 — possibly because one axis uses fewer quantization/dequantization steps

## Conclusion

At this long-sequence shape (N=75600), the Triton-based HW quantized kernels cannot yet compete with cuDNN BF16 flash attention on B200. The quantization overhead and less-optimized Triton codegen dominate. These kernels may show relative improvement on memory-bound regimes (smaller batch, GQA, or inference with KV-cache) where reduced memory bandwidth from lower-precision formats matters more than raw compute throughput.
