"""
Quantization primitives: round/quant functions (torch + triton pairs) and block-processing helpers.

This module contains the core numerical operations used by both host-side quantization
(quantize.py) and kernel-side P-quantization (p_quant_kernels.py).
"""

import torch
import triton
import triton.language as tl

# ── Constants ──

FP4_MAX = 6.0
FP8_MAX = 448.0
MICROSCALE_BLOCK_SIZE_16 = 16   # NVFP4 block size
MICROSCALE_BLOCK_SIZE_32 = 32   # MXFP4/MXFP8 block size
COMBINED_MAX = FP8_MAX * FP4_MAX  # 2688
E8M0_MIN = 5.877471754111686e-39   # 2**-127, smallest positive E8M0 value

# NVFP4 E2M1 representable values: ±{0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6}
NVFP4_E2M1_VALUES = [-6, -4, -3, -2, -1.5, -1, -0.75, -0.5, 0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6]

# ============================================================================
# E2M1 (FP4) quantization
# ============================================================================

@triton.jit
def apply_e2m1_quantization_triton(x):
    """
    Apply E2M1 quantization in Triton.

    Representable values: ±{0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6}
    """
    x_abs = tl.abs(x)
    sign = tl.where(x >= 0.0, 1.0, -1.0)

    # Find nearest FP4 E2M1 level
    quantized_abs = tl.where(x_abs < 0.25, 0.0,
                    tl.where(x_abs < 0.625, 0.5,    # (0.5 + 0.75) / 2 = 0.625
                    tl.where(x_abs < 0.875, 0.75,   # (0.75 + 1.0) / 2 = 0.875
                    tl.where(x_abs < 1.25, 1.0,     # (1.0 + 1.5) / 2 = 1.25
                    tl.where(x_abs < 1.75, 1.5,     # (1.5 + 2.0) / 2 = 1.75
                    tl.where(x_abs < 2.5, 2.0,      # (2.0 + 3.0) / 2 = 2.5
                    tl.where(x_abs < 3.5, 3.0,      # (3.0 + 4.0) / 2 = 3.5
                    tl.where(x_abs < 5.0, 4.0,      # (4.0 + 6.0) / 2 = 5.0
                                          6.0))))))))

    return quantized_abs * sign


def apply_e2m1_quantization_torch(x):
    """
    Apply E2M1 quantization in PyTorch.

    Representable values: ±{0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6}
    """
    x_abs = torch.abs(x)
    sign = torch.where(x >= 0.0, 1.0, -1.0)

    quantized_abs = torch.where(x_abs < 0.25, 0.0,
                    torch.where(x_abs < 0.625, 0.5,
                    torch.where(x_abs < 0.875, 0.75,
                    torch.where(x_abs < 1.25, 1.0,
                    torch.where(x_abs < 1.75, 1.5,
                    torch.where(x_abs < 2.5, 2.0,
                    torch.where(x_abs < 3.5, 3.0,
                    torch.where(x_abs < 5.0, 4.0,
                                  6.0))))))))

    return quantized_abs * sign


# ============================================================================
# E4M3 (FP8) quantization
# ============================================================================

@triton.jit
def apply_e4m3_quantization_triton(x):
    """
    Apply E4M3 quantization in Triton via FP32 -> FP8E4M3 -> FP32 round-trip.

    FP8 E4M3 representable range: [-448, 448] with 3-bit mantissa precision.
    """
    x_type = x.dtype
    return x.to(tl.float8e4nv).to(x_type)


def apply_e4m3_quantization_torch(x):
    """
    Apply E4M3 quantization in PyTorch via FP32 -> FP8E4M3 -> FP32 round-trip.

    FP8 E4M3 representable range: [-448, 448] with 3-bit mantissa precision.
    """
    return x.to(torch.float8_e4m3fn).to(x.dtype)


# ============================================================================
# Scale rounding functions
# ============================================================================

@triton.jit
def round_to_e4m3_triton(scale):
    """
    Round scale to E4M3 precision via FP32 -> E4M3 -> FP32 cast round-trip.

    This truncates the mantissa to 3 bits, matching the real CUDA kernel's behavior:
        reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
        SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
    """
    scale_type = scale.dtype
    return scale.to(tl.float8e4nv).to(scale_type)


def round_to_e4m3_torch(scales):
    """
    Round scales to E4M3 precision via FP32 -> E4M3 -> FP32 cast round-trip.

    This truncates the mantissa to 3 bits, matching the real CUDA kernel's behavior.
    """
    return scales.to(torch.float8_e4m3fn).to(scales.dtype)


@triton.jit
def round_to_e8m0_triton(scale):
    """
    Round scale to E8M0 precision (power-of-2 only).

    E8M0: 8 exponent bits, 0 mantissa bits. Represents values as 2^(e - 127).
    Uses ceil per OCP MX spec.
    """
    scale_type = scale.dtype
    abs_scale = tl.abs(scale)
    log2_scale = tl.log2(tl.maximum(abs_scale, 5.877471754111686e-39))  # E8M0_MIN = 2**-127
    rounded = tl.ceil(log2_scale)
    rounded = tl.maximum(tl.minimum(rounded, 127.0), -127.0)
    return tl.exp2(rounded).to(scale_type)


def round_to_e8m0_torch(scales):
    """
    Round scales to E8M0 precision (power-of-2 only).

    E8M0: 8 exponent bits, 0 mantissa bits. Represents values as 2^(e - 127).
    Uses ceil per OCP MX spec.
    """
    abs_scales = scales.abs()
    log2_scales = torch.log2(abs_scales.clamp(min=2**-127))
    rounded_log2 = torch.ceil(log2_scales)
    rounded_log2 = torch.clamp(rounded_log2, min=-127, max=127)
    return torch.exp2(rounded_log2)


# ============================================================================
# Block-processing helpers for P-quantization kernels
# ============================================================================
# These deduplicate the block masking / microscaling boilerplate that was
# copy-pasted 4-8 times in each P-quant kernel.

@triton.jit
def compute_block_max(tile, col_indices, block_start: tl.constexpr, block_end: tl.constexpr):
    """
    Compute per-row max of abs values for a single column block.

    Args:
        tile: [BLOCK_M, BLOCK_N] tensor
        col_indices: [BLOCK_N] arange
        block_start: start column (constexpr)
        block_end: end column (constexpr)

    Returns:
        [BLOCK_M] vector of per-row maxes within the block
    """
    mask = (col_indices >= block_start) & (col_indices < block_end)
    masked = tl.where(mask[None, :], tl.abs(tile), 0.0)
    return tl.max(masked, axis=1)


@triton.jit
def build_microscale_tensor_4(s0, s1, s2, s3, block_ids):
    """
    Build [M, N] microscale tensor from 4 per-row scale vectors.

    Used by MXFP4/MXFP8 schemes with block_size=32 (128/32 = 4 blocks).

    Args:
        s0..s3: [BLOCK_M] per-row scale vectors for blocks 0-3
        block_ids: [BLOCK_N] block ID for each column

    Returns:
        [BLOCK_M, BLOCK_N] microscale tensor
    """
    return (
        tl.where(block_ids[None, :] == 0, s0[:, None],
        tl.where(block_ids[None, :] == 1, s1[:, None],
        tl.where(block_ids[None, :] == 2, s2[:, None],
                                           s3[:, None])))
    )


@triton.jit
def build_microscale_tensor_8(s0, s1, s2, s3, s4, s5, s6, s7, block_ids):
    """
    Build [M, N] microscale tensor from 8 per-row scale vectors.

    Used by NVFP4 scheme with block_size=16 (128/16 = 8 blocks).

    Args:
        s0..s7: [BLOCK_M] per-row scale vectors for blocks 0-7
        block_ids: [BLOCK_N] block ID for each column

    Returns:
        [BLOCK_M, BLOCK_N] microscale tensor
    """
    return (
        tl.where(block_ids[None, :] == 0, s0[:, None],
        tl.where(block_ids[None, :] == 1, s1[:, None],
        tl.where(block_ids[None, :] == 2, s2[:, None],
        tl.where(block_ids[None, :] == 3, s3[:, None],
        tl.where(block_ids[None, :] == 4, s4[:, None],
        tl.where(block_ids[None, :] == 5, s5[:, None],
        tl.where(block_ids[None, :] == 6, s6[:, None],
                                           s7[:, None])))))))
    )
