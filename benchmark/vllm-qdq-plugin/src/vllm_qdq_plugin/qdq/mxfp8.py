# SPDX-License-Identifier: Apache-2.0
"""
MXFP8 input activation quant-dequant (QDQ).

Implements the MX specification for MXFP8: E4M3 data format (float8_e4m3fn,
bias=7, max=448) with per-group E8M0 (power-of-2) scales of group_size elements.
"""

import torch

FLOAT8_E8M0_MAX_EXP = 127

# float8_e4m3fn (OCP standard) parameters:
#   max finite value  = 1.110 × 2^8 = 448  (biased exp 1111, mantissa 110)
#   11111111          = NaN; bias = 7
# Values > 448 map to NaN (not clamped) so we must clamp explicitly before cast.
# The E8M0 scale offset uses floor(log2(448)) = 8: with block_max ∈ [2^n, 2^(n+1))
# and scale = 2^(n−8), x/scale can reach up to 512, so clamp to ±448 is required.
FLOAT8_E4M3_MAX_UNBIASED_EXP = 8
FLOAT8_E4M3FN_MAX = 448.0
FLOAT8_E4M3_MIN_NORMAL = 2.0**-6
FLOAT8_E4M3_SUBNORMAL_STEP = 2.0**-9


def _round_half_to_even(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest integer with round-half-to-even (banker's rounding).

    Explicitly handles ties by rounding to the nearest even integer, matching
    the tie-breaking behavior of the MXFP4 simulator.
    """
    floor_x = torch.floor(x)
    frac = x - floor_x
    tie = frac == 0.5
    round_up = frac > 0.5
    # On a tie, round up only when floor_x is odd (so floor_x+1 is even).
    round_up_on_tie = (floor_x % 2.0) != 0.0
    return floor_x + (round_up | (tie & round_up_on_tie)).to(floor_x.dtype)


def _quantize_to_e4m3fn_no_fp8_dtype(x: torch.Tensor) -> torch.Tensor:
    """Quantize fp32 tensor to E4M3FN value grid without using fp8 dtype casts.

    This avoids Triton/Inductor architecture constraints on fp8_e4m3 kernels.
    """
    x = x.to(torch.float32)
    sign = torch.sign(x)
    ax = torch.abs(x)

    # Finite E4M3FN range.
    ax = torch.clamp(ax, 0.0, FLOAT8_E4M3FN_MAX)

    # Subnormal region: values are multiples of 2^-9 in [0, 7*2^-9].
    sub_q = (
        _round_half_to_even(ax / FLOAT8_E4M3_SUBNORMAL_STEP)
        * FLOAT8_E4M3_SUBNORMAL_STEP
    )
    sub_q = torch.clamp(sub_q, 0.0, 7.0 * FLOAT8_E4M3_SUBNORMAL_STEP)

    # Normal region: step is 2^(e-3), where e=floor(log2(|x|)) in [-6, 8].
    safe_ax = torch.clamp(ax, min=FLOAT8_E4M3_MIN_NORMAL)
    e = torch.floor(torch.log2(safe_ax))
    e = torch.clamp(e, -6.0, 8.0)
    step = torch.exp2(e - 3.0)
    norm_q = _round_half_to_even(ax / step) * step

    # Rounding can cross exponent boundary; clamp to finite max.
    norm_q = torch.clamp(norm_q, FLOAT8_E4M3_MIN_NORMAL, FLOAT8_E4M3FN_MAX)

    q = torch.where(ax < FLOAT8_E4M3_MIN_NORMAL, sub_q, norm_q)
    return q * sign


def mxfp8_qdq(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Quantize-dequantize input to MXFP8 (E4M3 + E8M0 scales).

    Simulates MX MXFP8-E4M3 quantization:
      1. Compute a per-group E8M0 (power-of-2) scale from the group max.
      2. Divide by scale, then quantize to the float8_e4m3fn value grid using
         the dtype-free E4M3FN simulator.
      3. Multiply back by scale.

    Args:
        x: 2D tensor [M, K] in bf16/fp16
        group_size: number of elements per scale group (default 32)

    Returns:
        Tensor same shape and dtype as x, with MXFP8 E4M3 quantization noise applied.
    """
    orig_dtype = x.dtype
    assert x.dim() == 2, (
        f"mxfp8_qdq only supports 2D tensors for now, but got {x.dim()}D"
    )
    assert orig_dtype in (torch.float16, torch.bfloat16), (
        f"mxfp8_qdq only supports fp16/bf16 tensors, but got {orig_dtype}"
    )
    m, k = x.shape

    # Pad k to multiple of group_size
    pad = (group_size - k % group_size) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))

    x = x.reshape(m, -1, group_size)

    # --- E8M0 scale computation (done in fp32 to avoid fp16 overflow / underflow) ---
    # E8M0 represents only powers of 2; we choose the smallest power-of-2 scale
    # such that max_element / scale fits within float8_e4m3fn's representable range.
    block_max_f32 = torch.max(torch.abs(x), dim=-1).values.to(torch.float32)
    block_max_f32 = block_max_f32.clamp(min=torch.finfo(torch.float32).tiny)

    scale_exp = (
        FLOAT8_E8M0_MAX_EXP
        + torch.floor(torch.log2(block_max_f32)).to(torch.int32)
        - FLOAT8_E4M3_MAX_UNBIASED_EXP
    )
    scale_exp = torch.clamp(scale_exp, 0, 2 * FLOAT8_E8M0_MAX_EXP)
    scale = 2.0 ** (scale_exp - FLOAT8_E8M0_MAX_EXP)

    # --- Quantize to E4M3 grid, then dequantize ---
    # Use a dtype-free simulator so torch.compile/inductor never generates Triton
    # fp8e4 kernels (which are unsupported on some architectures).
    x_scaled = x.to(torch.float32) / scale[..., None]
    x_fp8 = _quantize_to_e4m3fn_no_fp8_dtype(x_scaled)
    x_fp8 = (x_fp8 * scale[..., None]).to(orig_dtype)
    return x_fp8.reshape(m, -1)[:, :k]
