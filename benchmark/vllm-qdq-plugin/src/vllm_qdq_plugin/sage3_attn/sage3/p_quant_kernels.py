"""
P-quantization Triton kernels for in-kernel attention probability quantization.

Each kernel self-registers via @register_p_quant and composes shared primitives
from quant_primitives.py, reducing the per-scheme boilerplate to the unique logic:
scale rounding method, data quantization, and single-vs-two-level factorization.

All functions have the signature: (p_tile, BLOCK_N: tl.constexpr) -> p_quantized

IMPORTANT: All P-quant kernels assume BLOCK_N=128. Block boundaries are hardcoded
(e.g., 0-16, 16-32, ..., 112-128 for block_size=16; 0-32, 32-64, ..., 96-128 for
block_size=32). The kernel launcher enforces this constraint via assertion.

NOTE: The epsilon value 1e-8 is used inline throughout these kernels rather than
as a module-level constant. Triton @jit functions cannot reference Python module-level
variables — they require literals or tl.constexpr parameters.
"""

import triton.language as tl

from .p_quant_registry import register_p_quant
from .quant_primitives import (
    apply_e2m1_quantization_triton,
    apply_e4m3_quantization_triton,
    round_to_e4m3_triton,
    round_to_e8m0_triton,
    compute_block_max,
    build_microscale_tensor_4,
    build_microscale_tensor_8,
)


# ============================================================================
# NVFP4: Two-level, E4M3 scales, block_size=16
# ============================================================================

@register_p_quant("nvfp4")
def p_quant_nvfp4(p_tile, BLOCK_N: tl.constexpr):
    """
    Two-level P quantization with fixed COMBINED_MAX + per-16-col-block microscaling.

    Matches the CUTE kernel's fused softmax behavior exactly:
    CUTE computes: P = exp(score*scale - max_scaled) * COMBINED_MAX, where COMBINED_MAX
    = 448 * 6 = 2688 is constant across all tiles. The returned p_quantized carries
    COMBINED_MAX (values in [0, ~2688]). The attention kernel's final normalization
    (output_tile / running_sum) cancels it, since both numerator and denominator
    carry the same factor — identical to CUTE.

    NOTE: This intentionally diverges from the original monolith
    (sageattention3_standalone.py) which uses per-tile adaptive global scaling.
    The monolith computes global_scales = max(abs(p_tile)) / COMBINED_MAX per tile,
    which gives non-peak tiles a smaller scale — different quantization behavior
    from the CUTE kernel.

    Level 1: Fixed COMBINED_MAX scaling (constant, not data-dependent)
    Level 2: Per-row, per-16-col-block microscale with E4M3 rounding

    For 128x128 tiles: 8 blocks of 16 columns each.
    """
    FP4_MAX: tl.constexpr = 6.0
    FP8_MAX: tl.constexpr = 448.0
    MICROSCALE_BLOCK_SIZE: tl.constexpr = 16
    COMBINED_MAX = FP8_MAX * FP4_MAX  # 2688

    # Level 1: Fixed COMBINED_MAX scaling (matches CUTE kernel's fused softmax)
    # CUTE: P = exp(score*scale - max_scaled) * 2688, constant across all tiles.
    # We cancel the factor inside this function to keep output in [0,1] range.
    p_level1 = p_tile * COMBINED_MAX

    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    # Level 2: Per-row, per-block microscaling (8 blocks of 16)
    b0 = compute_block_max(p_level1, col_indices, 0, 16)
    b1 = compute_block_max(p_level1, col_indices, 16, 32)
    b2 = compute_block_max(p_level1, col_indices, 32, 48)
    b3 = compute_block_max(p_level1, col_indices, 48, 64)
    b4 = compute_block_max(p_level1, col_indices, 64, 80)
    b5 = compute_block_max(p_level1, col_indices, 80, 96)
    b6 = compute_block_max(p_level1, col_indices, 96, 112)
    b7 = compute_block_max(p_level1, col_indices, 112, 128)

    # E4M3-rounded microscales
    s0 = round_to_e4m3_triton(tl.maximum(b0 / FP4_MAX, 1e-8))
    s1 = round_to_e4m3_triton(tl.maximum(b1 / FP4_MAX, 1e-8))
    s2 = round_to_e4m3_triton(tl.maximum(b2 / FP4_MAX, 1e-8))
    s3 = round_to_e4m3_triton(tl.maximum(b3 / FP4_MAX, 1e-8))
    s4 = round_to_e4m3_triton(tl.maximum(b4 / FP4_MAX, 1e-8))
    s5 = round_to_e4m3_triton(tl.maximum(b5 / FP4_MAX, 1e-8))
    s6 = round_to_e4m3_triton(tl.maximum(b6 / FP4_MAX, 1e-8))
    s7 = round_to_e4m3_triton(tl.maximum(b7 / FP4_MAX, 1e-8))

    microscale_final = build_microscale_tensor_8(s0, s1, s2, s3, s4, s5, s6, s7, block_ids)

    # Quantize and reconstruct — COMBINED_MAX stays baked in (matches CUTE kernel).
    # Cancels at attention kernel's final output_tile / running_sum.
    p_microscaled = p_level1 / microscale_final
    p_quantized = apply_e2m1_quantization_triton(p_microscaled)
    return p_quantized * microscale_final


# ============================================================================
# MXFP4: Two-level, E8M0 scales, block_size=32
# ============================================================================

@register_p_quant("mxfp4")
def p_quant_mxfp4(p_tile, BLOCK_N: tl.constexpr):
    """
    Two-level P quantization with MXFP4 microscaling (block_size=32, E8M0 scales).

    Uses the same fixed COMBINED_MAX pattern as nvfp4 (see p_quant_nvfp4 docstring).
    This is an INDEPENDENT DESIGN CHOICE — no CUTE MXFP4 kernel exists in this repo
    (CUTE uses E4M3 scales and NVFP4 instruction path, not E8M0/MXFP4). The fixed
    COMBINED_MAX is applied here as a principled decision for consistency, not as
    a CUTE alignment claim.

    Level 1: Fixed COMBINED_MAX scaling (constant, not data-dependent)
    Level 2: Per-row, per-32-col-block microscale with E8M0 rounding

    For 128x128 tiles: 4 blocks of 32 columns each.
    """
    FP4_MAX: tl.constexpr = 6.0
    FP8_MAX: tl.constexpr = 448.0
    MICROSCALE_BLOCK_SIZE: tl.constexpr = 32
    COMBINED_MAX = FP8_MAX * FP4_MAX  # 2688

    # Level 1: Fixed COMBINED_MAX scaling (same pattern as nvfp4)
    p_level1 = p_tile * COMBINED_MAX

    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    # Level 2: Per-row, per-block microscaling (4 blocks of 32)
    b0 = compute_block_max(p_level1, col_indices, 0, 32)
    b1 = compute_block_max(p_level1, col_indices, 32, 64)
    b2 = compute_block_max(p_level1, col_indices, 64, 96)
    b3 = compute_block_max(p_level1, col_indices, 96, 128)

    # E8M0-rounded microscales
    s0 = round_to_e8m0_triton(tl.maximum(b0 / FP4_MAX, 1e-8))
    s1 = round_to_e8m0_triton(tl.maximum(b1 / FP4_MAX, 1e-8))
    s2 = round_to_e8m0_triton(tl.maximum(b2 / FP4_MAX, 1e-8))
    s3 = round_to_e8m0_triton(tl.maximum(b3 / FP4_MAX, 1e-8))

    microscale_final = build_microscale_tensor_4(s0, s1, s2, s3, block_ids)

    # Quantize and reconstruct, canceling COMBINED_MAX on output
    p_microscaled = p_level1 / microscale_final
    p_quantized = apply_e2m1_quantization_triton(p_microscaled)
    return p_quantized * microscale_final / COMBINED_MAX


# ============================================================================
# MXFP4_S1: Single-level, E8M0 scales, block_size=32
# ============================================================================

@register_p_quant("mxfp4_s1")
def p_quant_mxfp4_s1(p_tile, BLOCK_N: tl.constexpr):
    """
    Single-level P quantization with per-block E8M0 ceil microscaling (MXFP4).

    No global FP32 per-row scale — only per-block E8M0 microscaling.
    This avoids the information loss from two-level global→local factorization.

    For 128x128 tiles: 4 blocks of 32 columns each.
    """
    FP4_MAX: tl.constexpr = 6.0
    MICROSCALE_BLOCK_SIZE: tl.constexpr = 32

    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    # Single-level: per-block microscaling directly on p_tile
    b0 = compute_block_max(p_tile, col_indices, 0, 32)
    b1 = compute_block_max(p_tile, col_indices, 32, 64)
    b2 = compute_block_max(p_tile, col_indices, 64, 96)
    b3 = compute_block_max(p_tile, col_indices, 96, 128)

    # E8M0-rounded microscales
    s0 = round_to_e8m0_triton(tl.maximum(b0 / FP4_MAX, 1e-8))
    s1 = round_to_e8m0_triton(tl.maximum(b1 / FP4_MAX, 1e-8))
    s2 = round_to_e8m0_triton(tl.maximum(b2 / FP4_MAX, 1e-8))
    s3 = round_to_e8m0_triton(tl.maximum(b3 / FP4_MAX, 1e-8))

    microscale_final = build_microscale_tensor_4(s0, s1, s2, s3, block_ids)

    # Quantize and reconstruct (single level — no global_scales)
    p_microscaled = p_tile / microscale_final
    p_quantized = apply_e2m1_quantization_triton(p_microscaled)
    return p_quantized * microscale_final


# ============================================================================
# MXFP4_S1_E4M3: Single-level, E4M3 scales, E2M1 data, block_size=32
# ============================================================================

@register_p_quant("mxfp4_s1_e4m3")
def p_quant_mxfp4_s1_e4m3(p_tile, BLOCK_N: tl.constexpr):
    """
    Single-level P quantization with E4M3 scales (not E8M0) and block_size=32.

    Like mxfp4_s1 but with finer-grained E4M3 microscales instead of E8M0
    power-of-2 ceil. This isolates the scale format contribution to P-quant error.

    For 128x128 tiles: 4 blocks of 32 columns each.
    """
    FP4_MAX: tl.constexpr = 6.0
    MICROSCALE_BLOCK_SIZE: tl.constexpr = 32

    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    b0 = compute_block_max(p_tile, col_indices, 0, 32)
    b1 = compute_block_max(p_tile, col_indices, 32, 64)
    b2 = compute_block_max(p_tile, col_indices, 64, 96)
    b3 = compute_block_max(p_tile, col_indices, 96, 128)

    # E4M3-rounded microscales (NOT E8M0)
    # Clamp after rounding: E4M3 min subnormal is 2^-9 ≈ 0.00195
    E4M3_MIN_SUBNORMAL = 1.953125e-3
    s0 = tl.maximum(round_to_e4m3_triton(tl.maximum(b0 / FP4_MAX, 1e-8)), E4M3_MIN_SUBNORMAL)
    s1 = tl.maximum(round_to_e4m3_triton(tl.maximum(b1 / FP4_MAX, 1e-8)), E4M3_MIN_SUBNORMAL)
    s2 = tl.maximum(round_to_e4m3_triton(tl.maximum(b2 / FP4_MAX, 1e-8)), E4M3_MIN_SUBNORMAL)
    s3 = tl.maximum(round_to_e4m3_triton(tl.maximum(b3 / FP4_MAX, 1e-8)), E4M3_MIN_SUBNORMAL)

    microscale_final = build_microscale_tensor_4(s0, s1, s2, s3, block_ids)

    p_microscaled = p_tile / microscale_final
    p_quantized = apply_e2m1_quantization_triton(p_microscaled)
    return p_quantized * microscale_final


# ============================================================================
# MXFP8_S1: Single-level, E8M0 scales, FP8 data, block_size=32
# ============================================================================

@register_p_quant("mxfp8_s1")
def p_quant_mxfp8_s1(p_tile, BLOCK_N: tl.constexpr):
    """
    Single-level P quantization with per-block E8M0 ceil microscaling (MXFP8).

    Like MXFP4_S1 but uses E4M3 data quantization (fp_max=448) instead of
    E2M1 (fp_max=6). No global FP32 per-row scale.

    For 128x128 tiles: 4 blocks of 32 columns each.
    """
    FP8_MAX_VAL: tl.constexpr = 448.0
    MICROSCALE_BLOCK_SIZE: tl.constexpr = 32

    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    # Single-level: per-block microscaling directly on p_tile
    b0 = compute_block_max(p_tile, col_indices, 0, 32)
    b1 = compute_block_max(p_tile, col_indices, 32, 64)
    b2 = compute_block_max(p_tile, col_indices, 64, 96)
    b3 = compute_block_max(p_tile, col_indices, 96, 128)

    # E8M0-rounded microscales (using FP8 max instead of FP4 max)
    s0 = round_to_e8m0_triton(tl.maximum(b0 / FP8_MAX_VAL, 1e-8))
    s1 = round_to_e8m0_triton(tl.maximum(b1 / FP8_MAX_VAL, 1e-8))
    s2 = round_to_e8m0_triton(tl.maximum(b2 / FP8_MAX_VAL, 1e-8))
    s3 = round_to_e8m0_triton(tl.maximum(b3 / FP8_MAX_VAL, 1e-8))

    microscale_final = build_microscale_tensor_4(s0, s1, s2, s3, block_ids)

    # E4M3 data quantization (not E2M1) and reconstruct
    p_microscaled = p_tile / microscale_final
    p_quantized = apply_e4m3_quantization_triton(p_microscaled)
    return p_quantized * microscale_final
