"""Shared Triton JIT helpers for MXFP hardware attention kernels."""

import triton
import triton.language as tl


@triton.jit
def compute_p_scale_inv(
    p_amax,
    FORMAT_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute E8M0 scale and inverse scale for P quantization.

    Uses multiply-by-inverse pattern to avoid NaN from GPU flush-to-zero (FTZ).
    When p_amax=0 (common for softmax groups far from peak):
      - log2(0) → -inf → clamp to -127
      - inv_scale = 2^127 (large normal float32)
      - p * inv_scale = 0 * 2^127 = 0 (safe)

    The old division pattern would compute scale=2^-127 (subnormal → FTZ → 0),
    then p / 0 = NaN.

    Args:
        p_amax: [BLOCK_M, BLOCK_N // 32] per-group max of P
        FORMAT_MAX: max representable value (448.0 for E4M3, 6.0 for E2M1)
        BLOCK_M: tile height
        BLOCK_N: tile width (reduction dim)

    Returns:
        p_e8m0: [BLOCK_M, BLOCK_N // 32] uint8 E8M0 scale encodings
        inv_scale_expanded: [BLOCK_M, BLOCK_N] float32 inverse scales for multiply
    """
    p_amax_normalized = p_amax / FORMAT_MAX
    p_log2 = tl.math.ceil(tl.math.log2(p_amax_normalized))
    p_log2 = tl.minimum(tl.maximum(p_log2, -127.0), 127.0)
    p_e8m0 = (p_log2 + 127.0).to(tl.uint8)

    inv_scale = tl.math.exp2(-p_log2)
    inv_scale_expanded = tl.reshape(
        tl.broadcast_to(inv_scale[:, :, None], [BLOCK_M, BLOCK_N // 32, 32]),
        [BLOCK_M, BLOCK_N],
    )

    return p_e8m0, inv_scale_expanded
