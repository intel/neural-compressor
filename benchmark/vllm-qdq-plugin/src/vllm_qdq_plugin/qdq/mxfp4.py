# SPDX-License-Identifier: Apache-2.0
"""
MXFP4 input activation quant-dequant (QDQ).

This implementation is adapted from vLLM's MXFP4 reference test helper
(`tests/quantization/reference_mxfp4.py`) but kept self-contained so the
plugin does not depend on vLLM internals.
"""

import torch

BFLOAT16_EXP_BIAS = 127
BFLOAT16_MANTISSA_BITS = 7
BFLOAT16_EXP_BITS = 8

FLOAT16_EXP_BIAS = 15
FLOAT16_MANTISSA_BITS = 10
FLOAT16_EXP_BITS = 5

FLOAT8_E8M0_MAX_EXP = 127
FLOAT4_EXP_BIAS = 1
FLOAT4_MANTISSA_BITS = 1

FLOAT16_VAL_TO_ADD = 1 << (FLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1)
FLOAT16_SIGN_EXPONENT_MASK = (
    (1 << (FLOAT16_EXP_BITS + 1)) - 1
) << FLOAT16_MANTISSA_BITS

BFLOAT16_VAL_TO_ADD = 1 << (BFLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1)
BFLOAT16_SIGN_EXPONENT_MASK = (
    (1 << (BFLOAT16_EXP_BITS + 1)) - 1
) << BFLOAT16_MANTISSA_BITS


def _fp_to_fp4_simulate(
    val: torch.Tensor,
    *,
    half_mantissa_bits: int,
    half_exp_bits: int,
    half_exp_bias: int,
) -> torch.Tensor:
    """Simulate casting fp16/bf16 input to float4_e2m1 values."""
    float_type = val.dtype
    val_view = val.view(torch.int16)

    exp = val_view >> half_mantissa_bits
    exp = exp & ((1 << half_exp_bits) - 1)
    exp = exp.view(torch.uint16).to(torch.int32)

    sign = (val_view >> (half_mantissa_bits + half_exp_bits)) & 1
    mantissa_last = (val_view >> (half_mantissa_bits - 1)) & 1

    exp_unbias = exp - half_exp_bias
    new_exp = exp_unbias + FLOAT4_EXP_BIAS
    exp_shift = (new_exp <= 0) * (1 - new_exp)

    tail_bits = half_mantissa_bits - FLOAT4_MANTISSA_BITS + exp_shift
    tail_bits[tail_bits >= 16] = 16

    mantissa_plus_one = val_view & ((1 << (half_mantissa_bits + 1)) - 1)
    half = 1 << (tail_bits - 1)
    tail = mantissa_plus_one & ((1 << tail_bits) - 1)

    round_close = tail < half
    round_away = tail > half
    tie = tail == half

    new_mantissa_close = (new_exp > 0) * mantissa_last
    new_exp_close = exp

    new_mantissa_away = torch.logical_and(new_exp > 0, mantissa_last == 0)
    new_exp_away = exp + torch.logical_or(new_exp <= 0, mantissa_last == 1)

    new_exp_tie = (exp > (half_exp_bias - 2)) * (exp + (mantissa_last == 1))

    new_exp = (
        round_away * new_exp_away + round_close * new_exp_close + tie * new_exp_tie
    )
    new_mantissa = round_away * new_mantissa_away + round_close * new_mantissa_close
    new_mantissa = new_mantissa + (new_exp > (2 + half_exp_bias)) * (new_mantissa == 0)

    new_exp = (new_exp >= (half_exp_bias - 2)) * torch.clamp(
        new_exp, half_exp_bias - 2, half_exp_bias + 2
    )

    sign = sign.to(torch.int32)
    new_mantissa = new_mantissa.to(torch.int32)

    qdq_val = (
        (sign << 15)
        + (new_exp << half_mantissa_bits)
        + (new_mantissa << (half_mantissa_bits - 1))
    )
    qdq_val = qdq_val.to(torch.uint16)
    return qdq_val.view(float_type)


def mxfp4_qdq(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Quantize-dequantize input to MXFP4 (E2M1 + E8M0 scales).

    Args:
        x: 2D tensor [M, K] in bf16/fp16
        group_size: number of elements per scale group (default 32)

    Returns:
        Tensor same shape and dtype as x, with MXFP4 quantization noise applied.
    """
    orig_dtype = x.dtype
    assert x.dim() == 2, (
        f"mxfp4_qdq only supports 2D tensors for now, but got {x.dim()}D"
    )
    assert orig_dtype in (torch.float16, torch.bfloat16), (
        f"mxfp4_qdq only supports fp16/bf16 tensors, but got {orig_dtype}"
    )
    m, k = x.shape

    # Pad k to multiple of group_size
    pad = (group_size - k % group_size) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))

    if orig_dtype == torch.float16:
        half_mantissa_bits = FLOAT16_MANTISSA_BITS
        half_exp_bits = FLOAT16_EXP_BITS
        half_exp_bias = FLOAT16_EXP_BIAS
        val_to_add = FLOAT16_VAL_TO_ADD
        sign_exponent_mask = FLOAT16_SIGN_EXPONENT_MASK
    else:
        half_mantissa_bits = BFLOAT16_MANTISSA_BITS
        half_exp_bits = BFLOAT16_EXP_BITS
        half_exp_bias = BFLOAT16_EXP_BIAS
        val_to_add = BFLOAT16_VAL_TO_ADD
        sign_exponent_mask = BFLOAT16_SIGN_EXPONENT_MASK

    x = x.reshape(m, -1, group_size)

    block_max = torch.max(torch.abs(x), dim=-1).values
    block_max = block_max.view(torch.uint16).to(torch.int32)
    block_max_uint = torch.bitwise_and(block_max + val_to_add, sign_exponent_mask)
    block_max = block_max_uint.to(torch.uint16).view(orig_dtype)

    scale_exp = (
        FLOAT8_E8M0_MAX_EXP + torch.floor(torch.log2(block_max)).to(torch.int32) - 2
    )
    scale_exp = torch.clamp(scale_exp, 0, 2 * FLOAT8_E8M0_MAX_EXP)
    scale = (2.0 ** (scale_exp - FLOAT8_E8M0_MAX_EXP)).to(orig_dtype)

    x = x / scale[..., None]
    x_fp4 = _fp_to_fp4_simulate(
        x,
        half_exp_bits=half_exp_bits,
        half_mantissa_bits=half_mantissa_bits,
        half_exp_bias=half_exp_bias,
    )
    x_fp4 = x_fp4 * scale[..., None]
    return x_fp4.reshape(m, -1)[:, :k].to(orig_dtype)
