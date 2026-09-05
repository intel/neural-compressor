"""NVFP4 E2M1 with UE5M3 scale activation quant-dequant."""

import torch


def decode_ue5m3(scale_bits: torch.Tensor) -> torch.Tensor:
    """Decode unsigned E5M3 bit patterns into float32 values."""
    if scale_bits.dtype != torch.uint8:
        raise TypeError(f"decode_ue5m3 expects uint8 scale bits, got {scale_bits.dtype}")

    bits = scale_bits.to(torch.int32)
    exponent = bits >> 3
    mantissa = bits & 0x7
    subnormal = mantissa.float() * 2.0**-17
    normal = torch.ldexp(1.0 + mantissa.float() * 0.125, exponent - 15)
    return torch.where(exponent == 0, subnormal, normal)


def _quantize_ue5m3_scale_reference(value: torch.Tensor) -> torch.Tensor:
    bits = value.contiguous().view(torch.int32)
    fp32_exponent = torch.bitwise_and(torch.bitwise_right_shift(bits, 23), 0xFF)
    fp32_mantissa = torch.bitwise_and(bits, 0x7FFFFF)
    ue5m3_exponent = fp32_exponent - 112
    ue5m3_mantissa = torch.bitwise_right_shift(fp32_mantissa, 20)
    remainder = torch.bitwise_and(fp32_mantissa, 0xFFFFF)
    increment = (remainder > 0x80000) | ((remainder == 0x80000) & (ue5m3_mantissa.bitwise_and(1) != 0))
    ue5m3_mantissa += increment.to(torch.int32)
    carry = ue5m3_mantissa == 8
    ue5m3_exponent += carry.to(torch.int32)
    ue5m3_mantissa = torch.where(carry, 0, ue5m3_mantissa)
    normal = torch.ldexp(1.0 + ue5m3_mantissa.float() * 0.125, ue5m3_exponent - 15)
    subnormal = torch.round(value * 131072.0) / 131072.0
    return torch.where(value < 2.0**-14, subnormal, normal).clamp_(max=114688.0)


def _nvfp4_e5m3_qdq_reference(x: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Quantize-dequantize BF16/FP16 activations using E2M1 and UE5M3 scales."""
    if x.dim() != 2:
        raise ValueError(f"nvfp4_e5m3_qdq expects a 2D tensor, got {x.dim()}D")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"nvfp4_e5m3_qdq expects BF16/FP16, got {x.dtype}")
    if group_size not in (16, 32):
        raise ValueError(f"nvfp4_e5m3_qdq requires group_size 16 or 32, got {group_size}")

    original_dtype = x.dtype
    rows, width = x.shape
    pad = (-width) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    groups = x.reshape(rows, -1, group_size)
    scales = groups.float().abs().amax(dim=-1).div_(6.0)
    scales = _quantize_ue5m3_scale_reference(scales)
    normalized = torch.where(
        scales.unsqueeze(-1) == 0,
        torch.zeros_like(groups, dtype=torch.float32),
        groups.float() / scales.unsqueeze(-1),
    )
    magnitude = normalized.abs()
    quantized = torch.where(
        magnitude < 2.0,
        torch.round(magnitude * 2.0) / 2.0,
        torch.where(magnitude < 4.0, torch.round(magnitude), 2.0 * torch.round(magnitude / 2.0)),
    ).clamp_(max=6.0)
    quantized.copysign_(normalized)
    return (quantized * scales.unsqueeze(-1)).reshape(rows, -1)[:, :width].to(original_dtype)


def nvfp4_e5m3_qdq(x: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Quantize-dequantize BF16/FP16 activations using E2M1 and UE5M3 scales."""
    if x.dim() != 2:
        raise ValueError(f"nvfp4_e5m3_qdq expects a 2D tensor, got {x.dim()}D")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"nvfp4_e5m3_qdq expects BF16/FP16, got {x.dtype}")
    if group_size not in (16, 32):
        raise ValueError(f"nvfp4_e5m3_qdq requires group_size 16 or 32, got {group_size}")
    if x.is_cuda and x.is_contiguous() and x.shape[1] % group_size == 0:
        from .nvfp4_e5m3_cute import nvfp4_e5m3_qdq_cute

        return nvfp4_e5m3_qdq_cute(x, group_size)
    return _nvfp4_e5m3_qdq_reference(x, group_size)
