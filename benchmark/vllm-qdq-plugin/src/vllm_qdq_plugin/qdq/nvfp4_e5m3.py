"""NVFP4 E2M1 with UE5M3 scale activation quant-dequant."""

import torch
from vllm_qdq_plugin import envs
from vllm_qdq_plugin.trace import trace_qdq

# Keep Marlin block scales inside the FP8-E4M3 range its S0E5M3 scale path is tuned for.
_MARLIN_SCALE_MAX = 448.0


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


def _nvfp4_e5m3_qdq_reference(
    x: torch.Tensor,
    group_size: int = 16,
    input_global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize-dequantize BF16/FP16 activations using E2M1 and UE5M3 scales.

    With ``input_global_scale`` (NVFP4_E5M3_GS), each block scale is
    ``UE5M3(global_scale * amax / 6)`` and dequantizes as ``scale / global_scale``.
    """
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
    if input_global_scale is not None:
        global_scale = _as_global_scale(input_global_scale, x.device)
        scales = _quantize_ue5m3_scale_reference(scales.mul_(global_scale))
        inverse_global_scale = torch.where(global_scale == 0, torch.zeros_like(global_scale), 1.0 / global_scale)
        scales = scales * inverse_global_scale
    else:
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


def _as_global_scale(global_scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    if global_scale.numel() != 1 or global_scale.dtype != torch.float32:
        raise ValueError("NVFP4_E5M3_GS QDQ requires one float32 input_global_scale value")
    return global_scale.reshape(()).to(device)


def nvfp4_e5m3_qdq(
    x: torch.Tensor,
    group_size: int = 16,
    *,
    input_global_scale: torch.Tensor | None = None,
    trace_op_name: str = "nvfp4_e5m3_qdq",
) -> torch.Tensor:
    """Quantize-dequantize BF16/FP16 activations using E2M1 and UE5M3 scales.

    ``input_global_scale`` enables NVFP4_E5M3_GS semantics; ``None`` keeps the
    global-scale-free NVFP4_E5M3 path.
    """
    if x.dim() != 2:
        raise ValueError(f"nvfp4_e5m3_qdq expects a 2D tensor, got {x.dim()}D")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"nvfp4_e5m3_qdq expects BF16/FP16, got {x.dtype}")
    if group_size not in (16, 32):
        raise ValueError(f"nvfp4_e5m3_qdq requires group_size 16 or 32, got {group_size}")
    if input_global_scale is not None:
        return _nvfp4_e5m3_gs_qdq(x, group_size, input_global_scale, trace_op_name)
    if not envs.VLLM_QDQ_CUTE:
        trace_qdq(trace_op_name, x.shape, x.dtype, backend="Reference", format_name="NVFP4_E5M3")
        return _nvfp4_e5m3_qdq_reference(x, group_size)
    if torch.compiler.is_compiling() and x.is_cuda:
        from .nvfp4_e5m3_cute import nvfp4_e5m3_qdq_cute

        trace_qdq(trace_op_name, x.shape, x.dtype, backend="CuTe", format_name="NVFP4_E5M3")
        return nvfp4_e5m3_qdq_cute(x, group_size)
    from .cute import cute_qdq_status, warn_reference_fallback

    available, reason = cute_qdq_status(x)
    if available and x.is_contiguous() and x.shape[1] % group_size == 0:
        from .nvfp4_e5m3_cute import nvfp4_e5m3_qdq_cute

        trace_qdq(trace_op_name, x.shape, x.dtype, backend="CuTe", format_name="NVFP4_E5M3")
        return nvfp4_e5m3_qdq_cute(x, group_size)
    if available and not x.is_contiguous():
        reason = "input is not contiguous"
    elif available:
        reason = f"K={x.shape[1]} is not divisible by group_size={group_size}"
    warn_reference_fallback("NVFP4_E5M3", reason)
    trace_qdq(trace_op_name, x.shape, x.dtype, backend="Reference", format_name="NVFP4_E5M3")
    return _nvfp4_e5m3_qdq_reference(x, group_size)


def _nvfp4_e5m3_gs_qdq(
    x: torch.Tensor,
    group_size: int,
    input_global_scale: torch.Tensor,
    trace_op_name: str,
) -> torch.Tensor:
    format_name = "NVFP4_E5M3_GS"
    global_scale = _as_global_scale(input_global_scale, x.device)
    if not envs.VLLM_QDQ_CUTE:
        trace_qdq(trace_op_name, x.shape, x.dtype, backend="Reference", format_name=format_name)
        return _nvfp4_e5m3_qdq_reference(x, group_size, global_scale)
    if torch.compiler.is_compiling() and x.is_cuda and group_size == 16:
        from .nvfp4_e5m3_cute import nvfp4_e5m3_gs_qdq_cute

        trace_qdq(trace_op_name, x.shape, x.dtype, backend="CuTe", format_name=format_name)
        return nvfp4_e5m3_gs_qdq_cute(x, global_scale)
    from .cute import cute_qdq_status, warn_reference_fallback

    available, reason = cute_qdq_status(x)
    if available and group_size == 16 and x.is_contiguous() and x.shape[1] % group_size == 0:
        from .nvfp4_e5m3_cute import nvfp4_e5m3_gs_qdq_cute

        trace_qdq(trace_op_name, x.shape, x.dtype, backend="CuTe", format_name=format_name)
        return nvfp4_e5m3_gs_qdq_cute(x, global_scale)
    if available and group_size != 16:
        reason = f"group_size={group_size} is unsupported"
    elif available and not x.is_contiguous():
        reason = "input is not contiguous"
    elif available:
        reason = f"K={x.shape[1]} is not divisible by group_size={group_size}"
    warn_reference_fallback(format_name, reason)
    trace_qdq(trace_op_name, x.shape, x.dtype, backend="Reference", format_name=format_name)
    return _nvfp4_e5m3_qdq_reference(x, group_size, global_scale)


def fold_ue5m3_global_scale(
    scales: torch.Tensor,
    row_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold per-row weight global scales into decoded UE5M3 block scales for FP4 Marlin.

    ``scales`` is ``[..., rows, groups]`` and ``row_global_scale`` is ``[..., rows]``
    (AutoRound convention: ``weight = q * scale / global_scale``). Returns float32
    block scales whose maximum fits Marlin's scale range and the float32 multiplier
    ``[...]`` that Marlin applies as the weight global scale. Rows whose global
    scale differs from the largest one are rescaled; equal global scales and the
    power-of-two range shift are exact.
    """
    row_global_scale = row_global_scale.to(device=scales.device, dtype=torch.float32)
    # An all-zero tensor exports global_scale=0 together with zero block scales.
    row_global_scale = torch.where(row_global_scale > 0, row_global_scale, torch.ones_like(row_global_scale))
    reference_scale = row_global_scale.amax(dim=-1, keepdim=True)
    ratio = reference_scale / row_global_scale
    scales = scales.float() * ratio.unsqueeze(-1)
    scale_max = scales.flatten(start_dim=-2).amax(dim=-1)
    shift = torch.where(
        scale_max > _MARLIN_SCALE_MAX,
        torch.ceil(torch.log2(scale_max / _MARLIN_SCALE_MAX)).exp2(),
        torch.ones_like(scale_max),
    )
    scales = scales / shift[..., None, None]
    if (ratio != 1).any():
        # Marlin truncates scales to 3 mantissa bits; round-to-nearest first to halve the error.
        scales = _quantize_ue5m3_scale_reference(scales.contiguous())
    return scales, shift / reference_scale.squeeze(-1)
