"""NVFP4 E2M1 with UE5M3 scale activation quant-dequant."""

import torch
from vllm_qdq_plugin.trace import trace_qdq

# Keep Marlin block scales inside the FP8-E4M3 range its S0E5M3 scale path is tuned for.
_MARLIN_SCALE_MAX = 448.0
_ROW_CHUNK_SIZE = 1024


def decode_ue5m3(scale_bits: torch.Tensor) -> torch.Tensor:
    """Decode unsigned E5M3 bit patterns into float32 values."""
    if scale_bits.dtype != torch.uint8:
        raise TypeError(f"decode_ue5m3 expects uint8 scale bits, got {scale_bits.dtype}")

    # A 256-entry lookup keeps the temporary footprint to one int32 index per scale.
    codes = torch.arange(256, dtype=torch.int32, device=scale_bits.device)
    exponent = codes >> 3
    mantissa = codes & 0x7
    subnormal = mantissa.float() * 2.0**-17
    normal = torch.ldexp(1.0 + mantissa.float() * 0.125, exponent - 15)
    table = torch.where(exponent == 0, subnormal, normal)
    return table.index_select(0, scale_bits.reshape(-1).to(torch.int32)).reshape(scale_bits.shape)


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

    rows = x.shape[0]
    global_scale = None
    if input_global_scale is not None:
        global_scale = _as_global_scale(input_global_scale, x.device, rows)
    if rows <= _ROW_CHUNK_SIZE:
        return _nvfp4_e5m3_qdq_reference_rows(x, group_size, global_scale)
    output = torch.empty_like(x)
    for start in range(0, rows, _ROW_CHUNK_SIZE):
        end = min(start + _ROW_CHUNK_SIZE, rows)
        chunk_scale = global_scale if global_scale is None or global_scale.ndim == 0 else global_scale[start:end]
        output[start:end] = _nvfp4_e5m3_qdq_reference_rows(x[start:end], group_size, chunk_scale)
    return output


def _nvfp4_e5m3_qdq_reference_rows(
    x: torch.Tensor,
    group_size: int,
    global_scale: torch.Tensor | None,
) -> torch.Tensor:
    original_dtype = x.dtype
    rows, width = x.shape
    pad = (-width) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    groups = x.reshape(rows, -1, group_size)
    scales = groups.float().abs().amax(dim=-1).div_(6.0)
    if global_scale is not None:
        scales = _quantize_ue5m3_scale_reference(scales.mul_(global_scale))
        inverse_global_scale = torch.where(global_scale == 0, torch.zeros_like(global_scale), 1.0 / global_scale)
        scales = scales * inverse_global_scale
    else:
        scales = _quantize_ue5m3_scale_reference(scales)
    normalized = torch.where(scales.unsqueeze(-1) == 0, 0.0, groups.float() / scales.unsqueeze(-1))
    magnitude = normalized.abs()
    quantized = torch.where(
        magnitude < 2.0,
        torch.round(magnitude * 2.0) / 2.0,
        torch.where(magnitude < 4.0, torch.round(magnitude), 2.0 * torch.round(magnitude / 2.0)),
    ).clamp_(max=6.0)
    quantized.copysign_(normalized)
    return (quantized * scales.unsqueeze(-1)).reshape(rows, -1)[:, :width].to(original_dtype)


def _as_global_scale(global_scale: torch.Tensor, device: torch.device, rows: int) -> torch.Tensor:
    if global_scale.dtype != torch.float32 or global_scale.numel() not in (1, rows):
        raise ValueError("NVFP4_E5M3_GS QDQ requires one float32 scale or one scale per row")
    if global_scale.numel() == 1:
        return global_scale.reshape(()).to(device)
    return global_scale.reshape(rows, 1).to(device)


def nvfp4_e5m3_qdq(
    x: torch.Tensor,
    group_size: int = 16,
    *,
    input_global_scale: torch.Tensor | None = None,
    topk: int = 1,
    inplace: bool = False,
    trace_op_name: str = "nvfp4_e5m3_qdq",
) -> torch.Tensor:
    """Quantize-dequantize BF16/FP16 activations using E2M1 and UE5M3 scales.

    ``input_global_scale`` enables NVFP4_E5M3_GS semantics; ``None`` keeps the
    global-scale-free NVFP4_E5M3 path. ``topk > 1`` repeats each input row ``topk``
    times (the scale may hold one value per repeated row); ``inplace`` writes into ``x``.
    """
    if x.dim() != 2:
        raise ValueError(f"nvfp4_e5m3_qdq expects a 2D tensor, got {x.dim()}D")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"nvfp4_e5m3_qdq expects BF16/FP16, got {x.dtype}")
    if group_size not in (16, 32):
        raise ValueError(f"nvfp4_e5m3_qdq requires group_size 16 or 32, got {group_size}")
    from .cute import use_nvfp4_cute

    format_name = "NVFP4_E5M3" if input_global_scale is None else "NVFP4_E5M3_GS"
    global_scale = None
    if input_global_scale is not None:
        global_scale = _as_global_scale(input_global_scale, x.device, x.shape[0] * topk)
    if use_nvfp4_cute(x, group_size, format_name):
        from .nvfp4_e5m3_cute import nvfp4_family_qdq_cute

        trace_qdq(trace_op_name, x.shape, x.dtype, backend="CuTe", format_name=format_name)
        return nvfp4_family_qdq_cute(x, group_size, "UE5M3", global_scale, topk=topk, inplace=inplace)

    trace_qdq(trace_op_name, x.shape, x.dtype, backend="Reference", format_name=format_name)
    source = x if topk == 1 else x[:, None, :].expand(-1, topk, -1).reshape(-1, x.shape[-1])
    result = _nvfp4_e5m3_qdq_reference(source, group_size, global_scale)
    return x.copy_(result) if inplace else result


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
