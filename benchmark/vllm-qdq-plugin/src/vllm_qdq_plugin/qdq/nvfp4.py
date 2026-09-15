"""NVFP4 E2M1 activation quant-dequant with FP8 E4M3 block scales."""

import torch

from vllm_qdq_plugin.trace import trace_qdq

_FP4_MAX = 6.0
_FP8_E4M3_MAX = 448.0


def _nvfp4_qdq_reference(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Quantize-dequantize activations using standard NVFP4 semantics."""
    if x.dim() != 2:
        raise ValueError(f"nvfp4_qdq expects a 2D tensor, got {x.dim()}D")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"nvfp4_qdq expects BF16/FP16, got {x.dtype}")
    if group_size != 16:
        raise ValueError(f"nvfp4_qdq requires group_size 16, got {group_size}")
    if input_global_scale.numel() != 1 or input_global_scale.dtype != torch.float32:
        raise ValueError("nvfp4_qdq requires one float32 input_global_scale value")
    if x.shape[1] % group_size:
        raise ValueError(f"nvfp4_qdq requires K divisible by group_size={group_size}, got K={x.shape[1]}")

    original_dtype = x.dtype
    groups = x.reshape(x.shape[0], -1, group_size).float()
    global_scale = input_global_scale.reshape(1, 1)
    scales = global_scale * groups.abs().amax(dim=-1).div_(_FP4_MAX)
    scales = scales.clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
    inverse_scales = torch.where(scales == 0, torch.zeros_like(scales), global_scale / scales)
    normalized = groups * inverse_scales.unsqueeze(-1)
    magnitude = normalized.abs()
    quantized = torch.where(
        magnitude < 2.0,
        torch.round(magnitude * 2.0) / 2.0,
        torch.where(magnitude < 4.0, torch.round(magnitude), 2.0 * torch.round(magnitude / 2.0)),
    ).clamp_(max=_FP4_MAX)
    quantized.copysign_(normalized)
    dequantized = quantized * (scales / global_scale).unsqueeze(-1)
    return dequantized.reshape_as(x).to(original_dtype)


def nvfp4_qdq(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int = 16,
    *,
    trace_op_name: str = "nvfp4_qdq",
) -> torch.Tensor:
    """Apply standard NVFP4 activation QDQ without using hardware FP4 GEMM."""
    trace_qdq(trace_op_name, x.shape, x.dtype, backend="Reference", format_name="NVFP4")
    return _nvfp4_qdq_reference(x, input_global_scale, group_size)
