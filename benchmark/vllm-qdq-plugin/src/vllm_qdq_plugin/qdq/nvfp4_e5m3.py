"""NVFP4 E2M1 with UE5M3 scale activation quant-dequant."""

import torch


def nvfp4_e5m3_qdq(x: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Quantize-dequantize BF16/FP16 activations using E2M1 and UE5M3 scales."""
    from xkernels import fp32_to_ue5m3, ue5m3_to_fp32

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
    scales = ue5m3_to_fp32(fp32_to_ue5m3(scales))
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
