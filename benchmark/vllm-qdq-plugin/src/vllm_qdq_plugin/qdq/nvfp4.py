"""NVFP4 E2M1 activation quant-dequant with FP8 E4M3 block scales."""

import torch
from vllm_qdq_plugin.trace import trace_qdq

_FP4_MAX = 6.0
_FP8_E4M3_MAX = 448.0
_ROW_CHUNK_SIZE = 1024


def _nvfp4_qdq_reference(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Quantize-dequantize activations using the decoded global scale.

    ``input_global_scale`` is the scale used by the QDQ equations. It is not
    the inverse scale consumed by native NVFP4 kernels.
    """
    if x.dim() != 2:
        raise ValueError(f"nvfp4_qdq expects a 2D tensor, got {x.dim()}D")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"nvfp4_qdq expects BF16/FP16, got {x.dtype}")
    if group_size != 16:
        raise ValueError(f"nvfp4_qdq requires group_size 16, got {group_size}")
    if input_global_scale.dtype != torch.float32 or input_global_scale.numel() not in (1, x.shape[0]):
        raise ValueError(
            "nvfp4_qdq requires one float32 input_global_scale value or one per input row"
        )
    if x.shape[1] % group_size:
        raise ValueError(f"nvfp4_qdq requires K divisible by group_size={group_size}, got K={x.shape[1]}")

    original_dtype = x.dtype
    input_global_scales = input_global_scale.reshape(-1)
    output = torch.empty_like(x)
    for row_start in range(0, x.shape[0], _ROW_CHUNK_SIZE):
        row_end = min(row_start + _ROW_CHUNK_SIZE, x.shape[0])
        chunk_x = x[row_start:row_end]
        groups = chunk_x.reshape(chunk_x.shape[0], -1, group_size).float()
        chunk_scales = (
            input_global_scales
            if input_global_scales.numel() == 1
            else input_global_scales[row_start:row_end]
        )
        global_scale = chunk_scales.reshape(-1, 1)
        scales = global_scale * (groups.abs().amax(dim=-1) * (1.0 / _FP4_MAX))
        scales = scales.clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
        dequant_scales = scales * torch.where(global_scale == 0, torch.zeros_like(global_scale), 1.0 / global_scale)
        output_scales = torch.where(
            dequant_scales == 0,
            torch.zeros_like(dequant_scales),
            1.0 / dequant_scales,
        )
        normalized = groups * output_scales.unsqueeze(-1)
        magnitude = normalized.abs()
        step1 = torch.round(2.0 * magnitude) / 2.0
        step2 = torch.round(magnitude)
        step3 = 2.0 * torch.round(magnitude / 2.0)
        mask1 = magnitude < 2.0
        mask2 = magnitude < 4.0
        quantized = (step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)).clamp_(-_FP4_MAX, _FP4_MAX)
        quantized = quantized.copysign(normalized)
        dequantized = quantized * torch.where(
            output_scales == 0,
            torch.zeros_like(output_scales),
            1.0 / output_scales,
        ).unsqueeze(-1)
        output[row_start:row_end] = dequantized.reshape_as(chunk_x).to(original_dtype)
    return output


def nvfp4_qdq(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int = 16,
    *,
    topk: int = 1,
    inplace: bool = False,
    trace_op_name: str = "nvfp4_qdq",
) -> torch.Tensor:
    """Apply standard NVFP4 activation QDQ without using hardware FP4 GEMM.

    ``topk > 1`` repeats each input row ``topk`` times, so ``input_global_scale`` may hold one
    value per repeated row; ``inplace`` writes the result back into ``x``.
    """
    from .cute import use_nvfp4_cute

    if (
        x.dim() == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and group_size == 16
        and use_nvfp4_cute(x, group_size, "NVFP4")
    ):
        if input_global_scale.dtype != torch.float32 or input_global_scale.numel() not in (1, x.shape[0] * topk):
            raise ValueError("nvfp4_qdq requires one float32 input_global_scale value or one per output row")
        from .nvfp4_e5m3_cute import nvfp4_family_qdq_cute

        trace_qdq(trace_op_name, x.shape, x.dtype, backend="CuTe", format_name="NVFP4")
        return nvfp4_family_qdq_cute(x, group_size, "E4M3", input_global_scale, topk=topk, inplace=inplace)

    trace_qdq(trace_op_name, x.shape, x.dtype, backend="Reference", format_name="NVFP4")
    source = x if topk == 1 else x[:, None, :].expand(-1, topk, -1).reshape(-1, x.shape[-1])
    result = _nvfp4_qdq_reference(source, input_global_scale, group_size)
    return x.copy_(result) if inplace else result
