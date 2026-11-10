"""CuTe DSL custom operators for NVFP4-family activation QDQ and NVFP4_E5M3 weight conversion."""

import torch
from vllm.utils.torch_utils import direct_register_custom_op


def _nvfp4_family_qdq_cute_impl(
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    scale_format: str,
    group_size: int,
    topk: int,
) -> torch.Tensor:
    from .cute_kernels import run_nvfp4_qdq

    return run_nvfp4_qdq(x, group_size, scale_format, global_scale, topk)


def _nvfp4_family_qdq_cute_fake(
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    scale_format: str,
    group_size: int,
    topk: int,
) -> torch.Tensor:
    del global_scale, scale_format, group_size
    return x.new_empty((x.shape[0] * topk, x.shape[1]))


direct_register_custom_op(
    op_name="qdq_plugin_nvfp4_family_qdq",
    op_func=_nvfp4_family_qdq_cute_impl,
    mutates_args=[],
    fake_impl=_nvfp4_family_qdq_cute_fake,
)


def _nvfp4_family_qdq_cute_inplace_impl(
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    scale_format: str,
    group_size: int,
) -> None:
    from .cute_kernels import run_nvfp4_qdq

    run_nvfp4_qdq(x, group_size, scale_format, global_scale, output=x)


def _nvfp4_family_qdq_cute_inplace_fake(
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    scale_format: str,
    group_size: int,
) -> None:
    del x, global_scale, scale_format, group_size


direct_register_custom_op(
    op_name="qdq_plugin_nvfp4_family_qdq_",
    op_func=_nvfp4_family_qdq_cute_inplace_impl,
    mutates_args=["x"],
    fake_impl=_nvfp4_family_qdq_cute_inplace_fake,
)


def nvfp4_family_qdq_cute(
    x: torch.Tensor,
    group_size: int,
    scale_format: str,
    global_scale: torch.Tensor | None = None,
    *,
    topk: int = 1,
    inplace: bool = False,
) -> torch.Tensor:
    """Run fused NVFP4 (``"E4M3"``) or NVFP4_E5M3 (``"UE5M3"``) activation QDQ.

    ``topk > 1`` returns each input row repeated ``topk`` times; ``inplace`` writes back into ``x``.
    """
    if global_scale is not None:
        global_scale = global_scale.reshape(-1).contiguous()
    if inplace and topk == 1 and x.is_contiguous():
        torch.ops.vllm.qdq_plugin_nvfp4_family_qdq_(x, global_scale, scale_format, group_size)
        return x
    result = torch.ops.vllm.qdq_plugin_nvfp4_family_qdq(x.contiguous(), global_scale, scale_format, group_size, topk)
    if inplace:
        return x.copy_(result)
    return result


def _nvfp4_e5m3_weight_dequant_cute_impl(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    from .cute_kernels import dequantize_nvfp4_e5m3_weight

    return dequantize_nvfp4_e5m3_weight(packed, scales, group_size)


def _nvfp4_e5m3_weight_dequant_cute_fake(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    del scales, group_size
    return torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        dtype=torch.bfloat16,
        device=packed.device,
    )


direct_register_custom_op(
    op_name="nvfp4_e5m3_weight_dequant_cute",
    op_func=_nvfp4_e5m3_weight_dequant_cute_impl,
    mutates_args=[],
    fake_impl=_nvfp4_e5m3_weight_dequant_cute_fake,
)


def nvfp4_e5m3_weight_dequant_cute(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Decode packed E2M1/UE5M3 weights into BF16 with CuTe DSL."""
    if not packed.is_cuda or not scales.is_cuda:
        raise ValueError("NVFP4_E5M3 CuTe weight dequant requires CUDA tensors")
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("NVFP4_E5M3 packed weights and scales must use uint8")
    if packed.dim() != 2 or scales.dim() != 2:
        raise ValueError("NVFP4_E5M3 packed weights and scales must be 2D")
    if packed.device != scales.device:
        raise ValueError("NVFP4_E5M3 packed weights and scales must be on the same device")
    if group_size not in (16, 32):
        raise ValueError(f"NVFP4_E5M3 requires group_size 16 or 32, got {group_size}")
    if packed.shape[0] != scales.shape[0] or packed.shape[1] * 2 != scales.shape[1] * group_size:
        raise ValueError(
            f"NVFP4_E5M3 weight/scale shapes are incompatible: {tuple(packed.shape)}, {tuple(scales.shape)}"
        )
    return torch.ops.vllm.nvfp4_e5m3_weight_dequant_cute(packed, scales, group_size)
