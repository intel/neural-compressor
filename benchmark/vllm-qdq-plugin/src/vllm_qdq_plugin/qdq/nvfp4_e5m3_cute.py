"""CuTe DSL custom operators for NVFP4_E5M3 activation and weight conversion."""

import torch
from vllm.utils.torch_utils import direct_register_custom_op


def _nvfp4_e5m3_qdq_cute_impl(x: torch.Tensor, group_size: int) -> torch.Tensor:
    from .cute_kernels import run_nvfp4_e5m3_qdq

    return run_nvfp4_e5m3_qdq(x, group_size)


def _nvfp4_e5m3_qdq_cute_fake(x: torch.Tensor, group_size: int) -> torch.Tensor:
    del group_size
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="nvfp4_e5m3_qdq_cute",
    op_func=_nvfp4_e5m3_qdq_cute_impl,
    mutates_args=[],
    fake_impl=_nvfp4_e5m3_qdq_cute_fake,
)


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


def nvfp4_e5m3_qdq_cute(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Run fused CuTe activation QDQ."""
    return torch.ops.vllm.nvfp4_e5m3_qdq_cute(x, group_size)


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
