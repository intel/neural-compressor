"""Torch custom operators backed by xkernels NVFP4 kernels."""

from typing import Any

import torch
from vllm.utils.torch_utils import direct_register_custom_op


def _mxfp4_ue5m3_bf16_gemm_impl(
    activations: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    marlin_weight: torch.Tensor,
    marlin_scales: torch.Tensor,
    workspace: torch.Tensor,
    bias: torch.Tensor | None,
    n: int,
    k: int,
    padded_n: int,
    padded_k: int,
    block_size: int,
) -> torch.Tensor:
    from xkernels.marlin import MarlinMXFP4UE5M3TensorCoreWeight, mxfp4_ue5m3_marlin_bf16_gemm

    prepared = MarlinMXFP4UE5M3TensorCoreWeight(
        packed=packed,
        scales=scales,
        marlin_weight=marlin_weight,
        marlin_scales=marlin_scales,
        workspace=workspace,
        n=n,
        k=k,
        padded_n=padded_n,
        padded_k=padded_k,
        block_size=block_size,
    )
    return mxfp4_ue5m3_marlin_bf16_gemm(activations, prepared, bias)


def _mxfp4_ue5m3_bf16_gemm_fake(
    activations: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    marlin_weight: torch.Tensor,
    marlin_scales: torch.Tensor,
    workspace: torch.Tensor,
    bias: torch.Tensor | None,
    n: int,
    k: int,
    padded_n: int,
    padded_k: int,
    block_size: int,
) -> torch.Tensor:
    del packed, scales, marlin_weight, marlin_scales, workspace, bias, k, padded_n, padded_k, block_size
    return torch.empty((*activations.shape[:-1], n), dtype=activations.dtype, device=activations.device)


direct_register_custom_op(
    op_name="nvfp4_mxfp4_ue5m3_bf16_gemm",
    op_func=_mxfp4_ue5m3_bf16_gemm_impl,
    mutates_args=["workspace"],
    fake_impl=_mxfp4_ue5m3_bf16_gemm_fake,
)


def mxfp4_ue5m3_bf16_gemm(
    activations: torch.Tensor,
    prepared: Any,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the xkernels Marlin GEMM behind a Dynamo-visible custom op."""
    return torch.ops.vllm.nvfp4_mxfp4_ue5m3_bf16_gemm(
        activations,
        prepared.packed,
        prepared.scales,
        prepared.marlin_weight,
        prepared.marlin_scales,
        prepared.workspace,
        bias,
        prepared.n,
        prepared.k,
        prepared.padded_n,
        prepared.padded_k,
        prepared.block_size,
    )
