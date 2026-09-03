"""Batched NVFP4_E5M3 MoE execution using vLLM token alignment."""

from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import apply_moe_activation
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import _e2m1_inline
from vllm.triton_utils import tl, triton


@triton.jit
def _ue5m3_to_float(scale_bits):
    scale_bits = scale_bits.to(tl.int32)
    exponent = scale_bits >> 3
    mantissa = scale_bits & 0x7
    subnormal = mantissa.to(tl.float32) * 0.00000762939453125
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(exponent.to(tl.float32) - 15.0)
    decoded = tl.where(exponent == 0, subnormal, normal)
    return tl.where(scale_bits == 0xFF, float("nan"), decoded)


@triton.jit
def _fused_moe_ue5m3_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    packed_block_k: tl.constexpr = BLOCK_SIZE_K // 2
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    token_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    token_ids = tl.load(sorted_token_ids_ptr + token_offsets).to(tl.int64)
    token_mask = token_ids < num_valid_tokens
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert_id == -1:
        return

    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    packed_k_offsets = tl.arange(0, packed_block_k)
    a_ptrs = a_ptr + token_ids[:, None] // TOP_K * stride_am + k_offsets[None, :] * stride_ak
    b_ptrs = b_ptr + expert_id * stride_be + n_offsets[:, None] * stride_bn + packed_k_offsets[None, :] * stride_bk
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = k_offsets[None, :] < K - k_block * BLOCK_SIZE_K
        packed_k_mask = packed_k_offsets[None, :] < K // 2 - k_block * packed_block_k
        activations = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask, other=0.0)
        packed_weight = tl.load(
            b_ptrs,
            mask=(n_offsets[:, None] < N) & packed_k_mask,
            other=0,
        )
        low = _e2m1_inline(packed_weight & 0x0F)
        high = _e2m1_inline((packed_weight >> 4) & 0x0F)
        scale_ptrs = (
            b_scale_ptr
            + expert_id * stride_bse
            + n_offsets[:, None] * stride_bsn
            + ((packed_k_offsets[None, :] + packed_block_k * k_block) // (GROUP_SIZE // 2)) * stride_bsk
        )
        scale_bits = tl.load(
            scale_ptrs,
            mask=(n_offsets[:, None] < N) & packed_k_mask,
            other=0,
        )
        scales = _ue5m3_to_float(scale_bits)
        weight = tl.trans(tl.interleave(low * scales, high * scales)).to(tl.bfloat16)
        accumulator = tl.dot(activations, weight, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += packed_block_k * stride_bk

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(topk_weights_ptr + token_ids, mask=token_mask, other=0.0)
        accumulator *= routed_weight[:, None]

    output_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = c_ptr + token_ids[:, None] * stride_cm + output_offsets[None, :] * stride_cn
    output_mask = token_mask[:, None] & (output_offsets[None, :] < N)
    tl.store(output_ptrs, accumulator.to(tl.bfloat16), mask=output_mask)


def _invoke_fused_moe_ue5m3(
    activations: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    group_size: int,
    mul_routed_weight: bool,
    config: dict[str, Any],
) -> None:
    n = weight.shape[1]
    k = activations.shape[1]
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    em = sorted_token_ids.shape[0]
    if activations.shape[0] < block_m:
        em = min(em, activations.shape[0] * top_k * block_m)
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, block_n),)
    _fused_moe_ue5m3_kernel[grid](
        activations,
        weight,
        output,
        scales,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        em,
        output.shape[0],
        activations.stride(0),
        activations.stride(1),
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        scales.stride(0),
        scales.stride(2),
        scales.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        TOP_K=top_k,
        GROUP_SIZE=group_size,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


@torch.compiler.assume_constant_result
def _select_moe_config(
    w13: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    num_tokens: int,
) -> dict[str, Any]:
    num_experts = w13.shape[0]
    hidden_size = w13.shape[2] * 2
    intermediate_size = w2.shape[2] * 2
    config = try_get_optimal_moe_config(
        (num_experts, w13.shape[1], hidden_size),
        (num_experts, hidden_size, intermediate_size),
        top_k,
        None,
        num_tokens,
    )
    required = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages")
    if any(key not in config for key in required):
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    return config


def fused_moe_ue5m3(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    activation_config: ApplyMoEActivationConfig,
    group_size: int,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run both routed expert projections without a Python expert loop."""
    from vllm_qdq_plugin.qdq.nvfp4_e5m3 import nvfp4_e5m3_qdq

    num_tokens, hidden_size = x.shape
    top_k = topk_ids.shape[1]
    config = _select_moe_config(w13, w2, top_k, num_tokens)
    global_num_experts = expert_map.numel() if expert_map is not None else w13.shape[0]
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        global_num_experts,
        expert_map,
        ignore_invalid_experts=True,
    )
    num_assignments = num_tokens * top_k
    gate_up = torch.zeros((num_assignments, w13.shape[1]), dtype=x.dtype, device=x.device)
    quantized_x = nvfp4_e5m3_qdq(x, group_size)
    _invoke_fused_moe_ue5m3(
        quantized_x,
        w13,
        w13_scale,
        gate_up,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        group_size,
        False,
        config,
    )
    activated = torch.empty((num_assignments, w13.shape[1] // 2), dtype=x.dtype, device=x.device)
    apply_moe_activation(activation, activated, gate_up, activation_config=activation_config)
    activated = nvfp4_e5m3_qdq(activated, group_size)
    expert_output = torch.zeros((num_assignments, hidden_size), dtype=x.dtype, device=x.device)
    _invoke_fused_moe_ue5m3(
        activated,
        w2,
        w2_scale,
        expert_output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        1,
        group_size,
        True,
        config,
    )
    return expert_output.view(num_tokens, top_k, hidden_size).sum(dim=1)
