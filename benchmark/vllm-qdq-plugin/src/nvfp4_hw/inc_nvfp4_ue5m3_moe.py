"""AutoRound NVFP4_E5M3 MoE execution backed by xkernels."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs

from .fused_moe_ue5m3 import fused_moe_ue5m3


class INCNvfp4UE5M3MoEMethod(FusedMoEMethodBase):
    """Weight-only MoE using packed E2M1 weights and raw UE5M3 scales."""

    def __init__(self, moe: FusedMoEConfig, group_size: int | tuple[int, int]) -> None:
        super().__init__(moe)
        if group_size not in (16, 32):
            raise ValueError(f"NVFP4_E5M3 requires group_size 16 or 32, got {group_size}")
        self.group_size = group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if hidden_size % self.group_size or intermediate_size_per_partition % self.group_size:
            raise ValueError(f"NVFP4_E5M3 dimensions must be divisible by group_size {self.group_size}")

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        w13_size = self.moe.w13_num_shards * intermediate_size_per_partition

        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_size, hidden_size // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size_per_partition // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        scale_attrs = {
            **extra_weight_attrs,
            "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
        }
        w13_scale = torch.nn.Parameter(
            torch.empty(num_experts, w13_size, hidden_size // self.group_size, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, scale_attrs)

        w2_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, scale_attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight_packed.data, requires_grad=False)
        del layer.w13_weight_packed
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight_packed.data, requires_grad=False)
        del layer.w2_weight_packed
        layer.w13_weight_scale = torch.nn.Parameter(layer.w13_weight_scale.contiguous(), requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(layer.w2_weight_scale.contiguous(), requires_grad=False)

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if x.dtype != torch.bfloat16:
            raise TypeError(f"NVFP4_E5M3 xkernels require bfloat16 activations, got {x.dtype}")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError("NVFP4_E5M3 does not support apply_router_weight_on_input")

        activation_config = ApplyMoEActivationConfig(
            clamp_limit=self.moe.swiglu_limit,
            alpha=1.0 if self.moe.swiglu_alpha is None else self.moe.swiglu_alpha,
            beta=0.0 if self.moe.swiglu_beta is None else self.moe.swiglu_beta,
            activation_situ_beta=self.moe.activation_situ_beta,
            activation_situ_linear_beta=self.moe.activation_situ_linear_beta,
        )
        return fused_moe_ue5m3(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            layer.activation,
            activation_config,
            self.group_size,
            layer.expert_map,
        )
