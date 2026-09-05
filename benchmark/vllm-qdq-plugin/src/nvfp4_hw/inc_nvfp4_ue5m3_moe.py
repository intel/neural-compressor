"""AutoRound NVFP4_E5M3 MoE execution using vLLM FP4 Marlin."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, apply_moe_activation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import scalar_types
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import decode_ue5m3, nvfp4_e5m3_qdq


class INCNvfp4UE5M3MoEMethod(FusedMoEMethodBase):
    """Weight-only MoE using packed E2M1 weights and raw UE5M3 scales."""

    def __init__(self, moe: FusedMoEConfig, group_size: int | tuple[int, int]) -> None:
        super().__init__(moe)
        if group_size != 16:
            raise ValueError(f"NVFP4_E5M3 Marlin MoE requires group_size 16, got {group_size}")
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
        layer.hidden_size = hidden_size
        layer.intermediate_size_per_partition = intermediate_size_per_partition
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
        w13_weight = layer.w13_weight_packed.detach().contiguous()
        del layer.w13_weight_packed
        w2_weight = layer.w2_weight_packed.detach().contiguous()
        del layer.w2_weight_packed
        w13_weight_scale = decode_ue5m3(layer.w13_weight_scale)
        w2_weight_scale = decode_ue5m3(layer.w2_weight_scale)
        global_scale = torch.ones(layer.num_experts, dtype=torch.float32, device=w13_weight.device)
        (
            w13_weight,
            w13_weight_scale,
            w13_weight_scale_2,
            w2_weight,
            w2_weight_scale,
            w2_weight_scale_2,
        ) = prepare_nvfp4_moe_layer_for_marlin(
            layer=layer,
            w13=w13_weight,
            w13_scale=w13_weight_scale,
            w13_scale_2=global_scale,
            w2=w2_weight,
            w2_scale=w2_weight_scale,
            w2_scale_2=global_scale,
            is_act_and_mul=self.moe.is_act_and_mul,
        )
        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)
        layer.w13_weight_scale_2 = torch.nn.Parameter(w13_weight_scale_2, requires_grad=False)
        layer.w2_weight_scale_2 = torch.nn.Parameter(w2_weight_scale_2, requires_grad=False)

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
            raise TypeError(f"NVFP4_E5M3 MoE requires bfloat16 activations, got {x.dtype}")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError("NVFP4_E5M3 does not support apply_router_weight_on_input")

        activation_config = ApplyMoEActivationConfig(
            clamp_limit=self.moe.swiglu_limit,
            alpha=1.0 if self.moe.swiglu_alpha is None else self.moe.swiglu_alpha,
            beta=0.0 if self.moe.swiglu_beta is None else self.moe.swiglu_beta,
            activation_situ_beta=self.moe.activation_situ_beta,
            activation_situ_linear_beta=self.moe.activation_situ_linear_beta,
        )

        def activation_with_qdq(activation, output, activation_input, *, topk_ids, expert_map) -> None:
            apply_moe_activation(
                activation,
                output,
                activation_input,
                activation_config=activation_config,
                topk_ids=topk_ids,
                expert_map=expert_map,
            )
            output.copy_(nvfp4_e5m3_qdq(output.contiguous(), self.group_size))

        quantized_x = nvfp4_e5m3_qdq(x.contiguous(), self.group_size)
        return fused_marlin_moe(
            hidden_states=quantized_x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            bias1=None,
            bias2=None,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            activation=layer.activation,
            activation_func=activation_with_qdq,
            expert_map=layer.expert_map,
            global_scale1=layer.w13_weight_scale_2,
            global_scale2=layer.w2_weight_scale_2,
            workspace=layer.workspace,
            activation_config=activation_config,
        )
