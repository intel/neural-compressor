"""AutoRound standard NVFP4 MoE using activation QDQ and FP4 Marlin."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported, RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, apply_moe_activation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import scalar_types
from vllm_qdq_plugin.qdq.nvfp4 import nvfp4_qdq


class INCNvfp4QDQMoEMethod(FusedMoEMethodBase):
    """Weight-only Marlin MoE with standard NVFP4 activation QDQ."""

    def __init__(self, moe: FusedMoEConfig, group_size: int | tuple[int, int]) -> None:
        super().__init__(moe)
        if group_size != 16:
            raise ValueError(f"NVFP4 QDQ Marlin MoE requires group_size 16, got {group_size}")
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
            raise ValueError(f"NVFP4 QDQ dimensions must be divisible by group_size {self.group_size}")

        layer.num_experts = num_experts
        layer.hidden_size = hidden_size
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.params_dtype = params_dtype
        num_shards = self.moe.w13_num_shards
        w13_size = num_shards * intermediate_size_per_partition

        weight_specs = (
            ("w13_weight_packed", (num_experts, w13_size, hidden_size // 2)),
            ("w2_weight_packed", (num_experts, hidden_size, intermediate_size_per_partition // 2)),
        )
        for name, shape in weight_specs:
            weight = torch.nn.Parameter(torch.empty(*shape, dtype=torch.uint8), requires_grad=False)
            layer.register_parameter(name, weight)
            set_weight_attrs(weight, extra_weight_attrs)

        scale_attrs = {**extra_weight_attrs, "quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        scale_specs = (
            ("w13_weight_scale", (num_experts, w13_size, hidden_size // self.group_size)),
            ("w2_weight_scale", (num_experts, hidden_size, intermediate_size_per_partition // self.group_size)),
        )
        for name, shape in scale_specs:
            scale = torch.nn.Parameter(torch.empty(*shape, dtype=torch.float8_e4m3fn), requires_grad=False)
            layer.register_parameter(name, scale)
            set_weight_attrs(scale, scale_attrs)

        tensor_attrs = {**extra_weight_attrs, "quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        tensor_specs = (
            ("w13_weight_global_scale", (num_experts, num_shards)),
            ("w2_weight_global_scale", (num_experts,)),
            ("w13_input_global_scale", (num_experts, num_shards)),
            ("w2_input_global_scale", (num_experts,)),
        )
        for name, shape in tensor_specs:
            scale = torch.nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)
            layer.register_parameter(name, scale)
            set_weight_attrs(scale, tensor_attrs)

    @staticmethod
    def _shared_input_scale(stored_scale: torch.Tensor, name: str) -> torch.Tensor:
        if torch.unique(stored_scale).numel() != 1:
            raise ValueError(f"NVFP4 QDQ Marlin MoE requires one shared {name} across all experts")
        return (1.0 / stored_scale.flatten()[0]).to(torch.float32)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13_weight = layer.w13_weight_packed.detach().contiguous()
        del layer.w13_weight_packed
        w2_weight = layer.w2_weight_packed.detach().contiguous()
        del layer.w2_weight_packed

        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            raise ValueError("NVFP4 QDQ requires matching gate/up weight global scales")
        w13_weight_global_scale = 1.0 / layer.w13_weight_global_scale[:, 0].contiguous()
        w2_weight_global_scale = 1.0 / layer.w2_weight_global_scale
        layer.w13_input_scale = torch.nn.Parameter(
            self._shared_input_scale(layer.w13_input_global_scale, "w13 input global scale"),
            requires_grad=False,
        )
        layer.w2_input_scale = torch.nn.Parameter(
            self._shared_input_scale(layer.w2_input_global_scale, "w2 input global scale"),
            requires_grad=False,
        )

        (
            w13_weight,
            w13_weight_scale,
            w13_weight_global_scale,
            w2_weight,
            w2_weight_scale,
            w2_weight_global_scale,
        ) = prepare_nvfp4_moe_layer_for_marlin(
            layer=layer,
            w13=w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=w13_weight_global_scale,
            w2=w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=w2_weight_global_scale,
            is_act_and_mul=self.moe.is_act_and_mul,
        )
        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)
        layer.w13_weight_scale_2 = torch.nn.Parameter(w13_weight_global_scale, requires_grad=False)
        layer.w2_weight_scale_2 = torch.nn.Parameter(w2_weight_global_scale, requires_grad=False)

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
            raise TypeError(f"NVFP4 QDQ MoE requires bfloat16 activations, got {x.dtype}")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError("NVFP4 QDQ does not support apply_router_weight_on_input")

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
            output.copy_(
                nvfp4_qdq(
                    output.contiguous(),
                    layer.w2_input_scale,
                    self.group_size,
                    trace_op_name="nvfp4_moe_activation",
                )
            )

        quantized_x = nvfp4_qdq(
            x.contiguous(),
            layer.w13_input_scale,
            self.group_size,
            trace_op_name="nvfp4_moe_input",
        )
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
