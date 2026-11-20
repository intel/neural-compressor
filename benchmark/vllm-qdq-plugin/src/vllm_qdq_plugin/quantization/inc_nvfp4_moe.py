"""AutoRound standard NVFP4 MoE using activation QDQ and FP4 Marlin."""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported, RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, apply_moe_activation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import scalar_types
from vllm_qdq_plugin.qdq.nvfp4 import nvfp4_qdq
from vllm_qdq_plugin.qdq.nvfp4_e5m3_cute import try_fused_silu_and_mul_qdq

logger = init_logger(__name__)


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
    def _collapse_uniform(scales: torch.Tensor) -> torch.Tensor:
        """Return a 0-d scale when every expert shares it, so QDQ can skip per-route expansion."""
        scales = scales.to(torch.float32).contiguous()
        if torch.equal(scales, scales[:1].expand_as(scales)):
            return scales[0].clone().reshape(())
        return scales

    @staticmethod
    def _routed_input_scales(
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        per_expert_scales: torch.Tensor,
    ) -> torch.Tensor:
        if per_expert_scales.ndim == 0:
            return per_expert_scales
        expert_ids = topk_ids.reshape(-1).long()
        if expert_map is not None:
            global_ids = expert_ids.clamp(min=0, max=expert_map.numel() - 1)
            expert_ids = expert_map[global_ids].long()
        local_ids = expert_ids.clamp(min=0, max=per_expert_scales.numel() - 1)
        return per_expert_scales.index_select(0, local_ids)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            raise ValueError("NVFP4 QDQ requires matching gate/up weight global scales within each expert")
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_input_global_scale[:, 0], layer.w13_input_global_scale[:, 1]
        ):
            logger.warning_once(
                "NVFP4 QDQ gate/up input scales differ within an expert; using the larger stored scale."
            )
        w13_input_scale = layer.w13_input_global_scale.max(dim=1).values

        w13_weight = layer.w13_weight_packed.detach().contiguous()
        del layer.w13_weight_packed
        w2_weight = layer.w2_weight_packed.detach().contiguous()
        del layer.w2_weight_packed
        w13_weight_global_scale = 1.0 / layer.w13_weight_global_scale[:, 0].contiguous()
        w2_weight_global_scale = 1.0 / layer.w2_weight_global_scale
        layer.w13_input_scale = torch.nn.Parameter(self._collapse_uniform(w13_input_scale), requires_grad=False)
        layer.w2_input_scale = torch.nn.Parameter(
            self._collapse_uniform(layer.w2_input_global_scale), requires_grad=False
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
            input_global_scale = self._routed_input_scales(topk_ids, expert_map, layer.w2_input_scale)
            if try_fused_silu_and_mul_qdq(
                activation,
                output,
                activation_input,
                activation_config,
                self.group_size,
                "E4M3",
                input_global_scale,
                format_name="NVFP4",
                trace_op_name="nvfp4_moe_activation",
            ):
                return
            apply_moe_activation(
                activation,
                output,
                activation_input,
                activation_config=activation_config,
                topk_ids=topk_ids,
                expert_map=expert_map,
            )
            nvfp4_qdq(
                output,
                input_global_scale,
                self.group_size,
                inplace=True,
                trace_op_name="nvfp4_moe_activation",
            )

        num_tokens, topk = topk_ids.shape
        if layer.w13_input_scale.ndim == 0:
            moe_topk_ids, moe_topk_weights = topk_ids, topk_weights
            quantized_x = nvfp4_qdq(x, layer.w13_input_scale, self.group_size, trace_op_name="nvfp4_moe_input")
        else:
            # Per-expert input scales: run each (token, expert) route as its own row; the kernel
            # reads token rows directly instead of materializing a top_k-expanded copy.
            moe_topk_ids = topk_ids.reshape(-1, 1).contiguous()
            moe_topk_weights = topk_weights.reshape(-1, 1).contiguous()
            quantized_x = nvfp4_qdq(
                x,
                self._routed_input_scales(moe_topk_ids, layer.expert_map, layer.w13_input_scale),
                self.group_size,
                topk=topk,
                trace_op_name="nvfp4_moe_input",
            )
        routed_output = fused_marlin_moe(
            hidden_states=quantized_x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            bias1=None,
            bias2=None,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            topk_weights=moe_topk_weights,
            topk_ids=moe_topk_ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            activation=layer.activation,
            activation_func=activation_with_qdq,
            expert_map=layer.expert_map,
            global_scale1=layer.w13_weight_scale_2,
            global_scale2=layer.w2_weight_scale_2,
            workspace=getattr(layer, "workspace", None),
            activation_config=activation_config,
        )
        if moe_topk_ids is not topk_ids:
            return routed_output.view(num_tokens, topk, -1).sum(dim=1)
        return routed_output
