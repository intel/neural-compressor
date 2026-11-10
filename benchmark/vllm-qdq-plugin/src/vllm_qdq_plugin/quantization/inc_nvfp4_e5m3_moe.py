"""AutoRound NVFP4_E5M3(_GS) MoE execution using vLLM FP4 Marlin."""

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
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import decode_ue5m3, fold_ue5m3_global_scale, nvfp4_e5m3_qdq


class INCNvfp4UE5M3MoEMethod(FusedMoEMethodBase):
    """MoE using packed E2M1 weights, raw UE5M3 scales, and optional per-expert global scales."""

    with_global_scale = False

    def __init__(
        self,
        moe: FusedMoEConfig,
        group_size: int | tuple[int, int],
        with_global_scale: bool = False,
    ) -> None:
        super().__init__(moe)
        if group_size != 16:
            raise ValueError(f"NVFP4_E5M3 Marlin MoE requires group_size 16, got {group_size}")
        self.group_size = group_size
        self.with_global_scale = with_global_scale

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

        scale_attrs = {**extra_weight_attrs, "quant_method": FusedMoeWeightScaleSupported.GROUP.value}
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

        if self.with_global_scale:
            num_shards = self.moe.w13_num_shards
            tensor_attrs = {**extra_weight_attrs, "quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            tensor_specs = (
                ("w13_weight_global_scale", (num_experts, num_shards)),
                ("w2_weight_global_scale", (num_experts,)),
                ("w13_input_global_scale", (num_experts, num_shards)),
                ("w2_input_global_scale", (num_experts,)),
            )
            for name, shape in tensor_specs:
                scale = torch.nn.Parameter(torch.ones(*shape, dtype=torch.float32), requires_grad=False)
                layer.register_parameter(name, scale)
                set_weight_attrs(scale, tensor_attrs)

    def _fold_global_scales(
        self,
        layer: RoutedExperts,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_shards = layer.w13_weight_global_scale.shape[-1]
        w13_row_scale = layer.w13_weight_global_scale.repeat_interleave(
            layer.w13_weight_scale.shape[1] // num_shards, dim=1
        )
        w2_row_scale = layer.w2_weight_global_scale[:, None].expand(-1, layer.w2_weight_scale.shape[1])
        w13_weight_scale, w13_global_scale = self._decode_scales(layer, layer.w13_weight_scale, w13_row_scale)
        w2_weight_scale, w2_global_scale = self._decode_scales(layer, layer.w2_weight_scale, w2_row_scale)

        w13_input_global_scale = self._prepare_input_global_scale(
            layer.w13_input_global_scale, "w13 input global scale"
        )
        w2_input_global_scale = self._prepare_input_global_scale(layer.w2_input_global_scale, "w2 input global scale")
        for name in (
            "w13_weight_global_scale",
            "w2_weight_global_scale",
            "w13_input_global_scale",
            "w2_input_global_scale",
        ):
            delattr(layer, name)
        if w13_input_global_scale is not None:
            layer.w13_input_global_scale = torch.nn.Parameter(w13_input_global_scale, requires_grad=False)
        if w2_input_global_scale is not None:
            layer.w2_input_global_scale = torch.nn.Parameter(w2_input_global_scale, requires_grad=False)
        return w13_weight_scale, w13_global_scale, w2_weight_scale, w2_global_scale

    @staticmethod
    def _decode_scales(
        layer: RoutedExperts,
        scale_bits: torch.Tensor,
        row_global_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode (and fold) UE5M3 scales one expert at a time to bound float32 temporaries."""
        # UE5M3 values and their folded/requantized forms are exact in BF16/FP16 mantissas.
        decoded = torch.empty(scale_bits.shape, dtype=layer.params_dtype, device=scale_bits.device)
        global_scale = torch.ones(scale_bits.shape[0], dtype=torch.float32, device=scale_bits.device)
        for expert in range(scale_bits.shape[0]):
            expert_scale = decode_ue5m3(scale_bits[expert])
            if row_global_scale is not None:
                expert_scale, global_scale[expert] = fold_ue5m3_global_scale(expert_scale, row_global_scale[expert])
            decoded[expert] = expert_scale
        return decoded, global_scale

    @staticmethod
    def _prepare_input_global_scale(stored_scale: torch.Tensor, name: str) -> torch.Tensor | None:
        if stored_scale.ndim == 2:
            if not torch.equal(stored_scale, stored_scale[:, :1].expand_as(stored_scale)):
                raise ValueError(f"NVFP4_E5M3_GS requires matching {name} across gate/up shards per expert")
            stored_scale = stored_scale[:, 0]
        elif stored_scale.ndim != 1:
            raise ValueError(f"NVFP4_E5M3_GS expects one {name} value per expert")

        stored_scale = stored_scale.to(torch.float32)
        if torch.all(stored_scale == 1):
            return None
        if torch.equal(stored_scale, stored_scale[:1].expand_as(stored_scale)):
            return stored_scale[0].reshape(())
        return stored_scale.contiguous()

    @staticmethod
    def _scale_for_routes(
        scale_per_expert: torch.Tensor | None,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if scale_per_expert is None or scale_per_expert.ndim == 0:
            return scale_per_expert
        expert_ids = topk_ids.reshape(-1).to(torch.long)
        if expert_map is not None:
            global_ids = expert_ids.clamp(min=0, max=expert_map.numel() - 1)
            expert_ids = expert_map[global_ids].to(torch.long)
        local_ids = expert_ids.clamp(min=0, max=scale_per_expert.numel() - 1)
        return scale_per_expert[local_ids]

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13_weight = layer.w13_weight_packed.detach().contiguous()
        del layer.w13_weight_packed
        w2_weight = layer.w2_weight_packed.detach().contiguous()
        del layer.w2_weight_packed
        if self.with_global_scale:
            w13_weight_scale, w13_global_scale, w2_weight_scale, w2_global_scale = self._fold_global_scales(layer)
        else:
            w13_weight_scale, w13_global_scale = self._decode_scales(layer, layer.w13_weight_scale, None)
            w2_weight_scale, w2_global_scale = self._decode_scales(layer, layer.w2_weight_scale, None)
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
            w13_scale_2=w13_global_scale,
            w2=w2_weight,
            w2_scale=w2_weight_scale,
            w2_scale_2=w2_global_scale,
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
            nvfp4_e5m3_qdq(
                output,
                self.group_size,
                input_global_scale=self._scale_for_routes(
                    getattr(layer, "w2_input_global_scale", None), topk_ids, expert_map
                ),
                inplace=True,
                trace_op_name="nvfp4_e5m3_moe_activation",
            )

        num_tokens, top_k = topk_ids.shape
        w13_input_scale = getattr(layer, "w13_input_global_scale", None)
        if w13_input_scale is not None and w13_input_scale.ndim == 1:
            # Per-expert input scales: run each (token, expert) route as its own row; the kernel
            # reads token rows directly instead of materializing a top_k-expanded copy.
            moe_topk_ids = topk_ids.reshape(-1, 1).contiguous()
            moe_topk_weights = topk_weights.reshape(-1, 1).contiguous()
            quantized_x = nvfp4_e5m3_qdq(
                x,
                self.group_size,
                input_global_scale=self._scale_for_routes(w13_input_scale, moe_topk_ids, layer.expert_map),
                topk=top_k,
                trace_op_name="nvfp4_e5m3_moe_input",
            )
        else:
            quantized_x = nvfp4_e5m3_qdq(
                x,
                self.group_size,
                input_global_scale=w13_input_scale,
                trace_op_name="nvfp4_e5m3_moe_input",
            )
            moe_topk_ids = topk_ids
            moe_topk_weights = topk_weights

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
            return routed_output.view(num_tokens, top_k, -1).sum(dim=1)
        return routed_output
