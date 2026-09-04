"""AutoRound NVFP4_E5M3 dense linear using vLLM FP4 Marlin."""

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.quantization.inc.schemes.inc_scheme import INCLinearScheme
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import decode_ue5m3

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig


class INCNvfp4UE5M3LinearMethod(INCLinearScheme):
    """W4A4 linear using packed E2M1 weights and raw UE5M3 scales."""

    def __init__(self, layer_config: "INCLayerConfig") -> None:
        if layer_config.group_size != 16:
            raise ValueError(f"NVFP4_E5M3 Marlin linear requires group_size 16, got {layer_config.group_size!r}")
        self.group_size = layer_config.group_size

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size
        if input_size_per_partition % self.group_size:
            raise ValueError(
                "NVFP4_E5M3 input size per partition must be divisible by " f"group_size {self.group_size}"
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight_packed.detach().contiguous(), requires_grad=False)
        del layer.weight_packed
        layer.weight_scale = Parameter(
            decode_ue5m3(layer.weight_scale).to(layer.params_dtype),
            requires_grad=False,
        )
        layer.weight_global_scale = Parameter(
            torch.ones((), dtype=torch.float32, device=layer.weight.device),
            requires_grad=False,
        )
        prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_qdq_plugin.qdq.nvfp4_e5m3 import nvfp4_e5m3_qdq

        if x.dtype != torch.bfloat16:
            raise TypeError(f"NVFP4_E5M3 dense linear requires bfloat16 activations, got {x.dtype}")
        if not hasattr(layer, "workspace"):
            raise RuntimeError("NVFP4_E5M3 dense weight has not been prepared for Marlin")

        flat_x = x.reshape(-1, x.shape[-1]).contiguous()
        quantized_x = nvfp4_e5m3_qdq(flat_x, self.group_size)
        output = apply_fp4_marlin_linear(
            input=quantized_x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
        return output.reshape(*x.shape[:-1], layer.output_size_per_partition)
