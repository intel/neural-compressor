"""AutoRound standard NVFP4 dense linear using QDQ and FP4 Marlin."""

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.inc.schemes.inc_scheme import INCLinearScheme
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter, PerTensorScaleParameter
from vllm_qdq_plugin.qdq.nvfp4 import nvfp4_qdq

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig

logger = init_logger(__name__)


class INCNvfp4QDQLinearMethod(INCLinearScheme):
    """Weight-only Marlin linear with standard NVFP4 activation QDQ."""

    def __init__(self, layer_config: "INCLayerConfig") -> None:
        if layer_config.group_size != 16:
            raise ValueError(f"NVFP4 QDQ Marlin linear requires group_size 16, got {layer_config.group_size!r}")
        self.group_size = layer_config.group_size

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

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
            raise ValueError(f"NVFP4 QDQ input size must be divisible by group_size {self.group_size}")

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition // 2, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        for name in ("weight_global_scale", "input_global_scale"):
            scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            scale.needs_scalar_to_array = True
            layer.register_parameter(name, scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if torch.unique(layer.input_global_scale).numel() != 1 or torch.unique(layer.weight_global_scale).numel() != 1:
            logger.warning_once(
                "NVFP4 QDQ requires fused projections to share global scales; using the largest stored divisor."
            )

        layer.weight = Parameter(layer.weight_packed.detach().contiguous(), requires_grad=False)
        del layer.weight_packed
        input_global_scale = layer.input_global_scale.max().to(torch.float32)
        weight_global_scale_inv = layer.weight_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
        layer.weight_global_scale = Parameter(1.0 / weight_global_scale_inv, requires_grad=False)
        prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise TypeError(f"NVFP4 QDQ dense linear requires bfloat16 activations, got {x.dtype}")
        if not hasattr(layer, "workspace"):
            raise RuntimeError("NVFP4 QDQ dense weight has not been prepared for Marlin")

        flat_x = x.reshape(-1, x.shape[-1]).contiguous()
        quantized_x = nvfp4_qdq(flat_x, layer.input_global_scale, self.group_size)
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
