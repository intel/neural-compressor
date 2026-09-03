"""AutoRound NVFP4_E5M3 dense linear execution backed by xkernels."""

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.quantization.inc.schemes.inc_scheme import INCLinearScheme
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter

from .xkernels_ops import mxfp4_ue5m3_bf16_gemm

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig


class INCNvfp4UE5M3LinearMethod(INCLinearScheme):
    """W4A4 linear using packed E2M1 weights and raw UE5M3 scales."""

    def __init__(self, layer_config: "INCLayerConfig") -> None:
        if not isinstance(layer_config.group_size, int) or layer_config.group_size not in (16, 32):
            raise ValueError(
                "NVFP4_E5M3 linear requires scalar group_size 16 or 32, " f"got {layer_config.group_size!r}"
            )
        self.group_size = layer_config.group_size
        self._prepared_weight = None

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
        from xkernels import prepare_mxfp4_ue5m3_weight

        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        self._prepared_weight = prepare_mxfp4_ue5m3_weight(
            layer.weight.contiguous(),
            layer.weight_scale.contiguous(),
            self.group_size,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_qdq_plugin.qdq.nvfp4_e5m3 import nvfp4_e5m3_qdq

        if self._prepared_weight is None:
            raise RuntimeError("NVFP4_E5M3 linear weights have not been prepared")
        if x.dtype != torch.bfloat16:
            raise TypeError(f"NVFP4_E5M3 xkernels require bfloat16 activations, got {x.dtype}")

        flat_x = x.reshape(-1, x.shape[-1]).contiguous()
        quantized_x = nvfp4_e5m3_qdq(flat_x, self.group_size)
        output = mxfp4_ue5m3_bf16_gemm(quantized_x, self._prepared_weight, bias)
        return output.reshape(*x.shape[:-1], layer.output_size_per_partition)
