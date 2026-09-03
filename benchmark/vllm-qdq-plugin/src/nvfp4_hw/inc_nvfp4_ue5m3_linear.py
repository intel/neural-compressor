"""AutoRound NVFP4_E5M3 dense linear with CuTe weight conversion."""

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.quantization.inc.schemes.inc_scheme import INCLinearScheme
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter
from vllm_qdq_plugin import envs

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
        self.weight_dequant_mode = envs.VLLM_NVFP4_E5M3_WEIGHT_DEQUANT_MODE
        config = get_current_vllm_config_or_none()
        linear_backend = config.kernel_config.linear_backend if config is not None else "auto"
        self._gemm_impl = dispatch_unquantized_gemm(linear_backend)

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
        if self.weight_dequant_mode == "PER_CALL":
            layer.weight_packed = Parameter(layer.weight_packed.contiguous(), requires_grad=False)
            layer.weight_scale = Parameter(layer.weight_scale.contiguous(), requires_grad=False)
            return

        from vllm_qdq_plugin.qdq.nvfp4_e5m3_cute import nvfp4_e5m3_weight_dequant_cute

        weight = nvfp4_e5m3_weight_dequant_cute(
            layer.weight_packed.contiguous(),
            layer.weight_scale.contiguous(),
            self.group_size,
        )
        layer.weight = Parameter(weight, requires_grad=False)
        del layer.weight_packed
        del layer.weight_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_qdq_plugin.qdq.nvfp4_e5m3 import nvfp4_e5m3_qdq

        if x.dtype != torch.bfloat16:
            raise TypeError(f"NVFP4_E5M3 dense linear requires bfloat16 activations, got {x.dtype}")
        if self.weight_dequant_mode == "ONCE" and not hasattr(layer, "weight"):
            raise RuntimeError("NVFP4_E5M3 dense weight has not been dequantized")
        if self.weight_dequant_mode == "PER_CALL" and not hasattr(layer, "weight_packed"):
            raise RuntimeError("NVFP4_E5M3 packed dense weight is unavailable")

        flat_x = x.reshape(-1, x.shape[-1]).contiguous()
        quantized_x = nvfp4_e5m3_qdq(flat_x, self.group_size)
        if self.weight_dequant_mode == "PER_CALL":
            from vllm_qdq_plugin.qdq.nvfp4_e5m3_cute import nvfp4_e5m3_weight_dequant_cute

            weight = nvfp4_e5m3_weight_dequant_cute(layer.weight_packed, layer.weight_scale, self.group_size)
        else:
            weight = layer.weight
        output = self._gemm_impl(layer, quantized_x, weight, bias)
        return output.reshape(*x.shape[:-1], layer.output_size_per_partition)
