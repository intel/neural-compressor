"""AutoRound NVFP4_E5M3(_GS) dense linear using vLLM FP4 Marlin."""

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
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import decode_ue5m3, fold_ue5m3_global_scale

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig

logger = init_logger(__name__)


def shared_input_global_scale(stored_scale: torch.Tensor, name: str) -> torch.Tensor | None:
    """Return one QDQ input global scale, or ``None`` when it is exactly 1 (model-free export)."""
    if torch.unique(stored_scale).numel() != 1:
        logger.warning_once(
            "NVFP4_E5M3_GS QDQ requires one shared %s; using the smallest stored value to avoid clipping.",
            name,
        )
    input_global_scale = stored_scale.min().to(torch.float32)
    if input_global_scale.item() == 1.0:
        return None
    return input_global_scale


class INCNvfp4UE5M3LinearMethod(INCLinearScheme):
    """W4A4 linear using packed E2M1 weights and raw UE5M3 scales.

    With ``with_global_scale`` (NVFP4_E5M3_GS), weights dequantize as
    ``q * scale / weight_global_scale`` and activations use ``input_global_scale``.
    """

    with_global_scale = False

    def __init__(self, layer_config: "INCLayerConfig", with_global_scale: bool = False) -> None:
        if layer_config.group_size != 16:
            raise ValueError(f"NVFP4_E5M3 Marlin linear requires group_size 16, got {layer_config.group_size!r}")
        self.group_size = layer_config.group_size
        self.with_global_scale = with_global_scale

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

        if self.with_global_scale:
            for name in ("weight_global_scale", "input_global_scale"):
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale.needs_scalar_to_array = True
                layer.register_parameter(name, scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight_packed.detach().contiguous(), requires_grad=False)
        del layer.weight_packed
        weight_scale = decode_ue5m3(layer.weight_scale)
        weight_global_scale = torch.ones((), dtype=torch.float32, device=layer.weight.device)
        if self.with_global_scale:
            logical_widths = getattr(layer, "logical_widths", [layer.output_size_per_partition])
            row_global_scale = layer.weight_global_scale.to(torch.float32).repeat_interleave(
                torch.tensor(logical_widths, device=layer.weight_global_scale.device)
            )
            if torch.unique(layer.weight_global_scale).numel() != 1:
                logger.warning_once(
                    "NVFP4_E5M3_GS fused projections use different weight global scales; "
                    "folding them into the block scales may add rounding error."
                )
            weight_scale, weight_global_scale = fold_ue5m3_global_scale(weight_scale, row_global_scale)
            input_global_scale = shared_input_global_scale(
                layer.input_global_scale, "input global scale across fused projections"
            )
            del layer.input_global_scale
            if input_global_scale is not None:
                layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.to(layer.params_dtype), requires_grad=False)
        layer.weight_global_scale = Parameter(weight_global_scale.reshape(()), requires_grad=False)
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
        if layer.weight.dtype != torch.int32:
            raise RuntimeError("NVFP4_E5M3 dense weight has not been prepared for Marlin")

        flat_x = x.reshape(-1, x.shape[-1]).contiguous()
        quantized_x = nvfp4_e5m3_qdq(
            flat_x,
            self.group_size,
            input_global_scale=getattr(layer, "input_global_scale", None),
        )
        output = apply_fp4_marlin_linear(
            input=quantized_x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=getattr(layer, "workspace", None),
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
        return output.reshape(*x.shape[:-1], layer.output_size_per_partition)
