"""INC scheme for AutoRound NVFP4_E5M3 weights."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig
    from vllm.model_executor.layers.quantization.inc.inc import INCConfig

from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.schemes.inc_scheme import INCScheme


class INCNvfp4UE5M3Scheme(INCScheme):
    """Select the CuTe dense and fused MoE implementations for NVFP4_E5M3."""

    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.data_type == "nvfp4_v2" and layer_config.bits == 4

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer, prefix
        from .inc_nvfp4_ue5m3_linear import INCNvfp4UE5M3LinearMethod

        return INCLinearMethod(INCNvfp4UE5M3LinearMethod(layer_config))

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, prefix
        from .inc_nvfp4_ue5m3_moe import INCNvfp4UE5M3MoEMethod

        return INCNvfp4UE5M3MoEMethod(layer.moe_config, layer_config.group_size)
