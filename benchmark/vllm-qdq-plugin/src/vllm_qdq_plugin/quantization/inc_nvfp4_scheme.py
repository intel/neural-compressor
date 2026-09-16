"""INC scheme for AutoRound standard NVFP4 QDQ."""

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.schemes.inc_scheme import INCScheme

from .inc_nvfp4_linear import INCNvfp4QDQLinearMethod

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig
    from vllm.model_executor.layers.quantization.inc.inc import INCConfig


class INCNvfp4QDQScheme(INCScheme):
    """Select QDQ + Marlin methods for standard NVFP4 checkpoints."""

    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.data_type == "nv_fp" and layer_config.bits == 4

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: torch.nn.Module,
        prefix: str,
        layer_config: "INCLayerConfig",
    ) -> INCLinearMethod:
        del config, layer, prefix
        return INCLinearMethod(INCNvfp4QDQLinearMethod(layer_config))

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: torch.nn.Module,
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, prefix
        from .inc_nvfp4_moe import INCNvfp4QDQMoEMethod

        return INCNvfp4QDQMoEMethod(layer.moe_config, layer_config.group_size)
