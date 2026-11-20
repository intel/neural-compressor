"""INC scheme for AutoRound NVFP4_E5M3 and NVFP4_E5M3_GS weights."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig
    from vllm.model_executor.layers.quantization.inc.inc import INCConfig

from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.schemes.inc_scheme import INCScheme

NVFP4_E5M3_DATA_TYPE = "nvfp4_v2"
NVFP4_E5M3_GS_DATA_TYPE = "nvfp4_v2_gs"


def is_nvfp4_e5m3_gs(data_type: str | None) -> bool:
    """Match AutoRound's NVFP4_E5M3_GS data types, including ``opt_rtn_``/``rtn_`` prefixes."""
    data_type = (data_type or "").lower()
    return NVFP4_E5M3_DATA_TYPE in data_type and data_type.endswith("_gs")


class INCNvfp4UE5M3Scheme(INCScheme):
    """Select dense and fused MoE implementations for NVFP4_E5M3(_GS)."""

    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        data_type = layer_config.data_type
        return layer_config.bits == 4 and (data_type == NVFP4_E5M3_DATA_TYPE or is_nvfp4_e5m3_gs(data_type))

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer, prefix
        from .inc_nvfp4_e5m3_linear import INCNvfp4UE5M3LinearMethod

        return INCLinearMethod(INCNvfp4UE5M3LinearMethod(layer_config, is_nvfp4_e5m3_gs(layer_config.data_type)))

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, prefix
        from .inc_nvfp4_e5m3_moe import INCNvfp4UE5M3MoEMethod

        return INCNvfp4UE5M3MoEMethod(
            layer.moe_config,
            layer_config.group_size,
            with_global_scale=is_nvfp4_e5m3_gs(layer_config.data_type),
        )
