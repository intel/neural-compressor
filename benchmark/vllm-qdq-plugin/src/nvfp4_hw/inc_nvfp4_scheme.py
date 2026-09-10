# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.schemes.inc_scheme import INCScheme

from .inc_nvfp4_linear import INCNvfp4LinearMethod
from .inc_nvfp4_moe import INCNvfp4MoEMethod

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig
    from vllm.model_executor.layers.quantization.inc.inc import INCConfig


class INCNvfp4Scheme(INCScheme):
    """INC scheme adapter that selects the native NVFP4 methods."""

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
        return INCLinearMethod(INCNvfp4LinearMethod(layer_config))

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: torch.nn.Module,
        prefix: str,
        layer_config: "INCLayerConfig",
    ) -> INCNvfp4MoEMethod:
        del config, prefix, layer_config
        return INCNvfp4MoEMethod(layer.moe_config)
