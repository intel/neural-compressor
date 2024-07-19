# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for HQQ."""

import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Optional

from typing_extensions import TypeAlias

__all__ = [
    "ConfigMappingType",
    "HQQModuleConfig",
    "QTensorConfig",
    "hqq_global_option",
    "default_hqq_module_config",
    "default_weight_quant_config",
    "default_scale_quant_config",
    "default_zero_quant_config",
]


class HQQGlobalOptions:
    """Global options for HQQ."""

    use_half = os.getenv("HQQ_NOT_USE_HALF", "0") == "0"


hqq_global_option = HQQGlobalOptions()


@dataclass
class QTensorConfig:
    """Configuration class for quantized tensors."""

    nbits: int
    channel_wise: bool = True
    group_size: int = 128
    optimize: bool = True
    round_zero: Optional[bool] = False
    pack: bool = True

    def __repr__(self) -> str:
        """Return a string representation of the QTensorConfig."""
        return (
            f"QTensorConfig(nbits={self.nbits}, channel_wise={self.channel_wise}, "
            f"group_size={self.group_size}, optimize={self.optimize}, "
            f"round_zero={self.round_zero}, pack={self.pack})"
        )


default_weight_quant_config = QTensorConfig(nbits=4, channel_wise=True, group_size=128, optimize=True, round_zero=True)
default_scale_quant_config = QTensorConfig(nbits=8, channel_wise=True, group_size=64, optimize=False, round_zero=None)
default_zero_quant_config = QTensorConfig(nbits=8, channel_wise=False, group_size=None, optimize=False, round_zero=None)


class HQQModuleConfig(
    namedtuple(
        "HQQModuleConfig",
        ["weight", "scale", "zero"],
    )
):
    """Configuration class for HQQModule.

    Args:
        weight (Any): The weight quantization configuration.
        scale (Any): The scale quantization configuration.
        zero (Any): The zero quantization configuration.
    """

    def __new__(
        cls,
        weight=default_weight_quant_config,
        scale=default_scale_quant_config,
        zero=default_zero_quant_config,
    ):
        """Create a new HQQModuleConfig."""
        return super().__new__(cls, weight, scale, zero)

    def __repr__(self) -> str:
        """Return a string representation of the HQQModuleConfig."""
        return (
            f"HQQModuleConfig(\n" f"    weight={self.weight},\n" f"    scale={self.scale},\n" f"    zero={self.zero}\n)"
        )


default_hqq_module_config = HQQModuleConfig(
    weight=default_weight_quant_config,
    scale=default_scale_quant_config,
    zero=default_zero_quant_config,
)


ConfigMappingType: TypeAlias = Dict[str, HQQModuleConfig]
