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


import os
import sys
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
    "convert_offical_config_into_hqq_config",
]


class HQQGlobalOptions:
    use_half = os.getenv("HQQ_NOT_USE_HALF", "0") == "0"


hqq_global_option = HQQGlobalOptions()


@dataclass
class QTensorConfig:
    nbits: int
    channel_wise: bool = True
    group_size: int = 128
    optimize: bool = True
    round_zero: Optional[bool] = False
    pack: bool = True

    def __repr__(self) -> str:
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
    def __new__(
        cls,
        weight=default_weight_quant_config,
        scale=default_scale_quant_config,
        zero=default_zero_quant_config,
    ):
        return super().__new__(cls, weight, scale, zero)

    def __repr__(self) -> str:
        return (
            f"HQQModuleConfig(\n" f"    weight={self.weight},\n" f"    scale={self.scale},\n" f"    zero={self.zero}\n)"
        )


default_hqq_module_config = HQQModuleConfig(
    weight=default_weight_quant_config,
    scale=default_scale_quant_config,
    zero=default_zero_quant_config,
)


def convert_offical_config_into_hqq_config(quant_config_offical) -> HQQModuleConfig:
    # def hqq_base_quant_config(nbits=4, group_size=64, quant_zero=True, quant_scale=False, scale_quant_group_size=128):
    #    assert nbits in Quantizer.SUPPORTED_BITS, "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
    #    if(group_size is not None):
    #       assert is_divisible(group_size, 8), "Invalid group_size param: the value should be a multiple of 8."
    #    weight_quant_params = {'nbits':nbits,'channel_wise':True,  'group_size':group_size, 'optimize':True, 'round_zero':True if nbits==4 else False}
    #    scale_quant_params  = {'nbits':8,    'channel_wise':True,  'group_size':scale_quant_group_size,        'optimize':False} if (quant_scale) else None
    #    zero_quant_params   = {'nbits':8,    'channel_wise':False, 'group_size':None,       'optimize':False} if (quant_zero)  else None
    #    return {'weight_quant_params':weight_quant_params, 'scale_quant_params':scale_quant_params, 'zero_quant_params':zero_quant_params}

    weight_quant_params = quant_config_offical["weight_quant_params"]
    scale_quant_params = quant_config_offical["scale_quant_params"]
    zero_quant_params = quant_config_offical["zero_quant_params"]
    weight = QTensorConfig(**weight_quant_params)
    scale = None
    zero = None
    if scale_quant_params:
        scale = QTensorConfig(**scale_quant_params)
    if zero_quant_params:
        zero = QTensorConfig(**zero_quant_params)
    return HQQModuleConfig(weight, scale, zero)


ConfigMappingType: TypeAlias = Dict[str, HQQModuleConfig]
