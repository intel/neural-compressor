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


import sys

# TODO: remove it before merge
hqq_offical_path = "/home/yliu7/workspace/hqq"
sys.path.insert(0, hqq_offical_path)

from collections import namedtuple
from typing import Any, Dict, Optional, Tuple, Union

__all__ = [
    "QuantTensorConfig",
    "HQQModuleConfig",
    "default_hqq_quant_config",
]


class QuantTensorConfig:
    def __init__(
        self,
        nbits: int,
        channel_wise: bool = True,
        group_size: int = 128,
        optimize: bool = True,
        round_zero: Optional[bool] = False,
        pack: bool = True,
    ) -> None:
        self.nbits = nbits
        self.channel_wise = channel_wise
        self.group_size = group_size
        self.optimize = optimize
        self.round_zero = round_zero
        self.pack = pack

    def __repr__(self) -> str:
        return (
            f"QuantTensorConfig(nbits={self.nbits}, channel_wise={self.channel_wise}, "
            f"group_size={self.group_size}, optimize={self.optimize}, "
            f"round_zero={self.round_zero}, pack={self.pack})"
        )


default_weight_quant_config = QuantTensorConfig(
    nbits=4, channel_wise=True, group_size=128, optimize=True, round_zero=True
)
default_scale_quant_config = QuantTensorConfig(
    nbits=8, channel_wise=True, group_size=64, optimize=False, round_zero=None
)
default_zero_quant_config = QuantTensorConfig(
    nbits=8, channel_wise=False, group_size=None, optimize=False, round_zero=None
)


# def hqq_base_quant_config(nbits=4, group_size=64, quant_zero=True, quant_scale=False, scale_quant_group_size=128):
#    assert nbits in Quantizer.SUPPORTED_BITS, "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
#    if(group_size is not None):
#       assert is_divisible(group_size, 8), "Invalid group_size param: the value should be a multiple of 8."
#    weight_quant_params = {'nbits':nbits,'channel_wise':True,  'group_size':group_size, 'optimize':True, 'round_zero':True if nbits==4 else False}
#    scale_quant_params  = {'nbits':8,    'channel_wise':True,  'group_size':scale_quant_group_size,        'optimize':False} if (quant_scale) else None
#    zero_quant_params   = {'nbits':8,    'channel_wise':False, 'group_size':None,       'optimize':False} if (quant_zero)  else None
#    return {'weight_quant_params':weight_quant_params, 'scale_quant_params':scale_quant_params, 'zero_quant_params':zero_quant_params}


def convert_offical_config_into_hqq_config(quant_config_offical):
    weight_quant_params = quant_config_offical["weight_quant_params"]
    scale_quant_params = quant_config_offical["scale_quant_params"]
    zero_quant_params = quant_config_offical["zero_quant_params"]
    weight_quant_config = QuantTensorConfig(**weight_quant_params)
    scale_quant_config = None
    zero_quant_config = None
    if scale_quant_params:
        scale_quant_config = QuantTensorConfig(**scale_quant_params)
    if zero_quant_params:
        zero_quant_config = QuantTensorConfig(**zero_quant_params)
    return HQQModuleConfig(weight_quant_config, scale_quant_config, zero_quant_config)


class HQQModuleConfig(
    namedtuple(
        "HQQModuleConfig",
        ["weight_quant_config", "scale_quant_config", "zero_quant_config"],
    )
):
    def __new__(
        cls,
        weight_quant_config=default_weight_quant_config,
        scale_quant_config=default_scale_quant_config,
        zero_quant_config=default_zero_quant_config,
    ):
        return super().__new__(cls, weight_quant_config, scale_quant_config, zero_quant_config)

    def __repr__(self) -> str:
        return (
            f"HQQModuleConfig( \nweight_quant_config={self.weight_quant_config}, \n"
            f"scale_quant_config={self.scale_quant_config}, \n"
            f"zero_quant_config={self.zero_quant_config})"
        )


default_hqq_quant_config = HQQModuleConfig(
    weight_quant_config=default_weight_quant_config,
    scale_quant_config=default_scale_quant_config,
    zero_quant_config=default_zero_quant_config,
)


class QTensorMetaInfo:
    def __init__(self, nbits, group_size, shape, axis, packing):
        self.nbits = nbits
        self.group_size = group_size
        self.shape = shape
        self.axis = axis
        self.packing = packing

    def __repr__(self) -> str:
        return (
            f"QTensorMetaInfo(nbits={self.nbits}, group_size={self.group_size}, "
            f"shape={self.shape}, axis={self.axis}, packing={self.packing})"
        )
