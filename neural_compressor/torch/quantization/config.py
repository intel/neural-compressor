#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""
#TODO
- [ ] support configs add, like RTNWeightOnlyQuantConfig() + GPTQWeightOnlyQuantConfig()
- [ ]
"""


import json
from abc import ABC, abstractclassmethod
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import torch

registered_configs = {}


def register_config(name=None):
    def decorator(config_cls):
        registered_configs[name] = config_cls
        return config_cls

    return decorator


class Backend(Enum):
    DEFAULT = "stock_pytorch"
    IPEX = "ipex"


class BaseConfig(ABC):
    """from_dict
    to_dict
    from_json_file
    to_json_file
    __repr__"""

    def __init__(self) -> None:
        self.global_config: Optional[BaseConfig] = None
        self.operator_type_config: Dict[Union[str, Callable], Optional[BaseConfig]] = {}
        self.operator_name_config: Dict[str, Optional[BaseConfig]] = {}
        # operator_type is the collective name for PyTorch's module type and operator type

    def set_global(self, config: BaseConfig):
        self.global_config = config

    def set_operator_type(self, operator_type: Union[str, Callable], config: BaseConfig) -> BaseConfig:
        self.operator_type_config[operator_type] = config
        return self

    def set_operator_name(self, operator_name: str, config: BaseConfig) -> BaseConfig:
        self.operator_name_config[operator_name] = config
        return self

    def __repr__(self) -> str:
        return f"{self.to_dict()}"

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "r") as file:
            config_dict = json.load(file)
        return cls(**config_dict)

    def to_json_file(self, filename):
        config_dict = self.to_dict()
        with open(filename, "w") as file:
            json.dump(config_dict, file, indent=4)


OperatorPatternType = List[Callable]


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[OperatorPatternType]
    backend: List[Backend]
    valid_func_lst: List[Callable] = []


@register_config(name="rtn_weight_only_quantization")
class RTNWeightQuantConfig(BaseConfig):
    supported_config_and_operators: List[OperatorConfig] = []

    def __init__(
        self,
        bits: int = 4,
        dtype: str = "nf4",
        group_size: int = 32,
        group_dim: int = -1,
        sym: bool = True,
        sym_full_range: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.bits = bits
        self.dtype = dtype
        self.group_size = group_size
        self.group_dim = group_dim
        self.sym = sym
        self.sym_full_range = sym_full_range
        self.kwargs = kwargs

    def to_dict(self):
        # TODO(Yi)
        return {
            "bits": self.bits,
            "dtype": self.dtype,
            "group_size": self.group_size,
            "group_dim": self.group_dim,
            "sym": self.sym,
            "sym_full_range": self.sym_full_range,
            **self.kwargs,
        }


# fwk developer
def _get_supported_config_and_operators(cls) -> List[OperatorConfig]:
    supported_config_and_operators = []
    linear_rtn_config = RTNWeightQuantConfig(
        bits=[4, 1, 2, 3, 5, 6, 7, 8], group_size=[32, -1, 1, 4, 8, 16, 64, 128, 256, 512, 1024], sym=[True, False]
    )
    operators = [[torch.nn.Linear], [torch.nn.functional.linear]]
    supported_config_and_operators.append(
        OperatorConfig(config=linear_rtn_config, operators=operators, backend=Backend.DEFAULT, valid_func_lst=[])
    )
    cls.supported_config_and_operators = supported_config_and_operators


_get_supported_config_and_operators(RTNWeightQuantConfig)

print(len(RTNWeightQuantConfig.supported_config_and_operators))

for cfg in RTNWeightQuantConfig.supported_config_and_operators:
    print(cfg.config)
    print(cfg.operators)
    print(cfg.backend)
    print(cfg.valid_func_lst)
    print("............")


print("==============")

## End-user
from neural_compressor.torch.quantization.config import BaseConfig, RTNWeightQuantConfig

global_config = RTNWeightQuantConfig(bits=4, dtype="nf4")
qconfig = BaseConfig()
qconfig.set_global(global_config)
conv_config = RTNWeightQuantConfig(bits=6, dtype="nf4")
qconfig.set_operator_type(torch.nn.Conv2d, conv_config)
conv1_config = RTNWeightQuantConfig(bits=4, dtype="int8")
qconfig.set_operator_name("model.conv1", conv1_config)

from neural_compressor.torch.quantization.quantize import quantize


class UserMolde(torch.nn.Module):
    pass


q_model = quantize(UserMolde(), qconfig)
