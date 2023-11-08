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

from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

from neural_compressor.common.config import (
    GLOBAL,
    OPERATOR_NAME,
    OPERATOR_TYPE,
    BaseConfig,
    register_config,
    registered_configs,
)

FRAMEWORK_NAME = "torch"


class Backend(Enum):
    DEFAULT = "stock_pytorch"
    IPEX = "ipex"


import torch

operator2str = {torch.nn.Linear: "Linear", torch.nn.functional.linear: "linear", torch.nn.Conv2d: "Conv2d"}

str2operator = {"Linear": torch.nn.Linear, "linear": torch.nn.functional.linear, "Conv2d": torch.nn.Conv2d}


@register_config(framework_name=FRAMEWORK_NAME, algo_name="rtn_weight_only_quant")
class RTNWeightQuantConfig(BaseConfig):
    supported_configs: List[OperatorConfig] = []
    params_list = ["weight_dtype", "weight_bits", "weight_group_size", "weight_sym", "act_dtype"]

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
    ):
        super().__init__()
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype

    def to_dict(self, depth=0):
        return super().to_dict(params_list=self.params_list, operator2str=operator2str, depth=depth)

    @classmethod
    def from_dict(cls, config_dict):
        return super(RTNWeightQuantConfig, cls).from_dict(config_dict=config_dict, str2operator=str2operator)


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    backend: List[Backend]
    valid_func_list: List[Callable] = []


def _register_supported_configs(cls) -> List[OperatorConfig]:
    supported_configs = []
    linear_rtn_config = RTNWeightQuantConfig(
        weight_dtype=["int", "int8", "int4", "nf4", "fp4", "fp4_e2m1_bnb", "fp4_e2m1"],
        weight_bits=[4, 1, 2, 3, 5, 6, 7, 8],
        weight_group_size=[32, -1, 1, 4, 8, 16, 64, 128, 256, 512, 1024],
        weight_sym=[True, False],
        act_dtype=["fp32"],
    )
    operators = [torch.nn.Linear, torch.nn.functional.linear]
    supported_configs.append(OperatorConfig(config=linear_rtn_config, operators=operators, backend=Backend.DEFAULT))
    cls.supported_configs = supported_configs


_register_supported_configs(RTNWeightQuantConfig)


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    return registered_configs.get(FRAMEWORK_NAME, {})


def parse_config_from_dict(config_dict: Dict) -> BaseConfig:
    torch_registered_configs = get_all_registered_configs()
    for key, val in config_dict.items():
        if key in torch_registered_configs:
            config = torch_registered_configs[key].from_dict(val)
            return config
        # TODO(Yi) parse multiple configs after support configs add
