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

import re
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import onnx

from neural_compressor.common import Logger
from neural_compressor.common.base_config import BaseConfig, register_config
from neural_compressor.common.utils import DEFAULT_WHITE_LIST, OP_NAME_OR_MODULE_TYPE, RTN

logger = Logger().get_logger()

FRAMEWORK_NAME = "onnxrt"


class Backend(Enum):
    DEFAULT = "onnxrt_cpu"
    CUDA = "onnxrt_cuda"


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    backend: List[Backend]
    valid_func_list: List[Callable] = []


######################## RNT Config ###############################


@register_config(framework_name=FRAMEWORK_NAME, algo_name=RTN)
class RTNConfig(BaseConfig):
    """Config class for round-to-nearest weight-only quantization."""

    supported_configs: List[OperatorConfig] = []
    node_params_list = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "accuracy_level",
    ]
    model_params_list = ["providers"]
    params_list = node_params_list + model_params_list
    name = RTN

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
        accuracy_level: int = 0,
        providers: list = ["CPUExecutionProvider"],
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init RTN weight-only quantization config.

        Args:
            weight_dtype (str): Data type for weights, default is "int".
            weight_bits (int): Number of bits used to represent weights, default is 4.
            weight_group_size (int): Size of weight groups, default is 32.
            weight_sym (bool): Indicates whether weights are symmetric, default is True.
            act_dtype (str): Data type for activations, default is "fp32".
        """
        super().__init__(white_list=white_list)
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype
        self.accuracy_level = accuracy_level
        self.providers = providers
        self._post_init()

    def get_model_params_dict(self):
        result = dict()
        for param in self.model_params_list:
            result[param] = getattr(self, param)
        return result

    def to_dict(self):
        return super().to_dict(params_list=self.params_list)

    @classmethod
    def from_dict(cls, config_dict):
        return super(RTNConfig, cls).from_dict(config_dict=config_dict)

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        linear_rtn_config = RTNConfig(
            weight_dtype=["int"],
            weight_bits=[4, 3, 8],
            weight_group_size=[32, -1, 1, 16, 64, 128, 256, 512, 1024],
            weight_sym=[True, False],
            act_dtype=["fp32"],
        )
        operators = ["MatMul"]
        supported_configs.append(OperatorConfig(config=linear_rtn_config, operators=operators, backend=Backend.DEFAULT))
        cls.supported_configs = supported_configs

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDict[Union[str, Callable], OrderedDict[str, BaseConfig]]:
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            # update model level setting
            config_mapping.update(config.get_model_params_dict())

            # update node level setting
            global_config = config.global_config
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            for op_name, op_type in model_info:
                if self.global_config is not None:
                    config_mapping[(op_name, op_type)] = global_config
                if op_type in op_type_config_dict:
                    config_mapping[(op_name, op_type)] = op_name_config_dict[op_type]
                for op_name_pattern in op_name_config_dict:
                    if re.match(op_name_pattern, op_name):
                        config_mapping[(op_name, op_type)] = op_name_config_dict[op_name_pattern]
        return config_mapping

    @staticmethod
    def get_model_info(model: Union[onnx.ModelProto, Path, str]) -> List[Tuple[str, Callable]]:
        if not isinstance(model, onnx.ModelProto):
            model = onnx.load(model)
        white_list = ["MatMul"]
        filter_result = []
        for node in model.graph.node:
            if node.op_type in white_list:
                pair = (node.name, node.op_type)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result


# TODO(Yi) run `register_supported_configs` for all registered config.
RTNConfig.register_supported_configs()


def get_default_rtn_config() -> RTNConfig:
    """Generate the default rtn config.

    Returns:
        the default rtn config.
    """
    return RTNConfig()
