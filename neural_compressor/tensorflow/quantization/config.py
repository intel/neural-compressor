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
from typing import Callable, Dict, List, NamedTuple, Optional, Union

import tensorflow as tf

from neural_compressor.common.base_config import BaseConfig, register_config, registered_configs
from neural_compressor.common.utility import DEFAULT_WHITE_LIST, OP_NAME_OR_MODULE_TYPE, STATIC_QUANT

FRAMEWORK_NAME = "keras"


class Backend(Enum):
    DEFAULT = "keras"
    ITEX = "itex"


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    backend: List[Backend]
    valid_func_list: List[Callable] = []


# mapping the torch module type and functional operation type to string representations
operator2str = {
    tf.keras.layers.Dense: "Dense",
    tf.keras.layers.DepthwiseConv2D: "DepthwiseConv2D",
    tf.keras.layers.Conv2D: "Conv2d",
    tf.keras.layers.SeparableConv2D: "SeparableConv2D",
    tf.keras.layers.AvgPool2D: "AvgPool2D",
    tf.keras.layers.AveragePooling2D: "AveragePooling2D",
    tf.keras.layers.MaxPool2D: "MaxPool2D",
    tf.keras.layers.MaxPooling2D: "MaxPooling2D",
}

# Mapping from string representations to their corresponding torch operation/module type
str2operator = {
    "Dense": tf.keras.layers.Dense,
    "DepthwiseConv2D": tf.keras.layers.DepthwiseConv2D,
    "Conv2d": tf.keras.layers.Conv2D,
    "SeparableConv2D": tf.keras.layers.SeparableConv2D,
    "AvgPool2D": tf.keras.layers.AvgPool2D,
    "AveragePooling2D": tf.keras.layers.AveragePooling2D,
    "MaxPool2D": tf.keras.layers.MaxPool2D,
    "MaxPooling2D": tf.keras.layers.MaxPooling2D,
}


@register_config(framework_name=FRAMEWORK_NAME, algo_name=STATIC_QUANT)
class StaticQuantConfig(BaseConfig):
    """Config class for keras static quantization."""

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "weight_dtype",
        "weight_sym",
        "weight_granularity",
        "act_dtype",
        "act_sym",
        "act_granularity",
    ]

    name = STATIC_QUANT

    def __init__(
        self,
        weight_dtype: str = "int8",
        weight_sym: bool = True,
        weight_granularity: str = "per_tensor",
        act_dtype: str = "int8",
        act_sym: bool = True,
        act_granularity: str = "per_tensor",
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init static quantization config.

        Args:
            weight_dtype (str): Data type for weights, default is "int".
            weight_sym (bool): Indicates whether weights are symmetric, default is True.
            weight_granularity (str): Calculate tensor-wise scales or channel-wise scales for weights.
            act_dtype (str): Data type for activations, default is "int8".
            act_sym (bool): Indicates whether activations are symmetric, default is True.
            act_granularity (str): Calculate tensor-wise scales or channel-wise scales for activations.
        """
        super().__init__(white_list=white_list)
        self.weight_dtype = weight_dtype
        self.weight_sym = weight_sym
        self.weight_granularity = weight_granularity
        self.act_dtype = act_dtype
        self.act_sym = act_sym
        self.act_granularity = act_granularity
        self._post_init()

    def to_dict(self):
        return super().to_dict(params_list=self.params_list, operator2str=operator2str)

    @classmethod
    def from_dict(cls, config_dict):
        return super(StaticQuantConfig, cls).from_dict(config_dict=config_dict, str2operator=str2operator)

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        static_quant_config = StaticQuantConfig(
            weight_dtype=["int8", "fp32"],
            weight_sym=[True, False],
            weight_granularity=["per_tensor", "per_channel"],
            act_dtype=["int8", "fp32"],
            act_sym=[True, False],
            act_granularity=["per_tensor", "per_channel"],
        )
        operators = [
            tf.keras.layers.Dense,
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
            tf.keras.layers.SeparableConv2D,
            tf.keras.layers.AvgPool2D,
            tf.keras.layers.MaxPool2D,
            tf.keras.layers.AveragePooling2D,
            tf.keras.layers.MaxPooling2D,
        ]
        supported_configs.append(
            OperatorConfig(config=static_quant_config, operators=operators, backend=Backend.DEFAULT)
        )
        cls.supported_configs = supported_configs


# TODO(Yi) run `register_supported_configs` for all registered config.
StaticQuantConfig.register_supported_configs()


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    """Get all registered configs for keras framework."""
    return registered_configs.get(FRAMEWORK_NAME, {})


def parse_config_from_dict(config_dict: Dict) -> BaseConfig:
    """Generate a BaseConfig instance from a dict."""
    keras_registered_configs = get_all_registered_configs()
    for key, val in config_dict.items():
        if key in keras_registered_configs:
            config = keras_registered_configs[key].from_dict(val)
            return config


def get_default_static_quant_config() -> StaticQuantConfig:
    """Generate the default static quant config.

    Returns:
        the default keras config.
    """
    return StaticQuantConfig()
