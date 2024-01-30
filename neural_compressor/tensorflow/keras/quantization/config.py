#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import tensorflow as tf

from neural_compressor.common import logger
from neural_compressor.common.base_config import BaseConfig, config_registry, register_config
from neural_compressor.common.utils import DEFAULT_WHITE_LIST, OP_NAME_OR_MODULE_TYPE, STATIC_QUANT

FRAMEWORK_NAME = "keras"


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


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
        supported_configs.append(OperatorConfig(config=static_quant_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model) -> List[Tuple[str, Callable]]:
        white_list = [
            "Dense",
            "Conv2d",
            "DepthwiseConv2D",
            "SeparableConv2D",
            "AvgPool2D",
            "AveragePooling2D",
            "MaxPool2D",
            "MaxPooling2D",
        ]
        filter_result = []

        for layer in model.model.layers:
            if layer.__class__.__name__ in white_list:
                pair = (layer.name, layer.__class__.__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result


# TODO(Yi) run `register_supported_configs` for all registered config.
StaticQuantConfig.register_supported_configs()


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    """Get all registered configs for keras framework."""
    registered_configs = config_registry.get_cls_configs()
    return registered_configs.get(FRAMEWORK_NAME, {})


def parse_keras_config_from_dict(config_dict: Dict) -> BaseConfig:
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


support_int8_weight = {"Dense", "Conv2d", "DepthwiseConv2D", "SeparableConv2D"}

support_int8_activation = {
    "Dense",
    "Conv2d",
    "DepthwiseConv2D",
    "SeparableConv2D",
    "AvgPool2D",
    "AveragePooling2D",
    "MaxPool2D",
    "MaxPooling2D",
}


def update_config(op_value: Dict, quant_config: StaticQuantConfig, op_key: Tuple):
    """Update op-wise config from global config or operator name config or operator type config."""
    op_value["activation"].update(
        {
            "dtype": quant_config.act_dtype,
            "quant_mode": "static",
            "scheme": ("sym" if quant_config.act_sym else "asym"),
            "granularity": quant_config.act_granularity,
            "algorithm": "minmax",
        }
    )
    if op_key[1] not in support_int8_weight:
        return
    op_value["weight"] = {
        "dtype": quant_config.weight_dtype,
        "scheme": "sym" if quant_config.weight_sym else "asym",
        "granularity": quant_config.weight_granularity,
        "algorithm": "minmax",
    }


def parse_to_keras_tune_cfg(model: tf.keras.Model, quant_config: StaticQuantConfig, calib_iteration: int) -> Dict:
    """The function that parses StaticQuantConfig to keras tuning config.

    Args:
        model: a fp32 model to be quantized.
        quant_config: a quantization configuration.
        calib_iteration: the iteration of calibration.

    Returns:
        tune_cfg: the tuning config for keras adaptor.
    """
    tune_cfg = {"op": OrderedDict()}
    for op_key, config in quant_config.items():
        op_value = {"activation": {}}

        update_config(op_value, config, op_key)

        tune_cfg["op"].update({op_key: op_value})
        tune_cfg["calib_iteration"] = calib_iteration

    return tune_cfg
