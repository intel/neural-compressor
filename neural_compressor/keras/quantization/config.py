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
from typing import Callable, Dict, List, NamedTuple, Union

import tensorflow as tf

from neural_compressor.common.base_config import BaseConfig, register_config, registered_configs
from neural_compressor.common.utility import KERAS_STATIC_QUANT
from neural_compressor.config import PostTrainingQuantConfig

FRAMEWORK_NAME = "keras"


class Backend(Enum):
    DEFAULT = "keras"
    ITEX = "itex"


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    backend: List[Backend]
    valid_func_list: List[Callable] = []


@register_config(framework_name=FRAMEWORK_NAME, algo_name=KERAS_STATIC_QUANT)
class KerasStaticQuantConfig(PostTrainingQuantConfig):
    """Config class for keras static quantization."""

    supported_configs: List[OperatorConfig] = []
    params_list = []
    name = KERAS_STATIC_QUANT

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        keras_static_quant_config = KerasStaticQuantConfig()
        operators = [tf.keras.layers.dense, tf.keras.layers.conv2d]
        supported_configs.append(
            OperatorConfig(config=keras_static_quant_config, operators=operators, backend=Backend.DEFAULT)
        )
        cls.supported_configs = supported_configs


# TODO(Yi) run `register_supported_configs` for all registered config.
KerasStaticQuantConfig.register_supported_configs()


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    return registered_configs.get(FRAMEWORK_NAME, {})


def parse_config_from_dict(config_dict: Dict) -> BaseConfig:
    keras_registered_configs = get_all_registered_configs()
    for key, val in config_dict.items():
        if key in keras_registered_configs:
            config = keras_registered_configs[key].from_dict(val)
            return config
        # TODO(Yi) parse multiple configs after support configs add


def get_default_keras_config() -> KerasStaticQuantConfig:
    """Generate the default keras config.

    Returns:
        the default keras config.
    """
    return KerasStaticQuantConfig()
