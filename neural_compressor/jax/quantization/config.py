#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024-2026 Intel Corporation
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
"""The configs of algorithms for JAX."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import keras

from neural_compressor.common.base_config import (
    DEFAULT_WHITE_LIST,
    OP_NAME_OR_MODULE_TYPE,
    BaseConfig,
    config_registry,
    register_config,
    register_supported_configs_for_fwk,
)
from neural_compressor.common.utils import DYNAMIC_QUANT, STATIC_QUANT

FRAMEWORK_NAME = "jax"


class OperatorConfig(NamedTuple):
    """The config for operator."""

    config: BaseConfig
    operators: List[str]


@register_config(framework_name=FRAMEWORK_NAME, algo_name=DYNAMIC_QUANT)
class DynamicQuantConfig(BaseConfig):
    """Config class for JAX Dynamic quantization.

    Dynamic quantization applies quantization to both weights and activations during runtime.
    This configuration supports various data types for flexible quantization strategies.

    Supported dtypes:
        - "fp8": 8-bit floating-point quantization (uses ml_dtypes.float8_e4m3 by default)

    FP8 formats available:
        - "fp8_e4m3": 4 exponent bits, 3 mantissa bits (default for "fp8")
        - "fp8_e5m2": 5 exponent bits, 2 mantissa bits
    """

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "weight_dtype",
        "activation_dtype",
    ]

    name = DYNAMIC_QUANT

    def __init__(
        self,
        weight_dtype: str = "fp8_e4m3",
        activation_dtype: str = "fp8_e4m3",
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init Dynamic quantization config.

        Args:
            weight_dtype (str): Data type for weights, default is "fp8_e4m3".
            activation_dtype (str): Data type for activations, default is "fp8_e4m3".
            white_list (list): A list of supported operators of this algorithm.
        """
        super().__init__(white_list=white_list)
        self.weight_dtype = weight_dtype
        self.activation_dtype = activation_dtype
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configs."""
        supported_configs = []
        dynamic_config = DynamicQuantConfig(
            weight_dtype=["fp8", "fp8_e4m3", "fp8_e5m2"],
            activation_dtype=["fp8", "fp8_e4m3", "fp8_e5m2"],
        )
        # Basic JAX operators for quantization
        operators = [keras.layers.Dense]
        supported_configs.append(OperatorConfig(config=dynamic_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model) -> List[Tuple[str, Callable]]:
        """Get concrete node names for supported operators."""
        white_list = ["Dense", "EinsumDense"]
        filter_result = []

        for layer in model._flatten_layers(recursive=True):
            if layer.__class__.__name__ in white_list:
                pair = (layer.name, layer.__class__.__name__)
                if pair not in filter_result:
                    filter_result.append(pair)

        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "DynamicQuantConfig", List["DynamicQuantConfig"]]:
        """Get a default config set for tuning."""
        return DynamicQuantConfig(weight_dtype=["fp8_e4m3", "fp8_e5m2"], activation_dtype=["fp8_e4m3", "fp8_e5m2"])

    @classmethod
    def from_json_string(cls, json_string: str) -> "DynamicQuantConfig":
        cfg = json.loads(json_string)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "DynamicQuantConfig":
        weight_dtype = config_dict.get("weight_dtype", "fp8_e4m3")
        activation_dtype = config_dict.get("activation_dtype", "fp8_e4m3")
        white_list = config_dict.get("white_list", DEFAULT_WHITE_LIST)
        return cls(weight_dtype=weight_dtype, activation_dtype=activation_dtype, white_list=white_list)


@register_config(framework_name=FRAMEWORK_NAME, algo_name=STATIC_QUANT)
class StaticQuantConfig(BaseConfig):
    """Config class for JAX Static quantization.

    Static quantization applies quantization to weights offline and activations during runtime
    using pre-computed calibration data. This configuration supports various data types for
    flexible quantization strategies.

    Supported dtypes:
        - "fp8": 8-bit floating-point quantization (uses ml_dtypes.float8_e4m3 by default)

    FP8 formats available:
        - "fp8_e4m3": 4 exponent bits, 3 mantissa bits (default for "fp8")
        - "fp8_e5m2": 5 exponent bits, 2 mantissa bits
    """

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "weight_dtype",
        "activation_dtype",
    ]

    name = STATIC_QUANT

    def __init__(
        self,
        weight_dtype: str = "fp8_e4m3",
        activation_dtype: str = "fp8_e4m3",
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init Static quantization config.

        Args:
            weight_dtype (str): Data type for weights, default is "fp8_e4m3".
            activation_dtype (str): Data type for activations, default is "fp8_e4m3".
            white_list (list): A list of supported operators of this algorithm.
        """
        super().__init__(white_list=white_list)
        self.weight_dtype = weight_dtype
        self.activation_dtype = activation_dtype
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configs."""
        supported_configs = []
        static_config = StaticQuantConfig(
            weight_dtype=["fp8", "fp8_e4m3", "fp8_e5m2"],
            activation_dtype=["fp8", "fp8_e4m3", "fp8_e5m2"],
        )
        # Basic JAX operators for quantization
        operators = [keras.layers.Dense]
        supported_configs.append(OperatorConfig(config=static_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model) -> List[Tuple[str, Callable]]:
        """Get concrete node names for supported operators."""
        white_list = ["Dense", "EinsumDense", "MultiHeadAttention"]
        filter_result = []

        for layer in model._flatten_layers(recursive=True):
            if layer.__class__.__name__ in white_list:
                pair = (layer.name, layer.__class__.__name__)
                if pair not in filter_result:
                    filter_result.append(pair)

        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "StaticQuantConfig", List["StaticQuantConfig"]]:
        """Get a default config set for tuning."""
        return StaticQuantConfig(weight_dtype=["fp8_e4m3", "fp8_e5m2"], activation_dtype=["fp8_e4m3", "fp8_e5m2"])

    @classmethod
    def from_json_string(cls, json_string: str) -> "StaticQuantConfig":
        cfg = json.loads(json_string)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "StaticQuantConfig":
        weight_dtype = config_dict.get("weight_dtype", "fp8_e5m2")
        activation_dtype = config_dict.get("activation_dtype", "fp8_e5m2")
        white_list = config_dict.get("white_list", DEFAULT_WHITE_LIST)
        return cls(weight_dtype=weight_dtype, activation_dtype=activation_dtype, white_list=white_list)


register_supported_configs_for_fwk(fwk_name=FRAMEWORK_NAME)


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    """Get all registered configs for JAX framework."""
    registered_configs = config_registry.get_cls_configs()
    return registered_configs.get(FRAMEWORK_NAME, {})


def get_default_dynamic_config() -> DynamicQuantConfig:
    """Generate the default Dynamic quantization config.

    Returns:
        the default JAX Dynamic quantization config.
    """
    return DynamicQuantConfig()


def get_default_static_config() -> StaticQuantConfig:
    """Generate the default Static quantization config.

    Returns:
        the default JAX Static quantization config.
    """
    return StaticQuantConfig()
