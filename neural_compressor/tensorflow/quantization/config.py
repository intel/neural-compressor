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
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import tensorflow as tf

from neural_compressor.common import logger
from neural_compressor.common.base_config import (
    DEFAULT_WHITE_LIST,
    OP_NAME_OR_MODULE_TYPE,
    BaseConfig,
    config_registry,
    register_config,
    register_supported_configs_for_fwk,
)
from neural_compressor.common.utils import SMOOTH_QUANT, STATIC_QUANT
from neural_compressor.tensorflow.utils import DEFAULT_SQ_ALPHA_ARGS


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


@register_config(framework_name="keras", algo_name=STATIC_QUANT)
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

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "StaticQuantConfig", List["StaticQuantConfig"]]:
        # TODO fwk owner needs to update it.
        return StaticQuantConfig(weight_sym=[True, False])


register_supported_configs_for_fwk(fwk_name="keras")


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    """Get all registered configs for keras framework."""
    registered_configs = config_registry.get_cls_configs()
    return registered_configs.get("keras", {})


def get_default_static_quant_config() -> StaticQuantConfig:
    """Generate the default static quant config.

    Returns:
        the default keras config.
    """
    return StaticQuantConfig()


@register_config(framework_name="tensorflow", algo_name=SMOOTH_QUANT)
class SmoothQuantConfig(BaseConfig):
    """Config class for tf smooth quantization."""

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "alpha",
        "folding",
        "percentile",
        "op_types",
        "scales_per_op",
        "record_max_info",
        "weight_clip",
        "auto_alpha_args",
    ]
    name = SMOOTH_QUANT

    def __init__(
        self,
        alpha: float = 0.5,
        folding: bool = False,
        percentile: float = 99.999,
        op_types: list = ["MatMul", "Conv2D"],
        scales_per_op: bool = True,
        record_max_info: bool = False,
        weight_clip: bool = True,
        auto_alpha_args: Dict = DEFAULT_SQ_ALPHA_ARGS,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init smooth quantization config.

        Args:
            alpha (float or str): alpha value to balance the quantization difficulty of activation and weight.
            folding (bool): whether fold those foldable Mul which are inserted for smooth quant.
            percentile (float): percentile of calibration to remove outliers
            op_types (list): the op type to be smooth quantized.
            scales_per_op (bool): True, each op will have an individual scale, mainlyfor accuracy.
                                  False, ops with the same input will share a scale, mainly for performance.
            record_max_info (bool): whether record the max info in model for alpha tuning.
            weight_clip (bool): whether to clip weight when calculating scales; by default it is on.
            auto_alpha_args (dict): settings for alpha tuning.
        """
        super().__init__()
        self.alpha = alpha
        self.folding = folding
        self.percentile = percentile
        self.op_types = op_types
        self.scales_per_op = scales_per_op
        self.record_max_info = record_max_info
        self.weight_clip = weight_clip
        self.auto_alpha_args = auto_alpha_args
        self.white_list = white_list
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        smooth_quant_config = SmoothQuantConfig()
        operators = ["MatMul", "Conv2D"]
        supported_configs.append(OperatorConfig(config=smooth_quant_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model) -> List[Tuple[str, Callable]]:
        white_list = ["MatMul", "Conv2D"]
        filter_result = []
        for node in model.graph_def.node:
            if node.op in white_list:
                pair = (node.name, node.op)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "SmoothQuantConfig", List["SmoothQuantConfig"]]:
        # TODO fwk owner needs to update it.
        return SmoothQuantConfig(alpha=0.5)


SmoothQuantConfig.register_supported_configs()


def get_default_sq_config() -> SmoothQuantConfig:
    """Generate the default rtn config.

    Returns:
        the default smooth quant config.
    """
    return SmoothQuantConfig()
