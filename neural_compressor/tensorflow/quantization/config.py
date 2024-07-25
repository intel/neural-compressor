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
"""Intel Neural Compressor TF quantization config API."""


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

FRAMEWORK_NAME = "tensorflow"

__all__ = [
    "StaticQuantConfig",
    "get_default_static_quant_config",
    "SmoothQuantConfig",
    "get_default_sq_config",
]


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


@register_config(framework_name=FRAMEWORK_NAME, algo_name=STATIC_QUANT)
class StaticQuantConfig(BaseConfig):
    """Config class for tf static quantization."""

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "weight_dtype",
        "weight_sym",
        "weight_granularity",
        "weight_algorithm",
        "act_dtype",
        "act_sym",
        "act_granularity",
        "act_algorithm",
    ]

    name = STATIC_QUANT

    def __init__(
        self,
        weight_dtype: str = "int8",
        weight_sym: bool = True,
        weight_granularity: str = "per_tensor",
        weight_algorithm: str = "minmax",
        act_dtype: str = "int8",
        act_sym: bool = True,
        act_granularity: str = "per_tensor",
        act_algorithm: str = "minmax",
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init static quantization config.

        Args:
            weight_dtype (str): Data type for weights, default is "int".
            weight_sym (bool): Indicates whether weights are symmetric, default is True.
            weight_granularity (str): Calculate tensor-wise scales or channel-wise scales for weights.
            weight_algorithm (str): Choose quantization algorithms for weights.
            act_dtype (str): Data type for activations, default is "int8".
            act_sym (bool): Indicates whether activations are symmetric, default is True.
            act_granularity (str): Calculate tensor-wise scales or channel-wise scales for activations.
            act_algorithm (str): Choose quantization algorithms for activations.
            white_list (list): A list of supported operators of this algorithm.
        """
        super().__init__(white_list=white_list)
        self.weight_dtype = weight_dtype
        self.weight_sym = weight_sym
        self.weight_granularity = weight_granularity
        self.weight_algorithm = weight_algorithm
        self.act_dtype = act_dtype
        self.act_sym = act_sym
        self.act_granularity = act_granularity
        self.act_algorithm = act_algorithm
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported config."""
        supported_configs = []
        static_quant_config = StaticQuantConfig(
            weight_dtype=["int8", "bf16", "fp32"],
            weight_sym=[True, False],
            weight_granularity=["per_tensor", "per_channel"],
            weight_algorithm=["minmax", "kl"],
            act_dtype=["int8", "bf16", "fp32"],
            act_sym=[True, False],
            act_granularity=["per_tensor"],
            act_algorithm=["minmax", "kl"],
        )
        operators = [
            tf.nn.conv2d,
            tf.raw_ops.FusedBatchNormV3,
            tf.nn.conv3d,
            tf.raw_ops.MatMul,
            tf.raw_ops.BatchMatMul,
            tf.raw_ops.BatchMatMulV2,
            tf.nn.depthwise_conv2d,
            tf.raw_ops.ConcatV2,
            tf.compat.v1.nn.fused_batch_norm,
            tf.nn.max_pool,
            tf.nn.avg_pool,
            tf.compat.v1.nn.conv2d_backprop_input,
            tf.raw_ops.Conv3DBackpropInputV2,
        ]
        supported_configs.append(OperatorConfig(config=static_quant_config, operators=operators))
        cls.supported_configs = supported_configs

    def get_model_info(self, model) -> List[Tuple[str, Callable]]:
        """Get concrete node names for supported operators."""
        white_list = [
            "Conv2D",
            "FusedBatchNormV3",
            "Conv3D",
            "_MklFusedInstanceNorm",
            "MatMul",
            "BatchMatMul",
            "BatchMatMulV2",
            "DepthwiseConv2dNative",
            "ConcatV2",
            "FusedBatchNorm",
            "FusedBatchNormV2",
            "MaxPool",
            "MaxPool3D",
            "AvgPool",
            "Conv2DBackpropInput",
            "Conv3DBackpropInputV2",
        ]
        for key in self._local_config.keys():
            if key in white_list:
                white_list.remove(key)
        filter_result = []
        for node in model.graph_def.node:
            if node.op in white_list:
                pair = (node.name, node.op)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "StaticQuantConfig", List["StaticQuantConfig"]]:
        """Get a default config set for tuning."""
        return StaticQuantConfig(
            weight_dtype=["int8", "fp32"],
            weight_sym=[True, False],
            weight_granularity=["per_tensor", "per_channel"],
            weight_algorithm=["minmax", "kl"],
            act_dtype=["int8", "fp32"],
            act_sym=[True, False],
            act_granularity=["per_tensor"],
            act_algorithm=["minmax", "kl"],
        )


register_supported_configs_for_fwk(fwk_name="keras")


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    """Get all registered configs for keras framework."""
    registered_configs = config_registry.get_cls_configs()
    return registered_configs.get(FRAMEWORK_NAME, {})


def get_default_static_quant_config() -> StaticQuantConfig:
    """Generate the default static quant config.

    Returns:
        the default tf config.
    """
    return StaticQuantConfig()


@register_config(framework_name=FRAMEWORK_NAME, algo_name=SMOOTH_QUANT)
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
        """Init RTN weight-only quantization config.

        Args:
            alpha (float): Value to balance input and weight quantization error, between 0 and 1, default is 0.5.
            folding (bool): Whether to fold mul into the previous layer, default is False.
            percentile (float): percentile of calibration to remove outliers, default is 99.99.
            op_types (list): The op types whose input tensor will be dumped, default is ["MatMul", "Conv2D"].
            scales_per_op (bool): Whether to set individual scale for every op, default is True.
            record_max_info (bool): whether record the max info in model for alpha tuning, default is False.
            weight_clip: Whether to clip weight when calculating scales, default is True.
            auto_alpha_args(dict) : Hyperparameters used to set the alpha search space in SQ auto-tuning,
                                    by default the search space is 0.0-1.0 with step_size 0.1.
            white_list (list): A list of supported operators of this algorithm.
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
        """Register supported configs."""
        supported_configs = []
        smooth_quant_config = SmoothQuantConfig()
        operators = ["MatMul", "Conv2D"]
        supported_configs.append(OperatorConfig(config=smooth_quant_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model) -> List[Tuple[str, Callable]]:
        """Get concrete node names for supported operators."""
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
        """Get a default config set for tuning."""
        return SmoothQuantConfig(alpha=0.5)


SmoothQuantConfig.register_supported_configs()


def get_default_sq_config() -> SmoothQuantConfig:
    """Generate the default rtn config.

    Returns:
        the default smooth quant config.
    """
    return SmoothQuantConfig()
