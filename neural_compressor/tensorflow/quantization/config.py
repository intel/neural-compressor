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

from neural_compressor.common.base_config import (
    BaseConfig,
    config_registry,
    register_config,
    register_supported_configs_for_fwk,
)
from neural_compressor.common.utils import DEFAULT_WHITE_LIST, OP_NAME_OR_MODULE_TYPE, STATIC_QUANT

FRAMEWORK_NAME = "tensorflow"


<<<<<<< HEAD
class Backend(Enum):
    DEFAULT = "tensorflow"


=======
>>>>>>> 02233fb484996bdce5a8b73eec623ed3b5fd1e47
class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


<<<<<<< HEAD
# mapping the torch module type and functional operation type to string representations
operator2str = {
    tf.nn.conv2d: "Conv2D",
    tf.raw_ops.FusedBatchNormV3: "FusedBatchNormV3",
    tf.nn.conv3d: "Conv3D",
    tf.raw_ops.MatMul: "MatMul",
    tf.raw_ops.BatchMatMul: "BatchMatMul",
    tf.raw_ops.BatchMatMulV2: "BatchMatMulV2",
    tf.nn.depthwise_conv2d: "DepthwiseConv2dNative",
    tf.raw_ops.ConcatV2: "ConcatV2",
    tf.compat.v1.nn.fused_batch_norm: "FusedBatchNorm",
    tf.nn.max_pool: "MaxPool",
    tf.nn.avg_pool: "AvgPool",
    tf.compat.v1.nn.conv2d_backprop_input: "Conv2DBackpropInput",
    tf.raw_ops.Conv3DBackpropInputV2: "Conv3DBackpropInputV2",
}

# Mapping from string representations to their corresponding torch operation/module type
str2operator = {
    "Conv2D": tf.nn.conv2d,
    "FusedBatchNormV3": tf.raw_ops.FusedBatchNormV3,
    "Conv3D": tf.nn.conv3d,
    "MatMul": tf.raw_ops.MatMul,
    "BatchMatMul": tf.raw_ops.BatchMatMul,
    "BatchMatMulV2": tf.raw_ops.BatchMatMulV2,
    "DepthwiseConv2dNative": tf.nn.depthwise_conv2d,
    "ConcatV2": tf.raw_ops.ConcatV2,
    "FusedBatchNorm": tf.compat.v1.nn.fused_batch_norm,
    "MaxPool": tf.nn.max_pool,
    "AvgPool": tf.nn.avg_pool,
    "Conv2DBackpropInput": tf.compat.v1.nn.conv2d_backprop_input,
    "Conv3DBackpropInputV2": tf.raw_ops.Conv3DBackpropInputV2,
}


=======
>>>>>>> 02233fb484996bdce5a8b73eec623ed3b5fd1e47
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
        supported_configs = []
        static_quant_config = StaticQuantConfig(
            weight_dtype=["int8", "fp32"],
            weight_sym=[True, False],
            weight_granularity=["per_tensor", "per_channel"],
            weight_algorithm=["minmax", "kl"],
            act_dtype=["int8", "fp32"],
            act_sym=[True, False],
            act_granularity=["per_tensor", "per_channel"],
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

    @classmethod
    def get_config_set_for_tuning(
        cls,
    ) -> Union[None, "StaticQuantConfig", List["StaticQuantConfig"]]:  # pragma: no cover
        # TODO fwk owner needs to update it.
        return StaticQuantConfig(weight_sym=[True, False])


register_supported_configs_for_fwk(fwk_name=FRAMEWORK_NAME)


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    """Get all registered configs for keras framework."""
    registered_configs = config_registry.get_cls_configs()
    return registered_configs.get(FRAMEWORK_NAME, {})


def parse_tf_config_from_dict(config_dict: Dict) -> BaseConfig:
    """Generate a BaseConfig instance from a dict."""
    tf_registered_configs = get_all_registered_configs()
    for key, val in config_dict.items():
        if key in tf_registered_configs:
            config = tf_registered_configs[key].from_dict(val)
            return config


def get_default_static_quant_config() -> StaticQuantConfig:
    """Generate the default static quant config.

    Returns:
        the default tf config.
    """
    return StaticQuantConfig()
