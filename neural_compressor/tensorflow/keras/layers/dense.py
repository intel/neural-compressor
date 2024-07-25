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
"""Initialize custom dense layers for Keras quantization."""

import json

import tensorflow as tf
from tensorflow.keras import activations, backend, constraints, initializers, regularizers
from tensorflow.keras.layers import Dense

from neural_compressor.tensorflow.utils import version1_gte_version2


class QDense(Dense):
    """The custom quantized Dense layer."""

    def __init__(
        self,
        name,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        act_min_value=None,
        act_max_value=None,
        weight_min_value=None,
        weight_max_value=None,
        granularity="per_tensor",
        quant_status="calib",
        quant_mode="SCALED",
        quant_T="s8",
        quant_round_mode="HALF_AWAY_FROM_ZERO",
        quant_narrow_range=False,
        quant_axis=None,
        **kwargs
    ):
        """Initialize custom quantized Dense layer."""
        super(QDense, self).__init__(
            name=name,
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        T_map = {"s8": tf.qint8, "u8": tf.quint8}
        self.weight_min_value = weight_min_value
        self.weight_max_value = weight_max_value
        self.act_min_value = act_min_value
        self.act_max_value = act_max_value
        self.granularity = granularity
        self.quant_status = quant_status
        self.quant_mode = quant_mode
        self.quant_T = T_map[quant_T]
        self.quant_round_mode = quant_round_mode
        self.quant_narrow_range = quant_narrow_range
        self.quant_axis = quant_axis

    def call(self, inputs):
        """The __call__ function of custom quantized Dense layer."""
        if self.quant_status == "calib" and not (
            version1_gte_version2(tf.__version__, "2.16.1") and isinstance(inputs, tf.keras.KerasTensor)
        ):
            if self.granularity == "per_tensor":
                self.act_min_value = tf.math.reduce_min(inputs)
                self.act_max_value = tf.math.reduce_max(inputs)
            else:
                self.act_min_value = tf.math.reduce_min(inputs, axis=1)
                self.act_max_value = tf.math.reduce_max(inputs, axis=1)
            kernel = self.kernel
        elif self.quant_status == "quantize":
            assert self.act_min_value is not None, "Invalid activation min-max values, please check calibration process"
            inputs, _, _ = tf.quantization.quantize(
                inputs,
                self.act_min_value,
                self.act_max_value,
                self.quant_T,
                mode=self.quant_mode,
                round_mode=self.quant_round_mode,
                narrow_range=self.quant_narrow_range,
                axis=self.quant_axis,
            )
            inputs = tf.quantization.dequantize(
                inputs,
                self.act_min_value,
                self.act_max_value,
                mode=self.quant_mode,
                narrow_range=self.quant_narrow_range,
                axis=self.quant_axis,
            )

            kernel_size = self.kernel.shape[-1]

            if not self.weight_min_value:
                self.weight_min_value = [-10000] * kernel_size
            if not self.weight_max_value:
                self.weight_max_value = [10000] * kernel_size

            # add the Q/DQ here
            kernel, _, _ = tf.quantization.quantize(
                self.kernel,
                self.weight_min_value,
                self.weight_max_value,
                tf.qint8,
                axis=1,
                mode="SCALED",
            )
            kernel = tf.quantization.dequantize(
                kernel,
                self.weight_min_value,
                self.weight_max_value,
                axis=1,
                mode="SCALED",
            )

        outputs = tf.keras.backend.dot(inputs, kernel)

        if self.use_bias:
            outputs = tf.keras.backend.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    @classmethod
    def from_config(cls, config):
        """Deserialize this class from a config dict."""
        return cls(**config)

    def get_config(self):
        """Serialize this class to a config dict."""
        config = super(QDense, self).get_config()
        config.update(
            {
                "act_min_value": self.act_min_value,
                "act_max_value": self.act_max_value,
                "weight_min_value": self.weight_min_value,
                "weight_max_value": self.weight_max_value,
                "granularity": self.granularity,
                "quant_status": self.quant_status,
                "quant_mode": self.quant_mode,
                "quant_T": "s8" if self.quant_T == tf.qint8 else "u8",
                "quant_round_mode": self.quant_round_mode,
                "quant_narrow_range": self.quant_narrow_range,
                "quant_axis": self.quant_axis,
            }
        )

        return config


def initialize_int8_dense(fp32_layer, q_config):
    """Initialize int8 dense."""
    kwargs = fp32_layer.get_config()

    param_list = [
        "name",
        "units",
        "activation",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "activity_regularizer",
        "bias_regularizer",
        "kernel_constraint",
        "bias_constraint",
    ]
    for p in param_list:  # pragma: no cover
        if p in kwargs:
            del kwargs[p]

    q_layer = QDense(
        name=fp32_layer.name,
        units=fp32_layer.units,
        activation=fp32_layer.activation,
        use_bias=fp32_layer.use_bias,
        kernel_initializer=fp32_layer.kernel_initializer,
        bias_initializer=fp32_layer.bias_initializer,
        kernel_regularizer=fp32_layer.kernel_regularizer,
        bias_regularizer=fp32_layer.bias_regularizer,
        activity_regularizer=fp32_layer.activity_regularizer,
        kernel_constraint=fp32_layer.kernel_constraint,
        bias_constraint=fp32_layer.bias_constraint,
        quant_T=q_config["T"],
        granularity=q_config["granularity"],
        **kwargs
    )

    return q_layer
