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
"""Initialize custom conv2d layers for Keras quantization."""

import json

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers

from neural_compressor.tensorflow.utils import version1_gte_version2

if version1_gte_version2(tf.__version__, "2.16.1"):
    from keras import ops
    from keras.src.layers.convolutional.base_conv import BaseConv  # pylint: disable=E0401
elif version1_gte_version2(tf.__version__, "2.13.0"):
    from keras.src.layers.convolutional.base_conv import Conv  # pylint: disable=E0401
else:
    from keras.layers.convolutional.base_conv import Conv  # pylint: disable=E0401

if version1_gte_version2(tf.__version__, "2.16.1"):  # pragma: no cover

    class QConv2D(BaseConv):
        """The custom quantized Conv2D layer."""

        def __init__(
            self,
            name,
            filters,
            kernel_size,
            strides=(1, 1),
            padding="valid",
            data_format=None,
            dilation_rate=(1, 1),
            groups=1,
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
            """Initialize custom quantized Conv2D layer."""
            super(QConv2D, self).__init__(
                name=name,
                rank=2,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                groups=groups,
                activation=activations.get(activation),
                use_bias=use_bias,
                kernel_initializer=initializers.get(kernel_initializer),
                bias_initializer=initializers.get(bias_initializer),
                kernel_regularizer=regularizers.get(kernel_regularizer),
                bias_regularizer=regularizers.get(bias_regularizer),
                activity_regularizer=regularizers.get(activity_regularizer),
                kernel_constraint=constraints.get(kernel_constraint),
                bias_constraint=constraints.get(bias_constraint),
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
            """The __call__ function of custom quantized Conv2D layer."""
            if self.quant_status == "calib" and not isinstance(inputs, tf.keras.KerasTensor):
                if self.granularity == "per_tensor":
                    self.act_min_value = tf.math.reduce_min(inputs)
                    self.act_max_value = tf.math.reduce_max(inputs)
                else:
                    self.act_min_value = tf.math.reduce_min(inputs, axis=1)
                    self.act_max_value = tf.math.reduce_max(inputs, axis=1)
                kernel = self.kernel
            elif self.quant_status == "quantize":
                assert (
                    self.act_min_value is not None
                ), "Invalid activation min-max values, please check calibration process"
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
                    self.kernel, self.weight_min_value, self.weight_max_value, tf.qint8, axis=3, mode="SCALED"
                )
                kernel = tf.quantization.dequantize(
                    kernel,
                    self.weight_min_value,
                    self.weight_max_value,
                    axis=3,
                    mode="SCALED",
                )

            outputs = self.convolution_op(
                inputs,
                kernel,
            )
            if self.use_bias:
                if self.data_format == "channels_last":
                    bias_shape = (1,) * (self.rank + 1) + (self.filters,)
                else:
                    bias_shape = (1, self.filters) + (1,) * self.rank
                bias = ops.reshape(self.bias, bias_shape)
                outputs += bias

            if self.activation is not None:
                return self.activation(outputs)
            return outputs

        @classmethod
        def from_config(cls, config):
            """Deserialize this class from a config dict."""
            return cls(**config)

        def get_config(self):
            """Serialize this class to a config dict."""
            config = super(QConv2D, self).get_config()
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

else:

    class QConv2D(Conv):
        """The custom quantized Conv2D layer."""

        def __init__(
            self,
            name,
            filters,
            kernel_size,
            strides=(1, 1),
            padding="valid",
            data_format=None,
            dilation_rate=(1, 1),
            groups=1,
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
            """Initialize custom quantized Conv2D layer."""
            super(QConv2D, self).__init__(
                name=name,
                rank=2,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                groups=groups,
                activation=activations.get(activation),
                use_bias=use_bias,
                kernel_initializer=initializers.get(kernel_initializer),
                bias_initializer=initializers.get(bias_initializer),
                kernel_regularizer=regularizers.get(kernel_regularizer),
                bias_regularizer=regularizers.get(bias_regularizer),
                activity_regularizer=regularizers.get(activity_regularizer),
                kernel_constraint=constraints.get(kernel_constraint),
                bias_constraint=constraints.get(bias_constraint),
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
            """The __call__ function of custom quantized Conv2D layer."""
            if self.quant_status == "calib":
                if self.granularity == "per_tensor":
                    self.act_min_value = tf.math.reduce_min(inputs)
                    self.act_max_value = tf.math.reduce_max(inputs)
                else:
                    self.act_min_value = tf.math.reduce_min(inputs, axis=1)
                    self.act_max_value = tf.math.reduce_max(inputs, axis=1)
                kernel = self.kernel
            elif self.quant_status == "quantize":
                assert (
                    self.act_min_value is not None
                ), "Invalid activation min-max values, please check calibration process"
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
                    self.kernel, self.weight_min_value, self.weight_max_value, tf.qint8, axis=3, mode="SCALED"
                )
                kernel = tf.quantization.dequantize(
                    kernel,
                    self.weight_min_value,
                    self.weight_max_value,
                    axis=3,
                    mode="SCALED",
                )
            outputs = tf.keras.backend.conv2d(
                inputs,
                kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

            if self.use_bias:
                outputs = tf.keras.backend.bias_add(outputs, self.bias, data_format=self.data_format)

            if self.activation is not None:
                return self.activation(outputs)

            return outputs

        @classmethod
        def from_config(cls, config):
            """Deserialize this class from a config dict."""
            return cls(**config)

        def get_config(self):
            """Serialize this class to a config dict."""
            config = super(QConv2D, self).get_config()
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


def initialize_int8_conv2d(fp32_layer, q_config):
    """Initialize int8 conv2d."""
    kwargs = fp32_layer.get_config()

    param_list = [
        "name",
        "filters",
        "kernel_size",
        "strides",
        "padding",
        "data_format",
        "dilation_rate",
        "groups",
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

    return QConv2D(
        name=fp32_layer.name,
        filters=fp32_layer.filters,
        kernel_size=fp32_layer.kernel_size,
        strides=fp32_layer.strides,
        padding=fp32_layer.padding,
        data_format=fp32_layer.data_format,
        dilation_rate=fp32_layer.dilation_rate,
        groups=fp32_layer.groups,
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
