#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
"""Initialize custom separable conv2d layers for Keras quantization."""

import json

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers

from neural_compressor.tensorflow.utils import version1_gte_version2

if version1_gte_version2(tf.__version__, "2.16.1"):
    from keras.src import ops
    from keras.src.layers.convolutional.base_separable_conv import BaseSeparableConv  # pylint: disable=E0401
elif version1_gte_version2(tf.__version__, "2.13.0"):
    from keras.src.layers.convolutional.base_separable_conv import SeparableConv  # pylint: disable=E0401
    from keras.src.utils import conv_utils  # pylint: disable=E0401
else:
    from keras.layers.convolutional.base_separable_conv import SeparableConv  # pylint: disable=E0401
    from keras.utils import conv_utils  # pylint: disable=E0401

if version1_gte_version2(tf.__version__, "2.16.1"):  # pragma: no cover

    class QSeparableConv2D(BaseSeparableConv):
        """The custom quantized SeparableConv2D layer."""

        def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            padding="valid",
            data_format=None,
            dilation_rate=(1, 1),
            depth_multiplier=1,
            activation=None,
            use_bias=True,
            depthwise_initializer="glorot_uniform",
            pointwise_initializer="glorot_uniform",
            bias_initializer="zeros",
            depthwise_regularizer=None,
            pointwise_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            pointwise_constraint=None,
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
            """Initialize custom quantized SeparableConv2D layer."""
            super().__init__(
                rank=2,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                depth_multiplier=depth_multiplier,
                activation=activations.get(activation),
                use_bias=use_bias,
                depthwise_initializer=initializers.get(depthwise_initializer),
                pointwise_initializer=initializers.get(pointwise_initializer),
                bias_initializer=initializers.get(bias_initializer),
                depthwise_regularizer=regularizers.get(depthwise_regularizer),
                pointwise_regularizer=regularizers.get(pointwise_regularizer),
                bias_regularizer=regularizers.get(bias_regularizer),
                activity_regularizer=regularizers.get(activity_regularizer),
                depthwise_constraint=constraints.get(depthwise_constraint),
                pointwise_constraint=constraints.get(pointwise_constraint),
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
            """The __call__ function of custom quantized SeparableConv2D layer."""
            if self.quant_status == "calib" and not isinstance(inputs, tf.keras.KerasTensor):
                if self.granularity == "per_tensor":
                    self.act_min_value = tf.math.reduce_min(inputs)
                    self.act_max_value = tf.math.reduce_max(inputs)
                else:
                    self.act_min_value = tf.math.reduce_min(inputs, axis=1)
                    self.act_max_value = tf.math.reduce_max(inputs, axis=1)
                depthwise_kernel = self.depthwise_kernel
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

                # (TODO) it's ugly that we can't get the point_wise min/max here
                depthwise_kernel, _, _ = tf.quantization.quantize(
                    self.depthwise_kernel, self.weight_min_value, self.weight_max_value, tf.qint8, axis=3, mode="SCALED"
                )
                depthwise_kernel = tf.quantization.dequantize(
                    depthwise_kernel,
                    self.weight_min_value,
                    self.weight_max_value,
                    axis=3,
                    mode="SCALED",
                )

            outputs = ops.separable_conv(
                inputs,
                depthwise_kernel,
                self.pointwise_kernel,
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                data_format=self.data_format,
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
            config = super(QSeparableConv2D, self).get_config()
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

    class QSeparableConv2D(SeparableConv):
        """The custom quantized SeparableConv2D layer."""

        def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            padding="valid",
            data_format=None,
            dilation_rate=(1, 1),
            depth_multiplier=1,
            activation=None,
            use_bias=True,
            depthwise_initializer="glorot_uniform",
            pointwise_initializer="glorot_uniform",
            bias_initializer="zeros",
            depthwise_regularizer=None,
            pointwise_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            pointwise_constraint=None,
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
            """Initialize custom quantized SeparableConv2D layer."""
            super().__init__(
                rank=2,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                depth_multiplier=depth_multiplier,
                activation=activations.get(activation),
                use_bias=use_bias,
                depthwise_initializer=initializers.get(depthwise_initializer),
                pointwise_initializer=initializers.get(pointwise_initializer),
                bias_initializer=initializers.get(bias_initializer),
                depthwise_regularizer=regularizers.get(depthwise_regularizer),
                pointwise_regularizer=regularizers.get(pointwise_regularizer),
                bias_regularizer=regularizers.get(bias_regularizer),
                activity_regularizer=regularizers.get(activity_regularizer),
                depthwise_constraint=constraints.get(depthwise_constraint),
                pointwise_constraint=constraints.get(pointwise_constraint),
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
            """The __call__ function of custom quantized SeparableConv2D layer."""
            if self.quant_status == "calib":
                if self.granularity == "per_tensor":
                    self.act_min_value = tf.math.reduce_min(inputs)
                    self.act_max_value = tf.math.reduce_max(inputs)
                else:
                    self.act_min_value = tf.math.reduce_min(inputs, axis=1)
                    self.act_max_value = tf.math.reduce_max(inputs, axis=1)
                depthwise_kernel = self.depthwise_kernel
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

                # (TODO) it's ugly that we can't get the point_wise min/max here
                depthwise_kernel, _, _ = tf.quantization.quantize(
                    self.depthwise_kernel, self.weight_min_value, self.weight_max_value, tf.qint8, axis=3, mode="SCALED"
                )
                depthwise_kernel = tf.quantization.dequantize(
                    depthwise_kernel,
                    self.weight_min_value,
                    self.weight_max_value,
                    axis=3,
                    mode="SCALED",
                )

            if self.data_format == "channels_last":
                strides = (1,) + self.strides + (1,)
            else:
                strides = (1, 1) + self.strides

            outputs = tf.compat.v1.nn.separable_conv2d(
                inputs,
                depthwise_kernel,
                self.pointwise_kernel,
                strides=strides,
                padding=self.padding.upper(),
                rate=self.dilation_rate,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4),
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
            config = super(QSeparableConv2D, self).get_config()
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


def initialize_int8_separable_conv2d(fp32_layer, q_config):
    """Initialize int8 separable conv2d."""
    kwargs = fp32_layer.get_config()

    param_list = [
        "name",
        "filters",
        "kernel_size",
        "strides",
        "padding",
        "data_format",
        "dilation_rate",
        "depth_multiplier",
        "activation",
        "use_bias",
        "depthwise_initializer",
        "bias_initializer",
        "pointwise_initializer",
        "depthwise_regularizer",
        "activity_regularizer",
        "bias_regularizer",
        "pointwise_regularizer",
        "depthwise_constraint",
        "bias_constraint",
        "pointwise_constraint",
    ]
    for p in param_list:  # pragma: no cover
        if p in kwargs:
            del kwargs[p]

    return QSeparableConv2D(
        name=fp32_layer.name,
        filters=fp32_layer.filters,
        kernel_size=fp32_layer.kernel_size,
        strides=fp32_layer.strides,
        padding=fp32_layer.padding,
        data_format=fp32_layer.data_format,
        dilation_rate=fp32_layer.dilation_rate,
        depth_multiplier=fp32_layer.depth_multiplier,
        activation=fp32_layer.activation,
        use_bias=fp32_layer.use_bias,
        depthwise_initializer=fp32_layer.depthwise_initializer,
        pointwise_initializer=fp32_layer.pointwise_initializer,
        bias_initializer=fp32_layer.bias_initializer,
        depthwise_regularizer=fp32_layer.depthwise_regularizer,
        pointwise_regularizer=fp32_layer.pointwise_regularizer,
        bias_regularizer=fp32_layer.bias_regularizer,
        activity_regularizer=fp32_layer.activity_regularizer,
        depthwise_constraint=fp32_layer.depthwise_constraint,
        pointwise_constraint=fp32_layer.pointwise_constraint,
        bias_constraint=fp32_layer.bias_constraint,
        quant_T=q_config["T"],
        granularity=q_config["granularity"],
        **kwargs
    )
