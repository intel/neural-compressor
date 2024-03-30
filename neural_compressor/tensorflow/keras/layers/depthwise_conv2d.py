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

import json

import tensorflow as tf
from tensorflow import quantization
from tensorflow.keras import activations, constraints, initializers, regularizers

from neural_compressor.tensorflow.utils import version1_gte_version2

if version1_gte_version2(tf.__version__, "2.13.0"):
    from keras.src.layers.convolutional.base_depthwise_conv import DepthwiseConv  # pylint: disable=E0401
    from keras.src.utils import conv_utils, tf_utils  # pylint: disable=E0401
else:
    from keras.layers.convolutional.base_depthwise_conv import DepthwiseConv  # pylint: disable=E0401
    from keras.utils import conv_utils, tf_utils  # pylint: disable=E0401


class QDepthwiseConv2D(DepthwiseConv):
    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        min_value=None,
        max_value=None,
        **kwargs
    ):
        super().__init__(
            2,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        depthwise_kernel_size = self.depthwise_kernel.shape[-1]

        if not self.min_value:
            self.min_value = [-10000] * depthwise_kernel_size
        if not self.max_value:
            self.max_value = [10000] * depthwise_kernel_size

        # add the Q/DQ here
        kernel, _, _ = quantization.quantize(
            self.depthwise_kernel, self.min_value, self.max_value, tf.qint8, axis=3, mode="SCALED"
        )
        kernel = quantization.dequantize(
            kernel,
            self.min_value,
            self.max_value,
            axis=3,
            mode="SCALED",
        )
        outputs = tf.keras.backend.depthwise_conv2d(
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
        return cls(**config)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == "channels_last":
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(
            rows,
            self.kernel_size[0],
            self.padding,
            self.strides[0],
            self.dilation_rate[0],
        )
        cols = conv_utils.conv_output_length(
            cols,
            self.kernel_size[1],
            self.padding,
            self.strides[1],
            self.dilation_rate[1],
        )
        if self.data_format == "channels_first":
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == "channels_last":
            return (input_shape[0], rows, cols, out_filters)


def initialize_int8_depthwise_conv2d(fp32_layer):
    kwargs = fp32_layer.get_config()
    q_name = fp32_layer.name

    if "name" in kwargs:
        del kwargs["name"]
    if "kernel_size" in kwargs:
        del kwargs["kernel_size"]
    if "strides" in kwargs:
        del kwargs["strides"]
    if "padding" in kwargs:
        del kwargs["padding"]
    if "depth_multiplier" in kwargs:
        del kwargs["depth_multiplier"]
    if "data_format" in kwargs:
        del kwargs["data_format"]
    if "dilation_rate" in kwargs:
        del kwargs["dilation_rate"]
    if "activation" in kwargs:
        del kwargs["activation"]
    if "use_bias" in kwargs:
        del kwargs["use_bias"]
    if "depthwise_initializer" in kwargs:
        del kwargs["depthwise_initializer"]
    if "bias_initializer" in kwargs:
        del kwargs["bias_initializer"]
    if "depthwise_regularizer" in kwargs:
        del kwargs["depthwise_regularizer"]
    if "activity_regularizer" in kwargs:
        del kwargs["activity_regularizer"]
    if "bias_regularizer" in kwargs:
        del kwargs["bias_regularizer"]
    if "depthwise_constraint" in kwargs:
        del kwargs["depthwise_constraint"]
    if "bias_constraint" in kwargs:
        del kwargs["bias_constraint"]
    if "min_value" in kwargs:
        del kwargs["min_value"]
    if "max_value" in kwargs:
        del kwargs["max_value"]

    return QDepthwiseConv2D(
        name=q_name,
        kernel_size=fp32_layer.kernel_size,
        strides=fp32_layer.strides,
        padding=fp32_layer.padding,
        depth_multiplier=fp32_layer.depth_multiplier,
        data_format=fp32_layer.data_format,
        dilation_rate=fp32_layer.dilation_rate,
        activation=fp32_layer.activation,
        use_bias=fp32_layer.use_bias,
        depthwise_initializer=fp32_layer.depthwise_initializer,
        bias_initializer=fp32_layer.bias_initializer,
        depthwise_regularizer=fp32_layer.depthwise_regularizer,
        bias_regularizer=fp32_layer.bias_regularizer,
        activity_regularizer=fp32_layer.activity_regularizer,
        depthwise_constraint=fp32_layer.depthwise_constraint,
        bias_constraint=fp32_layer.bias_constraint,
        min_value=fp32_layer.min_value,
        max_value=fp32_layer.max_value,
        **kwargs
    )
