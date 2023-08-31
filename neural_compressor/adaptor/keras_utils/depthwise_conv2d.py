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

import json

import tensorflow as tf
from tensorflow import quantization
from tensorflow.keras import activations, constraints, initializers, regularizers

from neural_compressor.adaptor.tf_utils.util import version1_gte_version2

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
        min_value,
        max_value,
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
        self.min_value = json.loads(min_value)
        self.max_value = json.loads(max_value)

    def call(self, inputs):
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
