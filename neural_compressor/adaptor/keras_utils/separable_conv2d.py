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
    from keras.src.layers.convolutional.base_separable_conv import SeparableConv  # pylint: disable=E0401
    from keras.src.utils import conv_utils  # pylint: disable=E0401
else:
    from keras.layers.convolutional.base_separable_conv import SeparableConv  # pylint: disable=E0401
    from keras.utils import conv_utils  # pylint: disable=E0401


class QSeparableConv2D(SeparableConv):
    def __init__(
        self,
        filters,
        kernel_size,
        min_value,
        max_value,
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
        **kwargs
    ):
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

        self.min_value = json.loads(min_value)
        self.max_value = json.loads(max_value)

    def call(self, inputs):
        if self.data_format == "channels_last":
            strides = (1,) + self.strides + (1,)
        else:
            strides = (1, 1) + self.strides
        # (TODO) it's ugly that we can't get the point_wise min/max here
        depthwise_kernel, _, _ = quantization.quantize(
            self.depthwise_kernel, self.min_value, self.max_value, tf.qint8, axis=3, mode="SCALED"
        )
        depthwise_kernel = quantization.dequantize(
            depthwise_kernel,
            self.min_value,
            self.max_value,
            axis=3,
            mode="SCALED",
        )

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
        return cls(**config)
