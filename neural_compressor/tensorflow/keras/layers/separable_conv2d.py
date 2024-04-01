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
    from keras.src.layers.convolutional.base_separable_conv import SeparableConv  # pylint: disable=E0401
    from keras.src.utils import conv_utils  # pylint: disable=E0401
else:
    from keras.layers.convolutional.base_separable_conv import SeparableConv  # pylint: disable=E0401
    from keras.utils import conv_utils  # pylint: disable=E0401


class QSeparableConv2D(SeparableConv):
    def __init__(
        self,
        name,
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
        min_value=None,
        max_value=None,
        **kwargs
    ):
        super().__init__(
            name=name,
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

        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        depthwise_kernel_size = self.depthwise_kernel.shape[-1]

        if not self.min_value:
            self.min_value = [-10000] * depthwise_kernel_size
        if not self.max_value:
            self.max_value = [10000] * depthwise_kernel_size

        # TODO it's ugly that we can't get the point_wise min/max here
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
        return cls(**config)


def initialize_int8_separable_conv2d(fp32_layer):
    kwargs = fp32_layer.get_config()

    if "name" in kwargs:
        del kwargs["name"]
    if "filters" in kwargs:
        del kwargs["filters"]
    if "kernel_size" in kwargs:
        del kwargs["kernel_size"]
    if "strides" in kwargs:
        del kwargs["strides"]
    if "padding" in kwargs:
        del kwargs["padding"]
    if "data_format" in kwargs:
        del kwargs["data_format"]
    if "dilation_rate" in kwargs:
        del kwargs["dilation_rate"]
    if "depth_multiplier" in kwargs:
        del kwargs["depth_multiplier"]
    if "activation" in kwargs:
        del kwargs["activation"]
    if "use_bias" in kwargs:
        del kwargs["use_bias"]
    if "depthwise_initializer" in kwargs:
        del kwargs["depthwise_initializer"]
    if "pointwise_initializer" in kwargs:
        del kwargs["pointwise_initializer"]
    if "bias_initializer" in kwargs:
        del kwargs["bias_initializer"]
    if "depthwise_regularizer" in kwargs:
        del kwargs["depthwise_regularizer"]
    if "pointwise_regularizer" in kwargs:
        del kwargs["pointwise_regularizer"]
    if "activity_regularizer" in kwargs:
        del kwargs["activity_regularizer"]
    if "bias_regularizer" in kwargs:
        del kwargs["bias_regularizer"]
    if "depthwise_constraint" in kwargs:
        del kwargs["depthwise_constraint"]
    if "pointwise_constraint" in kwargs:
        del kwargs["pointwise_constraint"]
    if "bias_constraint" in kwargs:
        del kwargs["bias_constraint"]
    if "min_value" in kwargs:
        del kwargs["min_value"]
    if "max_value" in kwargs:
        del kwargs["max_value"]

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
        min_value=fp32_layer.min_value,
        max_value=fp32_layer.max_value,
        **kwargs
    )
