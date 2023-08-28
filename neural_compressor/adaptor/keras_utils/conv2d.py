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
    from keras.src.layers.convolutional.base_conv import Conv  # pylint: disable=E0401
else:
    from keras.layers.convolutional.base_conv import Conv  # pylint: disable=E0401


class QConv2D(Conv):
    def __init__(
        self,
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
        min_value=-10000,
        max_value=10000,
        **kwargs
    ):
        super(QConv2D, self).__init__(
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
        self.min_value = json.loads(min_value)
        self.max_value = json.loads(max_value)

    def call(self, inputs):
        # add the Q/DQ here
        kernel, _, _ = quantization.quantize(
            self.kernel, self.min_value, self.max_value, tf.qint8, axis=3, mode="SCALED"
        )
        kernel = quantization.dequantize(
            kernel,
            self.min_value,
            self.max_value,
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
        return cls(**config)
