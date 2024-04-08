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
from tensorflow.keras import activations, backend, constraints, initializers, regularizers
from tensorflow.keras.layers import Dense


class QDense(Dense):
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
        min_value=None,
        max_value=None,
        **kwargs
    ):
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
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        kernel_size = self.kernel.shape[-1]

        if not self.min_value:
            self.min_value = [-10000] * kernel_size
        if not self.max_value:
            self.max_value = [10000] * kernel_size

        # add the Q/DQ here
        kernel, _, _ = quantization.quantize(
            self.kernel,
            self.min_value,
            self.max_value,
            tf.qint8,
            axis=1,
            mode="SCALED",
        )

        kernel = quantization.dequantize(
            kernel,
            self.min_value,
            self.max_value,
            axis=1,
            mode="SCALED",
        )
        outputs = tf.keras.backend.dot(inputs, kernel)

        if self.use_bias:
            outputs = tf.keras.backend.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


def initialize_int8_dense(fp32_layer):
    kwargs = fp32_layer.get_config()

    if "name" in kwargs:
        del kwargs["name"]
    if "units" in kwargs:
        del kwargs["units"]
    if "activation" in kwargs:
        del kwargs["activation"]
    if "use_bias" in kwargs:
        del kwargs["use_bias"]
    if "kernel_initializer" in kwargs:
        del kwargs["kernel_initializer"]
    if "bias_initializer" in kwargs:
        del kwargs["bias_initializer"]
    if "kernel_regularizer" in kwargs:
        del kwargs["kernel_regularizer"]
    if "activity_regularizer" in kwargs:
        del kwargs["activity_regularizer"]
    if "bias_regularizer" in kwargs:
        del kwargs["bias_regularizer"]
    if "kernel_constraint" in kwargs:
        del kwargs["kernel_constraint"]
    if "bias_constraint" in kwargs:
        del kwargs["bias_constraint"]
    if "min_value" in kwargs:
        del kwargs["min_value"]
    if "max_value" in kwargs:
        del kwargs["max_value"]

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
        min_value=fp32_layer.min_value,
        max_value=fp32_layer.max_value,
        **kwargs
    )

    return q_layer
