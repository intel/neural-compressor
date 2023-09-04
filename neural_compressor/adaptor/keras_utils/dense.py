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
from tensorflow.keras import activations, backend, constraints, initializers, regularizers
from tensorflow.keras.layers import Dense


class QDense(Dense):
    def __init__(
        self,
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
        min_value=-10000,
        max_value=10000,
        **kwargs
    ):
        super(QDense, self).__init__(
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
        self.min_value = json.loads(min_value)
        self.max_value = json.loads(max_value)

    def call(self, inputs):
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
