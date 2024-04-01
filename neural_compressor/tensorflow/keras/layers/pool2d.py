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
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D


class QAvgPool2D(AveragePooling2D):
    def __init__(
        self,
        name,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        min_value=-10000,
        max_value=10000,
        **kwargs
    ):
        super(QAvgPool2D, self).__init__(
            name=name, pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, **kwargs
        )
        self.min_value = min_value
        self.max_value = max_value


class QMaxPool2D(MaxPooling2D):
    def __init__(
        self,
        name,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        min_value=-10000,
        max_value=10000,
        **kwargs
    ):
        super(QMaxPool2D, self).__init__(
            name=name, pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, **kwargs
        )
        self.min_value = min_value
        self.max_value = max_value


def initialize_int8_avgpool(fp32_layer):
    kwargs = fp32_layer.get_config()

    if "name" in kwargs:
        del kwargs["name"]
    if "pool_size" in kwargs:
        del kwargs["pool_size"]
    if "strides" in kwargs:
        del kwargs["strides"]
    if "padding" in kwargs:
        del kwargs["padding"]
    if "data_format" in kwargs:
        del kwargs["data_format"]
    if "min_value" in kwargs:
        del kwargs["min_value"]
    if "max_value" in kwargs:
        del kwargs["max_value"]

    q_layer = QAvgPool2D(
        name=fp32_layer.name,
        pool_size=fp32_layer.pool_size,
        strides=fp32_layer.strides,
        padding=fp32_layer.padding,
        data_format=fp32_layer.data_format,
        min_value=fp32_layer.min_value,
        max_value=fp32_layer.max_value,
        **kwargs
    )

    return q_layer


def initialize_int8_maxpool(fp32_layer):
    kwargs = fp32_layer.get_config()

    if "name" in kwargs:
        del kwargs["name"]
    if "pool_size" in kwargs:
        del kwargs["pool_size"]
    if "strides" in kwargs:
        del kwargs["strides"]
    if "padding" in kwargs:
        del kwargs["padding"]
    if "data_format" in kwargs:
        del kwargs["data_format"]
    if "min_value" in kwargs:
        del kwargs["min_value"]
    if "max_value" in kwargs:
        del kwargs["max_value"]

    q_layer = QMaxPool2D(
        name=fp32_layer.name,
        pool_size=fp32_layer.pool_size,
        strides=fp32_layer.strides,
        padding=fp32_layer.padding,
        data_format=fp32_layer.data_format,
        min_value=fp32_layer.min_value,
        max_value=fp32_layer.max_value,
        **kwargs
    )

    return q_layer
