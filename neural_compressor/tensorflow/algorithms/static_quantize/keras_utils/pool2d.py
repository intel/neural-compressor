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
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D


class QAvgPool2D(AveragePooling2D):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        min_value=-10000,
        max_value=10000,
        **kwargs
    ):
        super(QAvgPool2D, self).__init__(
            pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, **kwargs
        )
        self.min_value = json.loads(min_value)
        self.max_value = json.loads(max_value)


class QMaxPool2D(MaxPooling2D):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        min_value=-10000,
        max_value=10000,
        **kwargs
    ):
        super(QMaxPool2D, self).__init__(
            pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, **kwargs
        )
        self.min_value = json.loads(min_value)
        self.max_value = json.loads(max_value)
