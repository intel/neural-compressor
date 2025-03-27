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
"""Initialize layer initializer functions."""

from neural_compressor.tensorflow.keras.layers.conv2d import initialize_int8_conv2d
from neural_compressor.tensorflow.keras.layers.dense import initialize_int8_dense
from neural_compressor.tensorflow.keras.layers.depthwise_conv2d import initialize_int8_depthwise_conv2d
from neural_compressor.tensorflow.keras.layers.pool2d import initialize_int8_avgpool, initialize_int8_maxpool
from neural_compressor.tensorflow.keras.layers.separable_conv2d import initialize_int8_separable_conv2d

layer_initializer_dict = {
    "QAvgPool2D": initialize_int8_avgpool,
    "QAveragePooling2D": initialize_int8_avgpool,
    "QMaxPool2D": initialize_int8_maxpool,
    "QMaxPooling2D": initialize_int8_maxpool,
    "QSeparableConv2D": initialize_int8_separable_conv2d,
    "QDepthwiseConv2D": initialize_int8_depthwise_conv2d,
    "QConv2D": initialize_int8_conv2d,
    "QDense": initialize_int8_dense,
}
