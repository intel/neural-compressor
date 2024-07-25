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
"""Initialize custom layers for Keras quantization."""

from neural_compressor.tensorflow.keras.layers.conv2d import QConv2D
from neural_compressor.tensorflow.keras.layers.dense import QDense
from neural_compressor.tensorflow.keras.layers.depthwise_conv2d import QDepthwiseConv2D
from neural_compressor.tensorflow.keras.layers.pool2d import QAvgPool2D, QMaxPool2D
from neural_compressor.tensorflow.keras.layers.separable_conv2d import QSeparableConv2D
from neural_compressor.tensorflow.keras.layers.layer_initializer import layer_initializer_dict
