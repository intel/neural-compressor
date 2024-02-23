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
"""Optimize layer config."""

from .quantize_layer_add import QuantizeLayerAdd
from .quantize_layer_bn import QuantizeLayerBatchNormalization


def config_quantizable_layers(model):
    """Configure the quantizable layers."""
    quantize_layer_mapping = {"Add": QuantizeLayerAdd, "BatchNormalization": QuantizeLayerBatchNormalization}

    for layer_class, quantize_layer in quantize_layer_mapping.items():
        quantize_layer_mapping[layer_class] = quantize_layer()

    for layer in model.layers:
        if layer.__class__.__name__ in quantize_layer_mapping:
            quantize_layer_mapping[layer.__class__.__name__](layer)
