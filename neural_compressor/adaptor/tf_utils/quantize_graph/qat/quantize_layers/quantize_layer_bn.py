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
"""Quantize Layer BatchNormalization Class."""

from .quantize_layer_base import QuantizeLayerBase


class QuantizeLayerBatchNormalization(QuantizeLayerBase):  # pragma: no cover
    """The class for quantization of BatchNormalization."""

    def __init__(self):
        """Initialize QuantizeLayerBatchNormalization class."""
        super().__init__()

    def _quantizable_bn(self):
        """Check if the input layer meets criteria of quantization.

        Args:
            layer (tf.keras.layers.Layer): The input layer.

        Returns:
            quantizable (bool): If this layer should be quantized.
        """
        input_layer = self._find_input_layers(self.layer)
        assert len(input_layer) == 1, "BatchNormalization only has one input."
        input_layer_class = input_layer.__class__.__name__
        if "Conv" not in input_layer_class:
            return True

        return False

    def __call__(self, layer):
        """The main logic of QuantizeLayerBatchNormalization.

        Neural Compressor will enumerate all layers of the input model to check
        if there are any layer meeting the criteria. The chosen ones will be marked
        as quantizable by QuantizeConfig.

        Args:
            layer (tf.keras.layers.Layer): The keras layer to be estimated.
        """
        self.layer = layer
        if self._quantizable_bn():
            self.quantize_config.add_quantize_recipe({self.layer.name: {"quantize": True}})
