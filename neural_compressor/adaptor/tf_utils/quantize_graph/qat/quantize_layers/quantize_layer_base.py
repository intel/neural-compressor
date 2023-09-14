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
"""QuantizeLayer Base Class."""

from ..quantize_config import global_config


class QuantizeLayerBase:  # pragma: no cover
    """QuantizeLayer Base Class."""

    def __init__(self):
        """Initialize QuantizeLayerBase class."""
        self.quantize_patterns = []
        assert "quantize_config" in global_config, "QuantizeConfig is not correctly created."
        self.quantize_config = global_config["quantize_config"]

    def _find_input_layers(self, layer):
        """Find all inputs of a specific layer.

        Args:
            layer (tf.keras.layers.Layer): The target keras layer that this method
                                           is to find its input layers.

        Returns:
            input_layers (list): List of input layers found by this method.
        """
        input_layers = []
        if isinstance(layer.input, list):
            for input_tensor in layer.input:
                input_layer = input_tensor._keras_history.layer
                input_layers.append(input_layer)
        else:
            input_layer = layer.input._keras_history.layer
            input_layers.append(input_layer)
        return input_layers

    def _find_patterns(self, layer):
        """Checks if the input layer can satisfy the patterns.

        Args:
            layer (tf.keras.layers.Layer): The input keras layer that this method
                                           is to find patterns.

        Returns:
            valid_patterns (bool): If the input layer can satisfy any pattern.
        """
        if not self.quantize_patterns:
            return False

        for quantize_pattern in self.quantize_patterns:
            index = len(quantize_pattern) - 2
            previous_layer = layer
            while index >= 0:
                previous_layer = self._find_input_layers(previous_layer)
                if quantize_pattern[index] not in previous_layer.__class__.__name__:
                    break
                index -= 1
            if index == -1:
                return True

        return False

    def __call__(self, layer):
        """The main logic of QuantizeLayerBase.

        Neural Compressor will enumerate all layers of the input model to check
        if there are any layer meeting the criteria. The chosen ones will be marked
        as quantizable by QuantizeConfig.

        Args:
            layer (tf.keras.layers.Layer): The keras layer to be estimated.
        """
        raise NotImplementedError()
