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
"""Quantization Add Layer Class."""

import logging

from .quantize_layer_base import QuantizeLayerBase

logger = logging.getLogger("neural_compressor")


class QuantizeLayerAdd(QuantizeLayerBase):  # pragma: no cover
    """The class for quantization of Add."""

    def __init__(self):
        """Initialize QuantizeLayerAdd class."""
        self.quantize_patterns = [
            ["Conv", "BatchNorm", "Add"],
            ["Conv", "BatchNorm", "Activation", "Add"],
            ["Conv", "BatchNorm", "Activation", "Dropout", "Add"],
        ]

        super().__init__()

    def _quantizable_add(self):
        """Check if the input layer meets criteria of quantization.

        Args:
            layer (tf.keras.layers.Layer): The input layer.

        Returns:
            quantizable (bool): If this layer should be quantized.
        """
        input_layer = self._find_input_layers(self.layer)
        if len(input_layer) == 1:
            logger.warning(
                "The layer 'Add' should have more than one input. "
                "You input a model with layer {} which has only one input".format(self.layer.name)
            )
            return False

        return True

    def __call__(self, layer):
        """The main logic of QuantizeLayerAdd.

        Neural Compressor will enumerate all layers of the input model to check
        if there are any layer meeting the criteria. The chosen ones will be marked
        as quantizable by QuantizeConfig.

        Args:
            layer (tf.keras.layers.Layer): The keras layer to be estimated.
        """
        self.layer = layer
        if self._quantizable_add():
            input_layers = self._find_input_layers(self.layer)
            fused_conv_index = None
            for i, input_layer in enumerate(input_layers):
                # Check that the input is a Conv pattern
                if "Conv" in input_layer.__class__.__name__ or self._find_patterns(input_layer):
                    if hasattr(input_layer, "outbound_nodes") and len(getattr(input_layer, "outbound_nodes")) == 1:
                        fused_conv_index = i
                        break

            input_indexes = [i for i in range(0, len(input_layers))]
            if fused_conv_index:
                del input_indexes[fused_conv_index]

            self.quantize_config.add_quantize_recipe({self.layer.name: {"quantize": True, "index": input_indexes}})
