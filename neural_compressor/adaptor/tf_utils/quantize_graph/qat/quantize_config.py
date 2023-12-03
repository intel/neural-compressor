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
"""QAT Quantize Config Class."""

import logging

global_config = {}
logger = logging.getLogger("neural_compressor")


class QuantizeConfig:
    """Class for building custom quantize config.

    There should be only one QuantizeConfig instance for global setting.
    """

    def __new__(cls):
        """Created a QuantizeConfig instance and add it to the global_config dict.

        Returns:
            instance (QuantizeConfig) : The created QuantizeConfig instance.
        """
        instance = super().__new__(cls)
        global_config["quantize_config"] = instance
        return instance

    def __init__(self):
        """Initialize QuantizeConfig instance."""
        self.quantize_recipe = {}
        self.model_name = None

    def add_quantize_recipe(self, quantize_recipe):  # pragma: no cover
        """Add custom recipe for quantization to the QuantizeConfig instance.

        Args:
            quantize_recipe (dict): A dict that decide whether given layers should be quantized.
                                    A typical quantize_recipe will be a dict of layer_name and
                                    dict as key-value pairs. In each value dict, there should be
                                    a {'quantize': bool} key-value pair and a {'index': list} pair.
                                    The latter one is used to decide which inputs should be quantized
                                    in some layers with multiple inputs.
                                    For example:
                                        {'conv5_block3_3_conv': {'quantize': False}
                                         'conv5_block3_3_add' : {'quantize': True, 'index': [1, 3]}
                                        }
        """
        self.quantize_recipe.update(quantize_recipe)

    def query_layer(self, layer_name):
        """Query if a specific layer is in the quantize_recipe dict.

        Args:
            layer_name (string): The input layer name.

        Returns:
            layer_recipe (dict): The quantize recipe for this input layer.
        """
        if layer_name in self.quantize_recipe:
            return self.quantize_recipe[layer_name]
        return {}

    def remove_layer(self, layer_name):  # pragma: no cover
        """Remove a specific layer from the quantize_recipe dict.

        Args:
            layer_name (string): The name of layer to be removed.
        """
        if layer_name in self.quantize_recipe:
            del self.quantize_recipe[layer_name]

    def remove_layers(self, layer_names):  # pragma: no cover
        """Remove a batch of layers from the quantize_recipe dict.

        Args:
            layer_names (List): The names of layers to be removed.
        """
        for layer_name in layer_names:
            self.remove_layer(layer_name)

    def get_quantize_recipe(self):  # pragma: no cover
        """Get the current recipe dict for quantization.

        Returns:
            quantize_recipe (dict): A dict that decide whether given layers should be quantized.
        """
        return self.quantize_recipe

    def is_empty(self):  # pragma: no cover
        """Check if the recipe of quantization is an empty dict.

        Returns:
            is_empty (bool): True if no custom recipe is updated to this class.
        """
        if self.quantize_recipe:
            return False
        return True

    def clear_quantize_recipe(self):  # pragma: no cover
        """Clear recipe of quantization to be an empty dict."""
        self.quantize_recipe.clear()


layer_wise_config = {
    "quantize_layers": {
        "Conv2D",
        "Dense",
        "DepthwiseConv2D",
        "MaxPooling2D",
        "AveragePooling2D",
        "GlobalAveragePooling2D",
    },
    "possible_quantize_layers": {"Multiply", "Concatenate", "Add", "BatchNormalization"},
    "weighted_layers": {"Conv2D", "Dense", "DepthwiseConv2D"},
    "multiple_inputs_layers": {"Multiply", "Concatenate", "Add"},
}
