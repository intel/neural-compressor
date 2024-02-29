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
"""QAT Quantize Helper Class."""

from .quantize_config import QuantizeConfig, global_config, layer_wise_config
from .quantize_layers.optimize_layer import config_quantizable_layers
from .quantize_wrapper import QuantizeWrapper


def init_quantize_config(model, quantize_recipe=None):
    """Initialize quantization config at the beginning of QAT process.

    Args:
        model_name (string): Special pre-optimized model name.
        quantize_recipe (dict): A dict that decide whether given layers should be quantized.

    Returns:
        config (QuantizeConfig): QuantizeConfig instance used to decide whether a specific layer
                                 should be quantized.
    """
    assert "quantize_config" not in global_config, (
        "quantize_config has been unexpectedly " "created. Please check your QAT workflow"
    )

    config = QuantizeConfig()
    config_quantizable_layers(model)

    if quantize_recipe:
        config.add_quantize_recipe(quantize_recipe)

    return config


def _is_quantizable_layer(layer):
    """Query if the input layer should be quantized.

    Args:
        layer (tf.keras.layers.Layer): input Keras layer

    Returns:
        capability (bool): whether the input layer is capable of quantization.
    """
    quantizable = True
    layer_class = layer.__class__.__name__

    quantize_config = global_config["quantize_config"]
    specific_layer_config = quantize_config.query_layer(layer.name)
    if specific_layer_config:
        # the layer is set to be unquantizable by QuantizeConfig
        if not specific_layer_config["quantize"]:
            return False
        else:
            if (
                layer_class in layer_wise_config["quantize_layers"]
                or layer_class in layer_wise_config["possible_quantize_layers"]
            ):
                return True

    if layer_class not in layer_wise_config["quantize_layers"]:
        quantizable = False

    return quantizable


def qat_clone_function(layer):
    """Wrap or leave given layer based on quantize config object parameters.

    Args:
        layer (tf.keras.layers.Layer): input Keras layer

    Returns:
        wrapped_layer (QuantizeWrapper): layer wrapped by QuantizeWrapper class.
    """
    wrapped_layer = layer
    if _is_quantizable_layer(layer):
        wrapped_layer = QuantizeWrapper(layer)

    return wrapped_layer
