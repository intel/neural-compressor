"""Static quantization algorithm entry point for JAX models."""

# Copyright (c) 2025-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, OrderedDict, Union

import keras

from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.utils import STATIC_QUANT
from neural_compressor.jax.quantization.layers_static import static_quant_mapping
from neural_compressor.jax.utils import register_algo
from neural_compressor.jax.utils.utility import (
    dtype_mapping,
    iterate_over_layers,
)


@register_algo(name=STATIC_QUANT)
def static_quantize(
    model: keras.Model,
    configs_mapping: Optional[OrderedDict[Union[str, str], OrderedDict[str, BaseConfig]]] = None,
    quant_config: Optional[BaseConfig] = None,
    calib_function: Optional[Callable] = None,
) -> keras.Model:
    """Quantize model using Static quantization algorithm.

    Args:
        model (keras.Model): JAX model to be quantized.
        configs_mapping (Optional[OrderedDict[Union[str, str], OrderedDict[str, BaseConfig]]]): Mapping of configurations
            for the algorithm.
        quant_config (Optional[BaseConfig]): Quantization configuration for wrapper selection.
        calib_function (Optional[Callable]): Calibration function used to collect activation statistics.

    Returns:
        keras.Model: The quantized model.
    """
    # Build set of layer paths that this algorithm should process
    layer_configs = {
        op_name: cfg for (op_name, _op_type), cfg in configs_mapping.items()
        if cfg.name == STATIC_QUANT
    }

    qmodel = model

    # Phase 1: Prepare layers and add observers
    for layer in qmodel._flatten_layers():
        if layer.__class__ not in static_quant_mapping:
            continue
        layer_id = layer.path if layer.path else layer.name
        if layer_id not in layer_configs:
            continue
        config = layer_configs[layer_id]
        weight_dtype = dtype_mapping[config.weight_dtype]
        activation_dtype = dtype_mapping[config.activation_dtype]
        static_quant_mapping[layer.__class__].prepare(
            layer, weight_dtype, activation_dtype, config.const_scale, config.const_weight
        )
        layer.add_observers()

    # Phase 2: Run calibration on original model with observers
    calib_function(qmodel)

    # Phase 3: Convert observed layers to quantized form
    operations = [
        lambda layer: layer.add_variables(),
        lambda layer: layer.convert(),
        lambda layer: layer.post_quantization_cleanup(),
    ]
    iterate_over_layers(qmodel, operations, filter_function=lambda c: c in static_quant_mapping.values())

    return qmodel
