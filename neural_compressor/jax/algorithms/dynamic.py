"""Dynamic quantization algorithm entry point for JAX models."""

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

from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Union

import keras

from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.utils import DYNAMIC_QUANT
from neural_compressor.jax.quantization.layers_dynamic import dynamic_quant_mapping
from neural_compressor.jax.utils import register_algo
from neural_compressor.jax.utils.utility import dtype_mapping, iterate_over_layers


@register_algo(name=DYNAMIC_QUANT)
def dynamic_quantize(
    model: keras.Model,
    configs_mapping: Optional[OrderedDict[Union[str, str], OrderedDict[str, BaseConfig]]] = None,
    quant_config: Optional[BaseConfig] = None,
    *args: Any,
    **kwargs: Any
) -> Any:
    """Quantize model using Dynamic quantization algorithm.

    Args:
        model (keras.Model): JAX model to be quantized.
        configs_mapping (Optional[OrderedDict[Union[str, str], OrderedDict[str, BaseConfig]]]): Mapping of configurations
            for the algorithm.
        quant_config (Optional[BaseConfig]): Quantization configuration for wrapper selection.
        *args (Any): Additional positional arguments (unused).
        **kwargs (Any): Additional keyword arguments (unused).

    Returns:
        keras.Model: The quantized model.
    """
    # Build set of layer paths that this algorithm should process
    layer_configs = {op_name: cfg for (op_name, _op_type), cfg in configs_mapping.items() if cfg.name == DYNAMIC_QUANT}

    qmodel = model

    for layer in qmodel._flatten_layers():
        if layer.__class__ not in dynamic_quant_mapping:
            continue
        layer_id = layer.path if layer.path else layer.name
        if layer_id not in layer_configs:
            continue
        config = layer_configs[layer_id]
        weight_dtype = dtype_mapping[config.weight_dtype]
        activation_dtype = dtype_mapping[config.activation_dtype]
        dynamic_quant_mapping[layer.__class__].prepare(
            layer, weight_dtype, activation_dtype, config.const_scale, config.const_weight
        )
        layer.add_variables()
        layer.post_quantization_cleanup()

    return qmodel
