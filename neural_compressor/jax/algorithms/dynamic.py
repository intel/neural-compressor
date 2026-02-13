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

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import keras
import ml_dtypes
from jax import numpy as jnp
from keras.models import clone_model
from keras.src.saving import serialization_lib

from neural_compressor.common import logger
from neural_compressor.common.utils import DYNAMIC_QUANT
from neural_compressor.jax.quantization.config import DynamicQuantConfig
from neural_compressor.jax.quantization.layers import convert_model_dynamic, dynamicquant_mapping
from neural_compressor.jax.utils import algos_mapping, register_algo
from neural_compressor.jax.utils.utility import dtype_mapping, iterate_over_layers


@register_algo(name=DYNAMIC_QUANT)
def dynamic_quantize(model: keras.Model, configs_mapping: DynamicQuantConfig, *args: Any, **kwargs: Any) -> Any:
    """Quantize model using Dynamic quantization algorithm.

    Args:
        model: a JAX model to be quantized.
        configs_mapping: mapping of configurations for the algorithm.

    Returns:
        q_model: the quantized model.
    """
    for _, value in configs_mapping.items():
        config = value
        break
    weight_dtype = dtype_mapping[config.weight_dtype]
    activation_dtype = dtype_mapping[config.activation_dtype]

    # TODO serialization/deserialisation doesn't work for Gemma3CausalLM model
    # Need to further investigation.
    # Instead of copying model we can mark model parameter as mutable.
    # config = serialization_lib.serialize_keras_object(model)
    # qmodel = serialization_lib.deserialize_keras_object(
    #     config, custom_objects={model.__class__.__name__: model.__class__}
    # )
    # qmodel.set_weights(model.get_weights())
    qmodel = model
    operations = [
        lambda layer: dynamicquant_mapping[layer.__class__].prepare(layer, weight_dtype, activation_dtype),
        lambda layer: layer.add_variables(),
        lambda layer: layer.post_quantization_cleanup(),
    ]
    iterate_over_layers(qmodel, operations, filter_function=lambda c: c in dynamicquant_mapping)

    return qmodel
