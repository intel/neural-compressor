# Copyright (c) 2025 Intel Corporation
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
from typing import Any, Callable, Optional, OrderedDict, Union

import jax
import keras
import ml_dtypes
from keras.models import clone_model
from keras.src.saving import serialization_lib
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM

from neural_compressor.common import logger
from neural_compressor.common.utils import STATIC_QUANT
from neural_compressor.jax.quantization.config import BaseConfig, StaticQuantConfig
from neural_compressor.jax.quantization.layers import staticquant_mapping
from neural_compressor.jax.quantization.saving import (
    WRAPPER_MAPPING,
    KerasQuantizedModelBackboneWrapper,
    KerasQuantizedModelWrapper,
)
from neural_compressor.jax.utils import register_algo
from neural_compressor.jax.utils.utility import (
    causal_lm_make_replace_generate_function,
    dtype_mapping,
    iterate_over_layers,
)


def prepare_deserialized_quantized_model(
    model: keras.Model,
    quant_config: StaticQuantConfig,
) -> KerasQuantizedModelWrapper:
    """Transform a loaded statically quantized model.

    It prepares the model for inference by preparing the quantized layers.
    Args:
        model: loaded base keras model
        quant_config: quantization configuration
    Returns:
        KerasQuantizedModelWrapper: the transformed quantized model
    """
    model_info = quant_config.get_model_info(model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)

    for _, value in configs_mapping.items():
        config = value
        break

    weight_dtype = dtype_mapping[config.weight_dtype]
    activation_dtype = dtype_mapping[config.activation_dtype]

    qmodel = model
    operations = [
        lambda layer: staticquant_mapping[layer.__class__].prepare(layer, weight_dtype, activation_dtype),
        lambda layer: layer.add_variables(),
        lambda layer: layer.post_quantization_cleanup(),
    ]

    iterate_over_layers(qmodel, operations, filter_function=lambda c: c in staticquant_mapping)
    if isinstance(qmodel, Backbone):
        qmodel = KerasQuantizedModelBackboneWrapper(qmodel, quant_config)
    else:
        wrapper_cls = WRAPPER_MAPPING.get(qmodel.__class__, KerasQuantizedModelWrapper)
        qmodel = wrapper_cls(qmodel, quant_config)
        if hasattr(qmodel, "backbone"):
            qmodel._tracker.unlock()
            qmodel.backbone = KerasQuantizedModelBackboneWrapper(qmodel.backbone, quant_config)
            qmodel._tracker.lock()

    return qmodel


@register_algo(name=STATIC_QUANT)
def static_quantize(
    model: keras.Model,
    configs_mapping: Optional[OrderedDict[Union[str, str], OrderedDict[str, BaseConfig]]] = None,
    quant_config: Optional[BaseConfig] = None,
    calib_function: Optional[Callable] = None,
) -> keras.Model:
    """Quantize model using Static quantization algorithm.

    Args:
        model: a JAX model to be quantized.
        configs_mapping: mapping of configurations for the algorithm.

    Returns:
        q_model: the quantized model
    """
    for _, value in configs_mapping.items():
        config = value
        break
    weight_dtype = dtype_mapping[config.weight_dtype]
    activation_dtype = dtype_mapping[config.activation_dtype]

    # TODO serialization/deserialization doesn't work for Gemma3CausalLM model
    # Need to further investigation.
    # Instead of copying model we can mark model parameter as mutable.
    # config = serialization_lib.serialize_keras_object(model)
    # qmodel = serialization_lib.deserialize_keras_object(
    #     config, custom_objects={model.__class__.__name__: model.__class__}
    # )
    # qmodel.set_weights(model.get_weights())
    qmodel = model

    if isinstance(qmodel, CausalLM):
        causal_lm_make_replace_generate_function(qmodel)

    operations = [
        lambda layer: staticquant_mapping[layer.__class__].prepare(layer, weight_dtype, activation_dtype),
        lambda layer: layer.add_observers(),
    ]
    iterate_over_layers(qmodel, operations, filter_function=lambda c: c in staticquant_mapping)
    calib_function(qmodel)

    operations = [
        lambda layer: layer.add_variables(),
        lambda layer: layer.convert(),
        lambda layer: layer.post_quantization_cleanup(),
    ]
    iterate_over_layers(qmodel, operations, filter_function=lambda c: c in staticquant_mapping.values())

    if isinstance(qmodel, CausalLM):
        causal_lm_make_replace_generate_function(qmodel, revert=True)

    if hasattr(qmodel, "backbone"):
        qmodel._tracker.unlock()
        qmodel.backbone = KerasQuantizedModelBackboneWrapper(qmodel.backbone, quant_config)
        qmodel._tracker.lock()

    wrapper_cls = WRAPPER_MAPPING.get(qmodel.__class__, KerasQuantizedModelWrapper)
    return wrapper_cls(qmodel, quant_config)
