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

from typing import Any, Callable, Optional, OrderedDict, Union

import keras
from keras_hub.src.models.causal_lm import CausalLM

from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.utils import STATIC_QUANT
from neural_compressor.jax.quantization.layers_static import static_quant_mapping
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
        lambda layer: static_quant_mapping[layer.__class__].prepare(layer, weight_dtype, activation_dtype),
        lambda layer: layer.add_observers(),
    ]
    iterate_over_layers(qmodel, operations, filter_function=lambda c: c in static_quant_mapping)
    calib_function(qmodel)

    operations = [
        lambda layer: layer.add_variables(),
        lambda layer: layer.convert(),
        lambda layer: layer.post_quantization_cleanup(),
    ]
    iterate_over_layers(qmodel, operations, filter_function=lambda c: c in static_quant_mapping.values())

    if isinstance(qmodel, CausalLM):
        causal_lm_make_replace_generate_function(qmodel, revert=True)

    if hasattr(qmodel, "backbone"):
        qmodel._tracker.unlock()
        qmodel.backbone = KerasQuantizedModelBackboneWrapper(qmodel.backbone, quant_config)
        qmodel._tracker.lock()

    wrapper_cls = WRAPPER_MAPPING.get(qmodel.__class__, KerasQuantizedModelWrapper)
    return wrapper_cls(qmodel, quant_config)
