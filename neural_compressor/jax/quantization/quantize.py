# Copyright (c) 2024-2026 Intel Corporation
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
"""Intel Neural Compressor JAX quantization base API."""

from typing import Callable, Dict, Tuple

import keras
from keras_hub.src.models.causal_lm import CausalLM

from neural_compressor.common import logger
from neural_compressor.common.utils import STATIC_QUANT, Mode, log_process
from neural_compressor.jax.quantization.config import JaxBaseConfig
from neural_compressor.jax.quantization.saving import (
    WRAPPER_MAPPING,
    KerasQuantizedModelBackboneWrapper,
)
from neural_compressor.jax.utils import algos_mapping, check_backend
from neural_compressor.jax.utils.utility import causal_lm_make_replace_generate_function

from .clone_model import clone_model


def need_apply(configs_mapping: Dict[Tuple[str, callable], JaxBaseConfig], algo_name):
    """Determine whether a quantization algorithm should be applied.

    Args:
        configs_mapping (Dict[Tuple[str, callable], JaxBaseConfig]): Mapping of layer identifiers to configs.
        algo_name (str): Algorithm name to check.

    Returns:
        bool: True if any config matches the algorithm name.
    """
    return any(config.name == algo_name for config in configs_mapping.values())


# fmt: off
@log_process(mode=Mode.QUANTIZE)
def quantize_model(
    model: keras.Model,
    quant_config: JaxBaseConfig,
    calib_function: Callable = None,
    inplace: bool = True
):
    """Return a quantized Keras model according to the given configuration.

    Args:
        model (keras.Model): FP32 Keras model to be quantized.
        quant_config (JaxBaseConfig): Quantization configuration. Can be a single config
            or a JaxComposableConfig (created via config1 + config2).
        calib_function (Callable, optional): Function used for model calibration, required for static quantization.
        inplace (bool): When True, the original model is modified in-place and should not be used afterward. False creates a copy of original model

    Returns:
        keras.Model: The quantized model.
    """
# fmt: on
    check_backend()
    if not inplace:
        model = clone_model(model)

    # Build configs_mapping via the common path. ComposableQuantConfig overrides
    # get_model_info / to_config_mapping to compose its sub-configs.
    model_info = quant_config.get_model_info(model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)

    # Pre-quantization setup for CausalLM models
    if isinstance(model, CausalLM):
        causal_lm_make_replace_generate_function(model)

    # Execute algorithms - static first to ensure calibration runs on original FP32/BF16 model
    algo_order = sorted(algos_mapping.keys(), key=lambda name: (0 if name == STATIC_QUANT else 1))
    for algo_name in algo_order:
        algo_func = algos_mapping[algo_name]
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to apply {algo_name} on the model.")
            model = algo_func(model, configs_mapping, calib_function)

    # Post-quantization: revert CausalLM generate function
    if isinstance(model, CausalLM):
        causal_lm_make_replace_generate_function(model, revert=True)

    if hasattr(model, "backbone"):
        model._tracker.unlock()
        model.backbone = KerasQuantizedModelBackboneWrapper(model.backbone, quant_config)
        model._tracker.lock()

    wrapper_cls = WRAPPER_MAPPING[model.__class__]
    return wrapper_cls(model, quant_config)
