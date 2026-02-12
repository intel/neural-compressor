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


from typing import Any, Callable, Dict, Tuple, Union

import keras

from neural_compressor.common import logger
from neural_compressor.common.base_config import BaseConfig, ComposableConfig, config_registry
from neural_compressor.common.utils import Mode, log_process
from neural_compressor.jax.utils import algos_mapping


def need_apply(configs_mapping: Dict[Tuple[str, callable], BaseConfig], algo_name):
    """Whether to apply the algorithm."""
    return any(config.name == algo_name for config in configs_mapping.values())


# fmt: off
@log_process(mode=Mode.QUANTIZE)
def quantize_model(
    model: keras.Model,
    quant_config: BaseConfig,
    calib_function: Callable = None,
    inplace: bool = True
):
    """Return a quantized Keras model according to the given configuration.

    Args:
        model:          FP32 Keras model to be quantized.
        quant_config:   Quantization configuration.
        calib_function: Function used for model calibration, required for static quantization.
        inplace:        When True, the original model is modified in-place and should not be used
                        afterward. A value of False is not yet supported.

    Returns:
        The quantized model.
    """
# fmt: on
    if not inplace:
        raise NotImplementedError("Out of place quantization is not supported yet. "
                                  "Please set parameter inplace=True for quantize_model() to modify the model in-place")

    model_info = quant_config.get_model_info(model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    for algo_name, algo_func in algos_mapping.items():
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to apply {algo_name} on the model.")
            model = algo_func(model, configs_mapping, quant_config, calib_function)
    return model
