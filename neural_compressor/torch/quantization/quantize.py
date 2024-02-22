# Copyright (c) 2024 Intel Corporation
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

import copy
from typing import Any, Callable, Dict, Tuple

import torch

from neural_compressor.common.base_config import BaseConfig, ComposableConfig, config_registry
from neural_compressor.common.utils import log_quant_execution
from neural_compressor.torch.quantization.config import SmoothQuantConfig, StaticQuantConfig
from neural_compressor.torch.utils import is_ipex_available, logger
from neural_compressor.torch.utils.utility import WHITE_MODULE_LIST, algos_mapping, get_model_info

FRAMEWORK_NAME = "torch"


def need_apply(configs_mapping: Dict[Tuple[str, callable], BaseConfig], algo_name):
    return any(config.name == algo_name for config in configs_mapping.values())


@log_quant_execution
def quantize(
    model: torch.nn.Module,
    quant_config: BaseConfig,
    run_fn: Callable = None,
    run_args: Any = None,
    inplace: bool = True,
    example_inputs: Any = None,
) -> torch.nn.Module:
    """The main entry to quantize model with static mode.

    Args:
        model: a float model to be quantized.
        quant_config: a quantization configuration.
        run_fn: a calibration function for calibrating the model. Defaults to None.
        run_args: positional arguments for `run_fn`. Defaults to None.
        example_inputs: used to trace torch model.

    Returns:
        The quantized model.
    """
    q_model = model if inplace else copy.deepcopy(model)
    registered_configs = config_registry.get_cls_configs()
    if isinstance(quant_config, dict):
        quant_config = ComposableConfig.from_dict(quant_config, config_registry=registered_configs[FRAMEWORK_NAME])
        logger.info(f"Parsed a config dict to construct the quantization config: {quant_config}.")
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), f"Please pass a dict or config instance as the quantization configuration, but got {type(quant_config)}."
    logger.info("Quantize model with config:")
    logger.info(quant_config.to_dict())
    # select quantization algo according to config

    if is_ipex_available and (
        isinstance(quant_config, StaticQuantConfig) or isinstance(quant_config, SmoothQuantConfig)
    ):
        model_info = quant_config.get_model_info(q_model, example_inputs)
    else:
        model_info = quant_config.get_model_info(model=q_model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)
    for algo_name, algo_func in algos_mapping.items():
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to apply {algo_name} on the model.")
            q_model = algo_func(
                q_model,
                configs_mapping,
                run_fn=run_fn,
                run_args=run_args,
                example_inputs=example_inputs,
            )
    return q_model
