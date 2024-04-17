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
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import torch

from neural_compressor.common import logger
from neural_compressor.common.base_config import BaseConfig, ComposableConfig, config_registry
from neural_compressor.common.utils import log_quant_execution
from neural_compressor.torch.algorithms import algo_quantizers

FRAMEWORK_NAME = "torch"


def need_apply(configs_mapping: Dict[Tuple[str, callable], BaseConfig], algo_name):
    return any(config.name == algo_name for config in configs_mapping.values())


def prepare(
    model: torch.nn.Module,
    quant_config: Union[BaseConfig, str, Path],
    inplace: bool = True,
):
    """Prepare the model for calibration.

    Insert observers into the model so that it can monitor the input and output tensors during calibration.

    Args:
        model (torch.nn.Module): origin model
        quant_config (Union[BaseConfig, str, Path]): path to quantization config

    Returns:
        model with observers
    """
    prepared_model = model
    registered_configs = config_registry.get_cls_configs()
    if isinstance(quant_config, dict):
        quant_config = ComposableConfig.from_dict(quant_config, config_registry=registered_configs[FRAMEWORK_NAME])
        logger.info(f"Parsed a config dict to construct the quantization config: {quant_config}.")
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), f"Please pass a dict or config instance as the quantization configuration, but got {type(quant_config)}."
    logger.info("Prepare model with config:")
    logger.info(quant_config.to_dict())
    # select quantization algo according to config

    model_info = quant_config.get_model_info(model=prepared_model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)
    for algo_name, algo_quantizer in algo_quantizers.items():
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to prepare model with {algo_name}.")
            quantizer = algo_quantizer(configs_mapping)
            prepared_model = quantizer.prepare(prepared_model)
    setattr(prepared_model, "prepared", True)
    setattr(prepared_model, "quant_config", quant_config)
    return prepared_model


def convert(model: torch.nn.Module, quant_config: Union[BaseConfig, str, Path] = None):
    """Convert the prepared model to a quantized model.

    Load the calibration results and apply post-processing to generate the quantized module.
    Then, swap out the original module with the newly created quantized module.

    Args:
        model (torch.nn.Module): the prepared model
        quant_config (Union[BaseConfig, str, Path]): path to quantization config
    """
    assert (
        getattr(model, "prepared", False) or quant_config is not None
    ), "Please pass quant_config to convert function."
    if getattr(model, "prepared", False):
        if quant_config is None:
            quant_config = model.quant_config

    registered_configs = config_registry.get_cls_configs()
    if isinstance(quant_config, dict):
        quant_config = ComposableConfig.from_dict(quant_config, config_registry=registered_configs[FRAMEWORK_NAME])
        logger.info(f"Parsed a config dict to construct the quantization config: {quant_config}.")
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), f"Please pass a dict or config instance as the quantization configuration, but got {type(quant_config)}."
    logger.info("Convert model with config:")
    logger.info(quant_config.to_dict())
    # select quantization algo according to config

    model_info = quant_config.get_model_info(model=model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)
    for algo_name, algo_quantizer in algo_quantizers.items():
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to convert model with {algo_name}.")
            quantizer = algo_quantizer(configs_mapping)
            q_model = quantizer.convert(model)
    return q_model


def finalize_calibration(model):
    from neural_compressor.torch.algorithms.fp8_quant import save_calib_result

    save_calib_result(model)


def load(model, fname="./saved_results"):
    pass
