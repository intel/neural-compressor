# Copyright (c) 2023 Intel Corporation
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

from typing import Any, Callable, Dict, Tuple

import torch

from neural_compressor.common.base_config import BaseConfig, ComposableConfig, registered_configs
from neural_compressor.common.logger import Logger
from neural_compressor.torch.quantization.config import FRAMEWORK_NAME
from neural_compressor.torch.utils import algos_mapping, get_model_info

logger = Logger().get_logger()


def need_apply(configs_mapping: Dict[Tuple[str, callable], BaseConfig], algo_name):
    return any(config.name == algo_name for config in configs_mapping.values())


def quantize(
    model: torch.nn.Module,
    quant_config: BaseConfig,
    calib_dataloader=None,
    calib_func: Callable = None,
    calib_func_arg: Any = None,
) -> torch.nn.Module:
    """The main entry to quantize model.

    Args:
        model: a float model to be quantized.
        quant_config: a quantization configuration.
        calib_dataloader: a calibration dataloader for calibrating the model. Defaults to None.
        calib_func: a calibration function for calibrating the model. Defaults to None.
        calib_func_arg: positional arguments for `calib_func`. Defaults to None.

    Returns:
        The quantized model.
    """
    # TODO (Yi) support combine more than one algo
    if isinstance(quant_config, dict):
        quant_config = ComposableConfig.from_dict(quant_config, config_registry=registered_configs[FRAMEWORK_NAME])
        logger.info(f"Parsed a config dict to construct the quantization config: {quant_config}.")
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), "Please pass a dict or config instance as the quantization configuration."
    logger.info(f"Quantize model with config: \n {quant_config.to_json_string()} \n")
    # select quantization algo according to config

    model_info = get_model_info(model=model, white_module_list=[torch.nn.Linear])
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)
    for algo_name, algo_func in algos_mapping.items():
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to apply {algo_name} on the model.")
            model = algo_func(model, configs_mapping, calib_dataloader, calib_func, calib_func_arg)
    return model
