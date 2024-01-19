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

from pathlib import Path
from typing import Dict, Tuple

import onnx

from neural_compressor.common.base_config import BaseConfig, ComposableConfig, config_registry
from neural_compressor.common.logger import Logger
from neural_compressor.onnxrt.quantization.config import FRAMEWORK_NAME
from neural_compressor.onnxrt.utils.utility import algos_mapping

logger = Logger().get_logger()


def need_apply(configs_mapping: Dict[Tuple[str, callable], BaseConfig], algo_name):
    return any(config.name == algo_name for config in configs_mapping.values() if hasattr(config, "name"))


# only for internal usage now
def _quantize(
    model_input: Tuple[Path, str],
    quant_config: BaseConfig,
) -> onnx.ModelProto:
    """The main entry to quantize a model.

    Args:
        model_input (Tuple[Path, str]): Path or str to the model to quantize.
        quant_config (BaseConfig): a quantization configuration.

    Returns:
        onnx.ModelProto: The quantized model.
    """
    registered_configs = config_registry.get_cls_configs()
    if isinstance(quant_config, dict):
        quant_config = ComposableConfig.from_dict(quant_config, config_registry=registered_configs[FRAMEWORK_NAME])
        logger.info(f"Parsed a config dict to construct the quantization config: {quant_config}.")
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), f"Please pass a dict or config instance as the quantization configuration, but got {type(quant_config)}."
    logger.info(f"Quantize model with config: \n {quant_config.to_json_string()} \n")

    # select quantization algo according to config
    model_info = quant_config.get_model_info(model=model_input)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)
    for algo_name, algo_func in algos_mapping.items():
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to apply {algo_name} on the model.")
            q_model = algo_func(model_input, configs_mapping)
    return q_model
