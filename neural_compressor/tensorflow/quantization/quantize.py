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

from typing import Any, Callable

import tensorflow as tf

from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.logger import Logger
from neural_compressor.common.utility import STATIC_QUANT
from neural_compressor.tensorflow.quantization.config import parse_config_from_dict
from neural_compressor.tensorflow.utils import algos_mapping

logger = Logger().get_logger()


def quantize_model(
    model: tf.keras.Model, quant_config: BaseConfig, calib_dataloader: Callable = None, calib_iteration: int = 100
) -> tf.keras.Model:
    """The main entry to quantize model.

    Args:
        model: a fp32 model to be quantized.
        quant_config: a quantization configuration.
        calib_dataloader: a data loader for calibration.
        calib_iteration: the iteration of calibration.

    Returns:
        q_model: the quantized model.
    """
    if isinstance(quant_config, dict):
        quant_config = parse_config_from_dict(quant_config)
        logger.info("Parsed dict to construct the quantization config.")
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), "Please pass a dict or config instance as the quantization configuration."
    logger.info(f"Quantize model with config: \n {quant_config.to_json_string()} \n")

    # select quantization algo according to config
    # TODO (Yi) support combine more than one algo
    if quant_config.name == STATIC_QUANT:
        quant_fn = algos_mapping[quant_config.name]
    else:
        raise NotImplementedError("Currently, only the basic algorithm is being ported.")
    qmodel = quant_fn(model, quant_config, calib_dataloader, calib_iteration)
    return qmodel
