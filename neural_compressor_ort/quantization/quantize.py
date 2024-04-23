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

from pathlib import Path
from typing import Union

import onnx
from onnxruntime.quantization.quantize import QuantConfig

from neural_compressor_ort.common import Logger
from neural_compressor_ort.common.base_config import BaseConfig, ComposableConfig, config_registry
from neural_compressor_ort.common.utils import log_quant_execution
from neural_compressor_ort.quantization.calibrate import CalibrationDataReader
from neural_compressor_ort.quantization.config import FRAMEWORK_NAME
from neural_compressor_ort.utils.utility import algos_mapping

logger = Logger().get_logger()


# ORT-like user-facing API
def quantize(
    model_input: Union[str, Path, onnx.ModelProto],
    model_output: Union[str, Path],
    quant_config: QuantConfig,
):
    from neural_compressor_ort.quantization.config import DynamicQuantConfig, StaticQuantConfig
    if isinstance(quant_config, StaticQuantConfig):
        if quant_config.extra_options.get("SmoothQuant", False):
            from neural_compressor_ort.quantization.algorithm_entry import smooth_quant_entry
            from neural_compressor_ort.quantization.config import generate_inc_sq_config

            inc_sq_config = generate_inc_sq_config(quant_config)
            smooth_quant_entry(
                model_input, inc_sq_config, quant_config.calibration_data_reader, model_output=model_output
            )
            return model
        else:
            # call static_quant_entry
            pass
    elif isinstance(quant_config, DynamicQuantConfig):
        from neural_compressor_ort.quantization.algorithm_entry import 
        inc_dynamic_config = generate_inc_dynamic_config(model_input, quant_config)
        dynamic_quantize_entry(model_input, quant_config, model_output=model_output)
    else:
        raise TypeError("Invalid quantization config type, it must be either StaticQuantConfig or DynamicQuantConfig.")


def _need_apply(quant_config: BaseConfig, algo_name):
    return quant_config.name == algo_name if hasattr(quant_config, "name") else False


# * only for internal usage now
@log_quant_execution
def _quantize(
    model_input: Union[Path, str],
    quant_config: BaseConfig,
    calibration_data_reader: CalibrationDataReader = None,
) -> onnx.ModelProto:
    """The main entry to quantize a model.

    Args:
        model_input (Union[Path, str]): Path or str to the model to quantize.
        quant_config (BaseConfig): a quantization configuration.
        calibration_data_reader (CalibrationDataReader, optional): dataloader for calibration.
            Defaults to None.

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
    logger.info(f"Quantize model with config: \n {quant_config} \n")

    # select quantization algo according to config
    for algo_name, algo_func in algos_mapping.items():
        if _need_apply(quant_config, algo_name):
            logger.info(f"Start to apply {algo_name} on the model.")
            q_model = algo_func(model_input, quant_config, calibration_data_reader=calibration_data_reader)
    return q_model
