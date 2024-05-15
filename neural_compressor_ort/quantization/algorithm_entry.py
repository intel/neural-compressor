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

import pathlib
import tempfile
from typing import Union

import onnx
from onnx_neural_compressor import config
from onnx_neural_compressor import constants
from onnx_neural_compressor import data_reader
from onnx_neural_compressor import logger
from onnx_neural_compressor import utility
from onnx_neural_compressor.algorithms.smoother import core
from onnx_neural_compressor.algorithms.weight_only import awq
from onnx_neural_compressor.algorithms.weight_only import gptq
from onnx_neural_compressor.algorithms.weight_only import rtn
from onnxruntime import quantization


###################### SmoothQuant Entry ##################################
@utility.register_algo(name=constants.SMOOTH_QUANT)
def smooth_quant_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.SmoothQuantConfig,
    calibration_data_reader: data_reader.CalibrationDataReader,
    model_output: Union[pathlib.Path, str] = None,
    *args,
    **kwargs
) -> Union[pathlib.Path, str, onnx.ModelProto]:
    """Apply smooth quant."""
    assert calibration_data_reader is not None, "Please provide calibration_data_reader"
    assert isinstance(
        calibration_data_reader, data_reader.CalibrationDataReader
    ), "Please follow onnx_neural_compressor/data_reader.py to implement calibration_data_reader"

    # smooth operation
    calibration_data_reader.rewind()
    smoother = core.Smoother(
        model,
        calibration_data_reader,
        providers=quant_config.providers,
    )
    smoothed_model = smoother.transform(**quant_config.to_dict())
    with tempfile.TemporaryDirectory(prefix="ort.quant.") as tmp_dir:
        # ORT quant API requires str input
        onnx.save_model(
            smoothed_model,
            pathlib.Path(tmp_dir).joinpath("smooth.onnx").as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="smooth.onnx_data",
            size_threshold=1024,
            convert_attribute=False,
        )

        # quant operation
        calibration_data_reader.rewind()

        # exclude Mul operations which are inserted during smooth operation
        excluded_nodes = [i.name for i in smoothed_model.graph.node if i.name.endswith("_smooth_mul")]
        quant_config.calibration_data_reader = calibration_data_reader
        quant_config.nodes_to_exclude.extend(excluded_nodes)
        quant_config.convert_to_ort_config()
        quantization.quantize(
            pathlib.Path(tmp_dir).joinpath("smooth.onnx").as_posix(),
            model_output or pathlib.Path(tmp_dir).joinpath("quant_model.onnx").as_posix(),
            quant_config,
        )
        model = model_output or onnx.load(pathlib.Path(tmp_dir).joinpath("quant_model.onnx").as_posix())

    return model


###################### RTN Algo Entry ##################################
@utility.register_algo(name=constants.RTN)
def rtn_quantize_entry(
    model: Union[pathlib.Path, str], quant_config: config.RTNConfig, *args, **kwargs
) -> onnx.ModelProto:
    """The main entry to apply rtn quantization."""
    # map config to each op
    model_info = quant_config.get_model_info(model=model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)
    model = rtn.apply_rtn_on_model(model, configs_mapping)
    return model


###################### GPTQ Algo Entry ##################################
@utility.register_algo(name=constants.GPTQ)
def gptq_quantize_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.GPTQConfig,
    calibration_data_reader: data_reader.CalibrationDataReader,
    *args,
    **kwargs
) -> onnx.ModelProto:
    """The main entry to apply gptq quantization."""
    assert calibration_data_reader is not None, "Please provide calibration_data_reader"
    assert isinstance(
        calibration_data_reader, data_reader.CalibrationDataReader
    ), "Please follow onnx_neural_compressor/data_reader.py to implement calibration_data_reader"

    # map config to each op
    model_info = quant_config.get_model_info(model=model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)

    # regenerate to ensure data exists
    calibration_data_reader.rewind()
    model = gptq.apply_gptq_on_model(model, configs_mapping, calibration_data_reader)
    return model


###################### AWQ Algo Entry ##################################
@utility.register_algo(name=constants.AWQ)
def awq_quantize_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.AWQConfig,
    calibration_data_reader: data_reader.CalibrationDataReader,
    *args,
    **kwargs
) -> onnx.ModelProto:
    """The main entry to apply awq quantization."""
    assert calibration_data_reader is not None, "Please provide calibration_data_reader"
    assert isinstance(
        calibration_data_reader, data_reader.CalibrationDataReader
    ), "Please follow onnx_neural_compressor/data_reader.py to implement calibration_data_reader"

    # map config to each op
    model_info = quant_config.get_model_info(model=model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)

    # regenerate to ensure data exists
    calibration_data_reader.rewind()
    model = awq.apply_awq_on_model(model, configs_mapping, calibration_data_reader)
    return model
