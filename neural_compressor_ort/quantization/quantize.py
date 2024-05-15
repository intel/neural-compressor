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
from typing import Union

import onnx
from onnxruntime.quantization.quantize import QuantConfig

from onnx_neural_compressor import config
from onnx_neural_compressor.quantization import algorithm_entry as algos


# ORT-like user-facing API
def quantize(
    model_input: Union[str, pathlib.Path, onnx.ModelProto],
    model_output: Union[str, pathlib.Path],
    quant_config: QuantConfig,
):
    if isinstance(quant_config, config.StaticQuantConfig):
        if quant_config.extra_options.get("SmoothQuant", False):
            nc_sq_config = config.generate_nc_sq_config(quant_config)
            algos.smooth_quant_entry(
                model_input, nc_sq_config, quant_config.calibration_data_reader, model_output=model_output
            )
        else:
            # call static_quant_entry
            pass
    elif isinstance(quant_config, config.DynamicQuantConfig):
        # call dynamic_quant_entry
        pass
    else:
        raise TypeError("Invalid quantization config type, it must be either StaticQuantConfig or DynamicQuantConfig.")
