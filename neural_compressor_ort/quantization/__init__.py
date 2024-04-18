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

from neural_compressor_ort.quantization.algorithm_entry import (
    smooth_quant_entry,
    rtn_quantize_entry,
    gptq_quantize_entry,
    awq_quantize_entry,
)
from neural_compressor_ort.quantization.calibrate import CalibrationDataReader
from neural_compressor_ort.quantization.config import (
    RTNConfig,
    get_default_rtn_config,
    GPTQConfig,
    get_default_gptq_config,
    AWQConfig,
    get_default_awq_config,
    SmoothQuantConfig,
    get_default_sq_config,
    StaticQuantConfig,
    DynamicQuantConfig,
    get_woq_tuning_config,
)
from neural_compressor_ort.quantization.autotune import autotune, get_all_config_set

from neural_compressor_ort.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
from neural_compressor_ort.quantization.matmul_nbits_quantizer import (
    RTNWeightOnlyQuantConfig,
    GPTQWeightOnlyQuantConfig,
    AWQWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)

from neural_compressor_ort.quantization.quantize import quantize

from onnxruntime.quantization.quant_utils import QuantFormat, QuantType

__all__ = [
    "smooth_quant_entry",
    "rtn_quantize_entry",
    "gptq_quantize_entry",
    "awq_quantize_entry",
    "RTNConfig",
    "get_default_rtn_config",
    "GPTQConfig",
    "get_default_gptq_config",
    "AWQConfig",
    "get_default_awq_config",
    "SmoothQuantConfig",
    "get_default_sq_config",
    "get_woq_tuning_config" ,
    "get_all_config_set",
    "StaticQuantConfig",
    "DynamicQuantConfig",
    "CalibrationDataReader",
    "autotune",
    "MatMul4BitsQuantizer",
    "MatMulNBitsQuantizer",
    "RTNWeightOnlyQuantConfig",
    "GPTQWeightOnlyQuantConfig",
    "AWQWeightOnlyQuantConfig",
    "quantize",
    "QuantFormat",
    "QuantType",
]
