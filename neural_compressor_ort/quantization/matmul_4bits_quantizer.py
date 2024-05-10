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

from typing import List, Union

import onnx
from onnxruntime.quantization import matmul_4bits_quantizer

from neural_compressor_ort.quantization import matmul_nbits_quantizer

RTNWeightOnlyQuantConfig = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig
AWQWeightOnlyQuantConfig = matmul_nbits_quantizer.AWQWeightOnlyQuantConfig
GPTQWeightOnlyQuantConfig = matmul_nbits_quantizer.GPTQWeightOnlyQuantConfig

class MatMul4BitsQuantizer(matmul_nbits_quantizer.MatMulNBitsQuantizer):

    def __init__(
        self,
        model: Union[onnx.ModelProto, str],
        block_size: int = 128,
        is_symmetric: bool = False,
        accuracy_level: int = 0,
        nodes_to_exclude=None,
        algo_config: matmul_4bits_quantizer.WeightOnlyQuantConfig = None,
        providers: List[str] = ["CPUExecutionProvider"],
    ):
        super().__init__(
            model=model,
            block_size=block_size,
            is_symmetric=is_symmetric,
            accuracy_level=accuracy_level,
            nodes_to_exclude=nodes_to_exclude,
            algo_config=algo_config,
            n_bits=4,
            providers=providers,
        )
