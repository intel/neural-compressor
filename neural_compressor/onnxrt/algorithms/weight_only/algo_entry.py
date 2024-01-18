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


from typing import Dict, Tuple, Union
from pathlib import Path

import onnx

from neural_compressor.common.logger import Logger
from neural_compressor.common.utility import RTN_WEIGHT_ONLY_QUANT
from neural_compressor.onnxrt.quantization.config import RTNWeightQuantConfig
from neural_compressor.onnxrt.utils.utility import register_algo

logger = Logger().get_logger()


###################### RTN Algo Entry ##################################
@register_algo(name=RTN_WEIGHT_ONLY_QUANT)
def rtn_quantize_entry(
    model: Union[Path, str], 
    configs_mapping: Dict[Tuple[str, callable], RTNWeightQuantConfig],
) -> onnx.ModelProto:
    """The main entry to apply rtn quantization."""
    from neural_compressor.onnxrt.algorithms.weight_only.rtn import apply_rtn_on_model
    model = apply_rtn_on_model(model, configs_mapping)
    return model
