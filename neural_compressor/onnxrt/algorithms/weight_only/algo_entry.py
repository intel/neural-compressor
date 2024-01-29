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
from typing import Dict, Tuple, Union

import onnx

from neural_compressor.common import logger
from neural_compressor.common.utils import RTN
from neural_compressor.onnxrt.quantization.config import RTNConfig
from neural_compressor.onnxrt.utils.utility import register_algo


###################### RTN Algo Entry ##################################
@register_algo(name=RTN)
def rtn_quantize_entry(model: Union[Path, str], quant_config: RTNConfig, *args, **kwargs) -> onnx.ModelProto:
    """The main entry to apply rtn quantization."""
    from neural_compressor.onnxrt.algorithms.weight_only.rtn import apply_rtn_on_model

    # map config to each op
    model_info = quant_config.get_model_info(model=model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)
    model = apply_rtn_on_model(model, configs_mapping)
    return model
