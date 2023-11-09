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

from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.utility import RTN_WEIGHT_ONLY_QUANT
from neural_compressor.torch.quantization.config import parse_config_from_dict
from neural_compressor.torch.utils import algos_mapping
from neural_compressor.utils import logger


def quantize(model, quant_config: BaseConfig):
    """The main entry to quantize model."""
    # TODO(Yi) add more args, like calib_func, calib_func_args
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
    if quant_config.name == RTN_WEIGHT_ONLY_QUANT:
        quant_fn = algos_mapping[quant_config.name]
    else:
        raise NotImplementedError("Currently, only the rtn algorithm is being ported.")
    qmodel = quant_fn(model, quant_config)
    return qmodel
