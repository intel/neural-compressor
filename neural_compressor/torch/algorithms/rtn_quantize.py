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


from typing import Dict

import torch

from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.logger import Logger
from neural_compressor.common.utility import RTN_WEIGHT_ONLY_QUANT
from neural_compressor.torch.algorithms.rtn import rtn_quantize as torch_rtn_quantize
from neural_compressor.torch.quantization.config import RTNWeightQuantConfig
from neural_compressor.torch.utils import fetch_module, register_algo, set_module

logger = Logger().get_logger()


def _apply_rtn_on_single_module(module: torch.nn.Module, quant_config: RTNWeightQuantConfig) -> torch.nn.Module:
    enable_full_range = quant_config.enable_full_range
    enable_mse_search = quant_config.enable_mse_search
    group_dim = quant_config.group_dim
    dtype = quant_config.weight_dtype
    num_bits = quant_config.weight_bits
    scheme = quant_config.weight_sym
    group_size = quant_config.weight_group_size
    return_int = quant_config.return_int
    return torch_rtn_quantize(
        module,
        num_bits,
        group_size,
        scheme,
        return_int=return_int,
        data_type=dtype,
        enable_full_range=enable_full_range,
        enable_mse_search=enable_mse_search,
        group_dim=group_dim,
    )


def _convert_quant_config_into_quant_config_mapping(
    fp32_model: torch.nn.Module, quant_config: BaseConfig
) -> Dict[str, BaseConfig]:
    # TODO(Yi) enhance it, currently we only assign the global config to module
    # model_info: List[Tuple[str, Callable]] = []
    linear_lst = []
    for name, module in fp32_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_lst.append(name)
    _quant_config = quant_config if quant_config.global_config is None else quant_config.global_config
    quant_config_mapping: Dict[str, BaseConfig] = {name: _quant_config for name in linear_lst}
    return quant_config_mapping


@register_algo(name=RTN_WEIGHT_ONLY_QUANT)
def rtn_quantize_entry(model: torch.nn.Module, quant_config: RTNWeightQuantConfig) -> torch.nn.Module:
    quant_config_mapping: Dict[str, RTNWeightQuantConfig] = _convert_quant_config_into_quant_config_mapping(
        model, quant_config
    )
    """The main entry to apply rtn quantization."""
    for op_name, quant_config in quant_config_mapping.items():
        original_module = fetch_module(model, op_name)
        logger.info(f"Apply RTN on module: {op_name}, {original_module}")
        rtn_module = _apply_rtn_on_single_module(original_module, quant_config)
        set_module(model, op_name, rtn_module)
    return model
