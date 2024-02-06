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
"""HQQ quantization APIs."""

import torch

from neural_compressor.torch.utils import logger

from .config import HQQModuleConfig, QTensorConfig
from .quantizer import HQQuantizer

__all__ = ["hqq_quantize"]


def _convert_hqq_module_config(config) -> HQQModuleConfig:
    # * 3.x API use `bits` for woq while HQQ internal API use `nbits`
    nbits = config.bits
    group_size = config.group_size
    quant_zero = config.quant_zero
    quant_scale = config.quant_scale
    scale_quant_group_size = config.scale_quant_group_size

    weight_qconfig = QTensorConfig(
        nbits=nbits, channel_wise=True, group_size=group_size, optimize=True, round_zero=True if nbits == 4 else False
    )
    zero_qconfig = None
    if quant_zero:
        zero_qconfig = QTensorConfig(nbits=8, channel_wise=False, group_size=None, optimize=False)
    scale_qconfig = None
    if quant_scale:
        scale_qconfig = QTensorConfig(nbits=8, channel_wise=True, group_size=scale_quant_group_size, optimize=False)
    hqq_module_config = HQQModuleConfig(weight=weight_qconfig, scale=scale_qconfig, zero=zero_qconfig)
    logger.debug(hqq_module_config)
    return hqq_module_config


def _parse_hqq_configs_mapping(configs_mapping):
    qconfig_mapping = {}
    for (op_name, op_type), quant_config in configs_mapping.items():
        if quant_config.skip_lm_head and "lm_head" in op_name:
            logger.warning("Skip quantizing %s due to `skip_lm_head` is True.", op_name)
            continue
        qconfig_mapping[op_name] = _convert_hqq_module_config(quant_config)
    return qconfig_mapping


@torch.no_grad()
def hqq_quantize(model: torch.nn.Module, configs_mapping, *args, **kwargs) -> torch.nn.Module:
    qconfig_mapping = _parse_hqq_configs_mapping(configs_mapping)
    hqq_quantizer = HQQuantizer(qconfig_mapping)
    q_model = hqq_quantizer.prepare(model)
    return q_model
