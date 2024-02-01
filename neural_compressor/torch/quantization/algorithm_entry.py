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

from typing import Callable, Dict, Tuple

import torch

from neural_compressor.common.utils import FP8_QUANT, GPTQ, HQQ, RTN  # unified namespace
from neural_compressor.torch.algorithms.weight_only import (
    HQQModuleConfig,
    HQQuantizer,
    QTensorConfig,
    gptq_quantize,
    rtn_quantize,
)
from neural_compressor.torch.quantization import GPTQConfig, HQQConfig, RTNConfig
from neural_compressor.torch.utils import logger, register_algo


###################### RTN Algo Entry ##################################
@register_algo(RTN)
@torch.no_grad()
def rtn_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], RTNConfig], *args, **kwargs
) -> torch.nn.Module:
    """The main entry to apply rtn quantization."""
    # rebuild weight_config for rtn_quantize function
    weight_config = {}
    for (op_name, op_type), quant_config in configs_mapping.items():
        weight_config[op_name] = {
            "dtype": quant_config.dtype,
            "bits": quant_config.bits,
            "scheme": "sym" if quant_config.use_sym else "asym",
            "group_size": quant_config.group_size,
            "group_dim": quant_config.group_dim,
            "use_full_range": quant_config.use_full_range,
            "use_mse_search": quant_config.use_mse_search,
            "use_layer_wise": quant_config.use_layer_wise,
            "export_compressed_model": quant_config.export_compressed_model,
            "use_double_quant": quant_config.use_double_quant,
            "double_quant_dtype": quant_config.double_quant_dtype,
            "double_quant_bits": quant_config.double_quant_bits,
            "double_quant_scheme": "sym" if quant_config.double_quant_use_sym else "asym",
            "double_quant_group_size": quant_config.double_quant_group_size,
        }

    model = rtn_quantize(model, weight_config=weight_config)
    return model


###################### GPTQ Algo Entry ##################################
@register_algo(GPTQ)
@torch.no_grad()
def gptq_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], GPTQConfig], *args, **kwargs
) -> torch.nn.Module:
    logger.info("Quantize model with the GPTQ algorithm.")

    model, quantization_perm = gptq_quantize(model=model, configs_mapping=configs_mapping, *args, **kwargs)
    # Assign the gptq config as an attribute of model
    model._gptq_quantization_perm = quantization_perm
    return model


###################### HHQ Algo Entry ##################################


def _convert_hqq_module_config(config: HQQConfig) -> HQQModuleConfig:
    nbits = config.nbits
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
    print(hqq_module_config)
    return hqq_module_config


def _parse_hqq_configs_mapping(configs_mapping):
    qconfig_mapping = {}
    for (op_name, op_type), quant_config in configs_mapping.items():
        qconfig_mapping[op_name] = _convert_hqq_module_config(quant_config)
    return qconfig_mapping


@register_algo(name=HQQ)
@torch.no_grad()
def hqq_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, Callable], HQQConfig], *args, **kwargs
) -> torch.nn.Module:
    logger.info("Quantize model with the HQQ algorithm.")
    qconfig_mapping = _parse_hqq_configs_mapping(configs_mapping)
    hqq_quantizer = HQQuantizer(qconfig_mapping)
    q_model = hqq_quantizer.prepare(model)
    return q_model


###################### Habana FP8 Algo Entry ##################################
from neural_compressor.torch.utils import is_hpex_available

if is_hpex_available():
    from neural_compressor.torch.algorithms.habana_fp8 import quantize

    @register_algo(FP8_QUANT)
    def fp8_quant_entry(model, qconfig_mapping, run_fn=None, run_args=None, inplace=True):
        return quantize(model, qconfig_mapping, run_fn=run_fn, run_args=run_args, inplace=inplace)
