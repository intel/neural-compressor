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

from typing import Dict

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver, PlaceholderObserver
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.x86_inductor_quantizer import QuantizationConfig, X86InductorQuantizer


def create_quant_spec_from_config(dtype, sym, granularity, algo, is_dynamic=False) -> QuantizationSpec:
    dtype_mapping: Dict[str, torch.dtype] = {"int8": torch.int8, "uint8": torch.uint8}
    select_dtype = dtype_mapping[dtype]
    min_max_mapping = {torch.int8: (-128, 127), torch.uint8: (0, 255)}
    qscheme_mapping = {
        "per_channel": {True: torch.per_channel_symmetric, False: torch.per_tensor_affine},
        "per_tensor": {True: torch.per_tensor_symmetric, False: torch.per_tensor_affine},
    }
    observer_mapping = {
        "placeholder": PlaceholderObserver,
        "minmax": MinMaxObserver,
        "kl": HistogramObserver,
    }
    # Force to use placeholder observer for dynamic quantization
    if is_dynamic:
        algo = "placeholder"
    # algo
    observer_or_fake_quant_ctr = observer_mapping[algo]
    # qscheme
    qscheme = qscheme_mapping[granularity][sym]
    quantization_spec = QuantizationSpec(
        dtype=select_dtype,
        quant_min=min_max_mapping[select_dtype][0],
        quant_max=min_max_mapping[select_dtype][1],
        observer_or_fake_quant_ctr=observer_or_fake_quant_ctr,
        qscheme=qscheme,
        is_dynamic=is_dynamic,
    )
    return quantization_spec


def _map_inc_config_to_torch_quant_config(inc_config, is_dynamic=False) -> QuantizationConfig:
    default_quant_config = xiq.get_default_x86_inductor_quantization_config(is_dynamic=is_dynamic)
    input_act_quant_spec = create_quant_spec_from_config(
        inc_config.act_dtype, inc_config.act_sym, inc_config.act_granularity, inc_config.act_algo, is_dynamic=is_dynamic
    )
    weight_quant_spec = create_quant_spec_from_config(
        inc_config.w_dtype, inc_config.w_sym, inc_config.w_granularity, inc_config.w_algo
    )
    quant_config = QuantizationConfig(
        input_activation=input_act_quant_spec,
        output_activation=default_quant_config.output_activation,
        weight=weight_quant_spec,
        bias=default_quant_config.bias,
        is_qat=False,
    )
    return quant_config


def create_xiq_quantizer_from_pt2e_config(config, is_dynamic=False) -> X86InductorQuantizer:
    quantizer = xiq.X86InductorQuantizer()
    # set global
    global_config = _map_inc_config_to_torch_quant_config(config, is_dynamic)
    quantizer.set_global(global_config)
    # Skip the local config for now (need torch 2.4)
    return quantizer
