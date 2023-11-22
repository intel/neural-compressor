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


import os
from typing import Dict, Tuple

import torch

from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.logger import Logger
from neural_compressor.common.utility import GPTQ, RTN_WEIGHT_ONLY_QUANT
from neural_compressor.torch.algorithms.rtn import rtn_quantize as torch_rtn_quantize
from neural_compressor.torch.quantization.config import GPTQConfig, RTNWeightQuantConfig
from neural_compressor.torch.utils import fetch_module, register_algo, set_module

logger = Logger().get_logger()


###################### RTN Algo Entry ##################################
def _apply_rtn_on_single_module(module: torch.nn.Module, quant_config: RTNWeightQuantConfig) -> torch.nn.Module:
    # TODO (Yi) remove it
    enable_full_range = quant_config.enable_full_range
    enable_mse_search = quant_config.enable_mse_search
    group_dim = quant_config.group_dim
    dtype = quant_config.weight_dtype
    num_bits = quant_config.weight_bits
    scheme = "sym" if quant_config.weight_sym else "asym"
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
def rtn_quantize_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], RTNWeightQuantConfig], *args, **kwargs
) -> torch.nn.Module:
    """The main entry to apply rtn quantization."""
    for (op_type, op_name), quant_config in configs_mapping.items():
        original_module = fetch_module(model, op_name)
        logger.info(f"Apply RTN on module: {op_name}, {original_module}")
        rtn_module = _apply_rtn_on_single_module(original_module, quant_config)
        set_module(model, op_name, rtn_module)
    return model


###################### GPTQ Algo Entry ##################################


def gptq_config_mapping(configs_mapping: Dict[Tuple[str, callable], GPTQConfig]):
    # convert GPTQ_CONFIG to gptq_quantize's weight config
    # convert tune_cfg to gptq_quantize's weight config
    """please refer to weight_config which can be analyzed by user-define API function weight_only.gptq_quantize
    keys of weight_config can not only be specific name, but can also be a re formula
    weight_config = {
        "layer_name_1": {
            'wbits': 4,
            'group_size': 128,
            'sym': False,
            'percdamp': 0.01,
            'actorder': True
        },
        "layer_name_2": {
            'wbits': 4,
            'group_size': 128,
            'sym': False,
            'percdamp': 0.01,
            'actorder': True
        }
        ...
    }
    """
    # for layer_wise quant mode
    model_path = None
    layer_wise = False
    # TODO (Yi) uncomment it when port layer-wise
    # if recipe_cfgs.get("layer_wise_quant", False):
    #     layer_wise = True
    #     from .torch_utils.layer_wise_quant.utils import LWQ_WORKSPACE, _get_path, register_weight_hooks

    #     os.makedirs(LWQ_WORKSPACE, exist_ok=True)
    #     # model_path = recipe_cfgs["layer_wise_quant_args"].get("model_path", None)
    #     model_path = model.path
    #     assert model_path, "model_path should not be None."
    #     model_path = _get_path(model_path)
    #     lwq_handles = register_weight_hooks(
    #         model, model_path, device=self.device, clean_weight=True, saved_path=LWQ_WORKSPACE
    #     )

    weight_config = {}
    for (op_type, op_name), op_config in configs_mapping.items():
        if op_config.weight_dtype == "fp32":
            continue
        else:
            weight_config[op_name] = {
                "wbits": op_config.weight_bits,
                "group_size": op_config.weight_group_size,
                "sym": op_config.weight_sym,
                "percdamp": op_config.percdamp,
                "act_order": op_config.act_order,
                "block_size": op_config.block_size,
            }
            nsamples = op_config.nsamples
            use_max_length = op_config.use_max_length
            pad_max_length = op_config.pad_max_length
            device = op_config.device

    if use_max_length and op_config.pad_max_length == 2048:
        logger.warning(
            "You choose to use unified sequence length for calibration, \
        but you have not set length value. Default sequence length is 2048 and this might cause inference error!"
        )

    return weight_config, nsamples, use_max_length, pad_max_length, device


@register_algo(name=GPTQ)
def gptq_quantize_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], GPTQConfig], dataloader, *args, **kwargs
) -> torch.nn.Module:
    logger.info("quantizing with the GPTQ algorithm")
    weight_config, nsamples, use_max_length, pad_max_length, device = gptq_config_mapping(configs_mapping)
    from neural_compressor.torch.algorithms.gptq import apply_gptq_quantize

    model, quantization_perm = apply_gptq_quantize(
        model=model,
        weight_config=weight_config,
        dataloader=dataloader,
        nsamples=nsamples,
        use_max_length=use_max_length,
        pad_max_length=pad_max_length,
        device=device,
        layer_wise=False,
        model_path=None,
    )
    # Assign the gptq config as an attribute of model
    model._gptq_quantization_perm = quantization_perm
    return model
