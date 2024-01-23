#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

from neural_compressor.torch.utils import logger
from neural_compressor.torch.utils.utility import set_module

from .utility import quant_tensor, search_clip


def rtn_quantize(
    model,
    num_bits=4,
    group_size=32,
    scheme="asym",
    quantile=1.0,
    weight_config={},
    return_int=False,
    data_type="int",
    enable_full_range=False,
    enable_mse_search=False,
    group_dim=1,
    **kwargs,
):
    """Quant the model with round to nearest method.

    Args:
        model: torch module
        num_bits: num bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to 32.
        scheme (str, optional): sym or asym. Defaults to "asym".
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        data_type (str, optional): select from int, nf4, fp4. Defaults to int.
        weight_config (dict, optional): specific layer wise configurations. Defaults to {}.
            For example,
                weight_config={
                    'fc2':
                        {
                            'dtype': 'int',
                            'bits': 4,
                            'group_size': 32,
                            'scheme': 'sym'
                            'gptq_perm': [1, 1, ...] # for gptq perm
                        }
                }
        return_int (bool, optional): Choose return fp32 or int32 model.
                                     Defaults to False.
        enable_full_range (bool, optional): Choose sym range whether use -2**(bits-1).
                                     Defaults to False.
        enable_mse_search (bool, optional):  Whether search clip range.
                                     Defaults to True.
        group_dim (int, optional):   0 means splitting output channel,
                                     1 means splitting input channel. Defaults to 1.

    Returns:
        model: fake quantized torch module
    """
    assert isinstance(model, torch.nn.Module), "only support torch module"
    supported_layers = ["Linear"]
    double_quant_dtype = kwargs.get("double_quant_dtype", "fp32")
    double_quant_config = {
        "double_quant": False if double_quant_dtype == "fp32" else True,
        "double_quant_dtype": double_quant_dtype,
        "double_quant_num_bits": kwargs.get("double_quant_num_bits", 8),
        "double_quant_scheme": kwargs.get("double_quant_scheme", "sym"),
        "double_quant_group_size": kwargs.get("double_quant_group_size", 256),
    }
    if return_int:
        compression_dtype = kwargs.get("compression_dtype", torch.int32)
        compression_dim = kwargs.get("compression_dim", 1)
        scale_dtype = kwargs.get("scale_dtype", torch.float32)
        device = kwargs.get("device", "cpu")
    for name, m in model.named_modules():
        if m.__class__.__name__ not in supported_layers:
            continue
        if name in weight_config:  # pragma: no cover
            data_type = weight_config[name].get("dtype", "int")
            num_bits = weight_config[name]["bits"]
            group_size = weight_config[name]["group_size"]
            scheme = weight_config[name]["scheme"]
            quantile = weight_config[name].get("quantile", 1.0)
        log_msg = (
            f"RTN quantization config: num_bits={num_bits}, group_size={group_size}, "
            + f"scheme={scheme}, quantile={quantile}"
        )
        if data_type != "int":
            log_msg += f", dtype={data_type}"
        elif scheme == "sym":  # nf4/fp4 is always [-7,7]
            log_msg += f", enable_full_range={enable_full_range}"
        if data_type == "fp32":
            continue
        logger.debug(f"RTN quantized module:{name, m}")
        logger.debug(log_msg)
        weight = m.weight.T if group_dim == 0 else m.weight
        if enable_mse_search:
            quantile = search_clip(m, num_bits, group_size, scheme, data_type, enable_full_range)
        if return_int:
            int_weight, scale, zp = quant_tensor(
                weight,
                num_bits,
                group_size,
                scheme,
                quantile,
                data_type=data_type,
                return_int=True,
                full_range=enable_full_range,
                **double_quant_config,
            )
            int_weight = int_weight.T if group_dim == 0 else int_weight
            scale = scale.T if group_dim == 0 else scale
            zp = zp.T if group_dim == 0 and zp is not None else zp
            from neural_compressor.torch.quantization.layers import WeightOnlyLinear

            new_module = WeightOnlyLinear(
                m.in_features,
                m.out_features,
                num_bits,
                group_size,
                dtype=data_type,
                zp=zp is not None,
                bias=m.bias is not None,
                compression_dtype=compression_dtype,
                compression_dim=compression_dim,
                scale_dtype=scale_dtype,
                device=device,
            )
            new_module.pack(int_weight, scale, zp, m.bias)
            if name == "":
                return new_module
            else:
                set_module(model, name, new_module)
        else:
            q_weight = quant_tensor(
                weight,
                num_bits,
                group_size,
                scheme,
                quantile,
                data_type=data_type,
                full_range=enable_full_range,
                **double_quant_config,
            )
            q_weight = q_weight.T if group_dim == 0 else q_weight
            m.weight.data.copy_(q_weight)
    return model


from neural_compressor.torch.quantization.config import RTNConfig


def apply_rtn_on_single_module(module: torch.nn.Module, quant_config: RTNConfig) -> torch.nn.Module:
    # TODO (Yi) remove it
    enable_full_range = quant_config.enable_full_range
    enable_mse_search = quant_config.enable_mse_search
    group_dim = quant_config.group_dim
    dtype = quant_config.weight_dtype
    num_bits = quant_config.weight_bits
    scheme = "sym" if quant_config.weight_sym else "asym"
    group_size = quant_config.weight_group_size
    return_int = quant_config.return_int
    double_quant_dtype = quant_config.double_quant_dtype
    double_quant_num_bits = quant_config.double_quant_bits
    double_quant_scheme = "sym" if quant_config.double_quant_sym else "asym"
    double_quant_group_size = quant_config.double_quant_group_size
    return rtn_quantize(
        module,
        num_bits,
        group_size,
        scheme,
        return_int=return_int,
        data_type=dtype,
        enable_full_range=enable_full_range,
        enable_mse_search=enable_mse_search,
        group_dim=group_dim,
        double_quant_dtype=double_quant_dtype,
        double_quant_scheme=double_quant_scheme,
        double_quant_num_bits=double_quant_num_bits,
        double_quant_group_size=double_quant_group_size,
    )
