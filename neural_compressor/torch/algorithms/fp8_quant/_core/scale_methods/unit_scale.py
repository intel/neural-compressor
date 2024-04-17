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

import functools

import torch

from ..common import *
from ..fp_utils import *


def set_linear_without_scale(mod, scales, params):
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    input_config = [
        quantdequant_config(None, functools.partial(cast_to_fp8_fcn, dtype=lp_dtype), None, None, lp_dtype, hp_dtype)
    ]
    output_config = quantdequant_config(
        None, None, None, functools.partial(cast_fcn, dtype=hp_dtype), lp_dtype, hp_dtype
    )
    weight_config = quantdequant_config(
        None, functools.partial(cast_to_fp8_fcn, dtype=lp_dtype), None, None, lp_dtype, hp_dtype
    )
    params_config = {"weight": weight_config}
    config = module_config(input_config, output_config, params_config)
    return config


def set_matmul_without_scale(mod, scales, params):
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    input_config = [
        quantdequant_config(None, functools.partial(cast_to_fp8_fcn, dtype=lp_dtype), None, None, lp_dtype, hp_dtype)
        for _ in range(2)
    ]
    output_config = quantdequant_config(
        None, None, None, functools.partial(cast_fcn, dtype=hp_dtype), lp_dtype, hp_dtype
    )
    config = module_config(input_config, output_config, {})
    return config


def set_kv_cache_without_scale(mod, scales, params):
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    input_config = quantdequant_config(
        None, functools.partial(cast_to_fp8_fcn, dtype=lp_dtype), None, None, lp_dtype, hp_dtype
    )
    output_config = quantdequant_config(
        None, None, None, functools.partial(cast_from_fp8_fcn, dtype=hp_dtype), lp_dtype, hp_dtype
    )
    config = module_config(input_config, output_config)
    return config


def linear_unit_scale_scales(mod, measurement, params):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    weight_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    output_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    return module_config((input_scale,), output_scale, {"weight": weight_scale})


def matmul_unit_scale_scales(mod, measurement, params):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = (torch.tensor(1.0, dtype=hp_dtype, device=device), torch.tensor(1.0, dtype=hp_dtype, device=device))
    output_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    return module_config(input_scale, output_scale, {})


def set_kv_cache_unit_scale(mod, scales, params):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = (torch.tensor(1.0, dtype=hp_dtype, device=device),)
    output_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    return module_config(input_scale, output_scale)
