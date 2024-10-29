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

import torch

from ..common import *
from ..fp_utils import *


def linear_single_scale_scales(mod, measurement, params, scale=1.0):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = torch.tensor(scale, dtype=hp_dtype, device=device)
    weight_scale = torch.tensor(scale, dtype=hp_dtype, device=device)
    output_scale = torch.tensor(scale, dtype=hp_dtype, device=device)
    return ModuleConfig((input_scale,), (output_scale,), {"weight": weight_scale})


def linear_hw_aligned_single_scale_scales(mod, measurement, params):
    device_for_scales = get_device_type_for_scales(mod)
    hw_aligned_single_scale = FP8_143_SCALES[device_for_scales][0]
    return linear_single_scale_scales(mod, measurement, params, hw_aligned_single_scale)


def fsdpa_single_scale_scales(mod, measurement, params, scale=1.0):
    device = torch.device("hpu")
    hp_dtype = torch.float32  #  params["hp_dtype"]
    q_scale = torch.tensor(scale, dtype=hp_dtype, device=device)
    k_scale = torch.tensor(scale, dtype=hp_dtype, device=device)
    v_scale = torch.tensor(scale, dtype=hp_dtype, device=device)
    softmax_scale = torch.tensor(scale, dtype=hp_dtype, device=device)
    input_scale = (q_scale, k_scale, v_scale, softmax_scale)
    output_scale = (torch.tensor(scale, dtype=hp_dtype, device=device),)
    return ModuleConfig(input_scale, output_scale, {})


def fsdpa_hw_aligned_single_scale_scales(mod, measurement, params):
    device_for_scales = get_device_type_for_scales(mod)
    hw_aligned_single_scale = FP8_143_SCALES[device_for_scales][0]
    return fsdpa_single_scale_scales(mod, measurement, params, hw_aligned_single_scale)


def matmul_single_scale_scales(mod, measurement, params, scale=1.0):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = (
        torch.tensor(1.0, dtype=hp_dtype, device=device),
        torch.tensor(1.0, dtype=hp_dtype, device=device),
    )
    output_scale = (torch.tensor(1.0, dtype=hp_dtype, device=device),)
    return ModuleConfig(input_scale, output_scale, {})


def matmul_hw_aligned_single_scale_scales(mod, measurement, params):
    device_for_scales = get_device_type_for_scales(mod)
    hw_aligned_single_scale = FP8_143_SCALES[device_for_scales][0]
    return matmul_single_scale_scales(mod, measurement, params, hw_aligned_single_scale)


def softmax_single_scale_scales(mod, measurement, params, scale=1.0):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = (torch.tensor(scale, dtype=hp_dtype, device=device),)
    output_scale = (torch.tensor(scale, dtype=hp_dtype, device=device),)
    return ModuleConfig(input_scale, output_scale)


def softmax_hw_aligned_single_scale_scales(mod, measurement, params):
    device_for_scales = get_device_type_for_scales(mod)
    hw_aligned_single_scale = FP8_143_SCALES[device_for_scales][0]
    return softmax_single_scale_scales(mod, measurement, params, hw_aligned_single_scale)


def kv_cache_single_scale_scales(mod, measurement, params, scale=1.0):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = (torch.tensor(scale, dtype=hp_dtype, device=device),)
    output_scale = (torch.tensor(scale, dtype=hp_dtype, device=device),)
    return ModuleConfig(input_scale, output_scale)


def kv_cache_hw_aligned_single_scale_scales(mod, measurement, params):
    device_for_scales = get_device_type_for_scales(mod)
    hw_aligned_single_scale = FP8_143_SCALES[device_for_scales][0]
    return kv_cache_single_scale_scales(mod, measurement, params, hw_aligned_single_scale)
