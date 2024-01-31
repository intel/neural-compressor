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

# TODO: remove it before merge !!!
import torch
from config import convert_offical_config_into_hqq_config
from core import HQQLinear

MB = 1024 * 1024


from utility import get_tensor_size, see_cuda_memory_usage


def get_float_linear_mod_size(mod):
    return get_tensor_size([mod.weight, mod.bias])


def new_hqq_api(mod, quant_config_offical):
    # see_cuda_memory_usage("Before Create the hqq_linear")
    float_size = get_float_linear_mod_size(mod)
    hqq_quant_config = convert_offical_config_into_hqq_config(quant_config_offical)
    hqqlinear = HQQLinear.from_float(mod, hqq_quant_config)
    del mod
    print(f"Create hqq_linear with size ({hqqlinear.get_size()/ MB}), the float mod size is {float_size/MB}")
    ratio = (hqqlinear.get_size() / float_size) / float_size
    if ratio > 0.1:
        print("Warning: hqq_linear size is larger more than 10% than float mod size, ratio: {ratio}")
    # see_cuda_memory_usage("After created the hqq_linear")
    return hqqlinear
