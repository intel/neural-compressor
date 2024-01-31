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

# !!! Remove it before merge !!!!

from copy import deepcopy

import pytest
import torch
from config import HQQModuleConfig, QuantTensorConfig
from core import HQQLinear, HQQTensorHandle
from utility import compare_two_tensor, is_divisible

######################
#### Test
#####################
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.use_deterministic_algorithms(True)
import numpy as np

np.random.seed(0)


def hqq_base_quant_config(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=False,
    scale_quant_group_size=128,
):
    assert nbits in HQQTensorHandle.SUPPORTED_BITS, "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
    if group_size is not None:
        assert is_divisible(group_size, 8), "Invalid group_size param: the value should be a multiple of 8."
    weight_quant_params = {
        "nbits": nbits,
        "channel_wise": True,
        "group_size": group_size,
        "optimize": True,
        "round_zero": True if nbits == 4 else False,
    }
    scale_quant_params = (
        {
            "nbits": 8,
            "channel_wise": True,
            "group_size": scale_quant_group_size,
            "optimize": False,
        }
        if (quant_scale)
        else None
    )
    zero_quant_params = (
        {"nbits": 8, "channel_wise": False, "group_size": None, "optimize": False} if (quant_zero) else None
    )
    return {
        "weight_quant_params": weight_quant_params,
        "scale_quant_params": scale_quant_params,
        "zero_quant_params": zero_quant_params,
    }


# Alias: follow similar Auto-GPTQ naming
BaseQuantizeConfig = hqq_base_quant_config


def create_hqq_module(float_linear, config):
    hqq_linear = HQQLinear.from_float(float_linear, quant_config=config)
    return hqq_linear


def create_hqq_module_official(float_linear, config):
    from hqq.core.common.modules import HQQLinear as HQQLinear_official

    hqq_linear_official = HQQLinear_official(float_linear, config)
    return hqq_linear_official


def create_hqq_quant_config_from_hqq_official_api(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=False,
    scale_quant_group_size=128,
):
    hqq_offical_config = hqq_base_quant_config(
        nbits=nbits,
        group_size=group_size,
        quant_zero=quant_zero,
        quant_scale=quant_scale,
        scale_quant_group_size=scale_quant_group_size,
    )
    hqq_quant_config = HQQModuleConfig(
        weight_quant_config=QuantTensorConfig(**hqq_offical_config["weight_quant_params"]),
        scale_quant_config=QuantTensorConfig(**hqq_offical_config["scale_quant_params"])
        if hqq_offical_config["scale_quant_params"] is not None
        else None,
        zero_quant_config=QuantTensorConfig(**hqq_offical_config["zero_quant_params"])
        if hqq_offical_config["zero_quant_params"] is not None
        else None,
    )
    print(f"[create_hqq_quant_config_from_hqq_official_api] hqq_quant_config: {hqq_quant_config}")
    print(f"[create_hqq_quant_config_from_hqq_official_api] hqq_offical_config: {hqq_offical_config}")
    return hqq_quant_config, hqq_offical_config


import time

from torch.utils.benchmark import Timer


@torch.no_grad()
def benchmark_cpu(f, config):
    start = time.time()
    times = 50
    for _ in range(times):
        float_linear = torch.nn.Linear(in_features=4096, out_features=4096)
        f(float_linear, config)
        del float_linear
    dur = time.time() - start
    return {"time(s) ": dur / times, "total time(s) ": dur}


float_linear = torch.nn.Linear(in_features=256, out_features=1024)
hqq_quant_config, hqq_offical_config = create_hqq_quant_config_from_hqq_official_api()
res_hqq_official = benchmark_cpu(create_hqq_module_official, hqq_offical_config)

# res_hqq = benchmark_cpu(create_hqq_module, hqq_quant_config)
# print(f"[benchmark_cpu] res_hqq: {res_hqq}")
print(f"[benchmark_cpu] res_hqq_official: {res_hqq_official}")


########################################
## Benchmarking the memory usage
#########################################

from tqdm import tqdm

MB = 2**20


@torch.no_grad()
def benchmark_cuda(f, config):
    from utility import see_cuda_memory_usage

    in_feats = 4096
    out_feats = 32000

    runs = 10
    float_linear_list = []
    hqq_linear_module_lst = []
    for i in tqdm(range(runs)):
        see_cuda_memory_usage("Before create float_linear")
        float_linear = torch.nn.Linear(in_features=in_feats, out_features=out_feats, device=torch.device(0))
        see_cuda_memory_usage("After create float_linear")
        hqq_linear = f(float_linear, config)
        hqq_size = hqq_linear.get_size()
        print(f"hqq size: {hqq_size / MB}")
        see_cuda_memory_usage(f"After create {i + 1} hqq_module")
        float_linear_list.append(float_linear)
        hqq_linear_module_lst.append(hqq_linear)
        # hqq_linear_module_lst.append(hqq_linear)
        torch.cuda.empty_cache()


# float_linear = torch.nn.Linear(in_features=256, out_features=1024)
hqq_quant_config, hqq_offical_config = create_hqq_quant_config_from_hqq_official_api()
res_hqq_mem = benchmark_cuda(create_hqq_module, hqq_quant_config)
