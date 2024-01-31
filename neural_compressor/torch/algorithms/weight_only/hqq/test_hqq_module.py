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

from copy import deepcopy

import pytest
import torch
from config import HQQModuleConfig, QTensorConfig
from core import HQQLinear, HQQTensorHandle
from utility import compare_two_tensor, is_divisible

######################
#### Test
#####################


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
        weight=QTensorConfig(**hqq_offical_config["weight_quant_params"]),
        scale=QTensorConfig(**hqq_offical_config["scale_quant_params"])
        if hqq_offical_config["scale_quant_params"] is not None
        else None,
        zero=QTensorConfig(**hqq_offical_config["zero_quant_params"])
        if hqq_offical_config["zero_quant_params"] is not None
        else None,
    )
    print(f"[create_hqq_quant_config_from_hqq_official_api] hqq_quant_config: {hqq_quant_config}")
    print(f"[create_hqq_quant_config_from_hqq_official_api] hqq_offical_config: {hqq_offical_config}")
    return hqq_quant_config, hqq_offical_config


def common_test(nbits=4, group_size=64, quant_zero=True, quant_scale=False, scale_quant_group_size=128, device=None):
    hqq_quant_config, hqq_offical_config = create_hqq_quant_config_from_hqq_official_api(
        nbits=nbits,
        group_size=group_size,
        quant_zero=quant_zero,
        quant_scale=quant_scale,
        scale_quant_group_size=scale_quant_group_size,
    )
    test_on_cuda = "cuda" in str(device)
    in_features = 64
    out_features = 128
    float_linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
    float_linear.to(device)
    float_linear_copy = deepcopy(float_linear)

    hqq_linear = HQQLinear.from_float(float_linear, quant_config=hqq_quant_config)

    from hqq.core.common.modules import HQQLinear as HQQLinear_official

    if test_on_cuda:
        from hqq.core.common.config import option

        option.use_cuda = True
        option.use_half = True
        from hqq.core.common.cuda_utils import see_memory_usage

        see_memory_usage("At the end of test")
    hqq_linear_official = HQQLinear_official(float_linear_copy, quant_config=hqq_offical_config)
    if test_on_cuda:
        hqq_linear_official.cuda(device_n=0)

    compare_two_tensor(hqq_linear.q_weight.val, hqq_linear_official.W_q, msg="The quantized weight")

    if quant_scale:
        compare_two_tensor(
            hqq_linear.q_weight.scale.val,
            hqq_linear_official.meta["scale_q"],
            msg="The quantized scale",
        )
    else:
        compare_two_tensor(hqq_linear.q_weight.scale, hqq_linear_official.meta["scale"], msg="float scale")
    if quant_zero:
        compare_two_tensor(
            hqq_linear.q_weight.zero.val,
            hqq_linear_official.meta["zero_q"],
            msg="The quantized zero",
        )
    else:
        compare_two_tensor(hqq_linear.q_weight.zero, hqq_linear_official.meta["zero"], msg="float zero")

    input = torch.randn(1, in_features)
    input = input.to(device)
    float_output = float_linear(input)
    print(hqq_linear.q_weight)
    input_half = deepcopy(input).half()
    hqq_output = hqq_linear(input_half)
    print(hqq_linear.q_weight)
    hqq_output_2 = hqq_linear(input_half)
    print(hqq_linear.q_weight)
    hqq_offical_output = hqq_linear_official(input_half)
    del float_linear, hqq_linear, hqq_linear_official
    del float_output, hqq_output, hqq_output_2, hqq_offical_output
    if test_on_cuda:
        see_memory_usage("At the end of test")


OS_ACCELERATOR = None


def force_set_accelerator_to_cpu():
    import os

    OS_ACCELERATOR = os.environ.get("ACCELERATOR", None)
    os.environ["ACCELERATOR"] = "cpu"


def revert_force_set_accelerator_to_cpu():
    import os

    if OS_ACCELERATOR is not None:
        os.environ["ACCELERATOR"] = OS_ACCELERATOR
    else:
        del os.environ["ACCELERATOR"]


@pytest.mark.parametrize(
    "nbits, group_size, quant_zero, quant_scale, scale_quant_group_size",
    [
        (4, 64, True, False, 128),
        (4, 64, False, False, 128),
        (4, 64, True, True, 128),
        (4, 64, False, True, 128),
        (8, 64, True, False, 128),
        (8, 64, False, False, 128),
        (8, 64, True, True, 128),
        (8, 64, False, True, 128),
        (4, 64, True, False, 64),
        (4, 64, False, False, 64),
        (4, 64, True, True, 64),
        (4, 64, False, True, 64),
    ],
)
def test_api_cpu(
    nbits,
    group_size,
    quant_zero,
    quant_scale,
    scale_quant_group_size,
):
    force_set_accelerator_to_cpu()
    common_test(
        nbits=nbits,
        group_size=group_size,
        quant_zero=quant_zero,
        quant_scale=quant_scale,
        scale_quant_group_size=scale_quant_group_size,
        device=torch.device("cpu"),
    )
    revert_force_set_accelerator_to_cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "nbits, group_size, quant_zero, quant_scale, scale_quant_group_size",
    [
        (4, 64, True, False, 128),
        (4, 64, False, False, 128),
        (4, 64, True, True, 128),
        (4, 64, False, True, 128),
        (8, 64, True, False, 128),
        (8, 64, False, False, 128),
        (8, 64, True, True, 128),
        (8, 64, False, True, 128),
        (4, 64, True, False, 64),
        (4, 64, False, False, 64),
        (4, 64, True, True, 64),
        (4, 64, False, True, 64),
    ],
)
def test_api_cuda(
    nbits,
    group_size,
    quant_zero,
    quant_scale,
    scale_quant_group_size,
):
    common_test(
        nbits=nbits,
        group_size=group_size,
        quant_zero=quant_zero,
        quant_scale=quant_scale,
        scale_quant_group_size=scale_quant_group_size,
        device=torch.device("cuda:0"),
    )


# Test single case
# force_set_accelerator_to_cpu()
# common_test(
#     nbits=4, group_size=64, quant_zero=False, quant_scale=False, scale_quant_group_size=128, device=torch.device(device="cpu")
# )
# revert_force_set_accelerator_to_cpu()
