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


def linear_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales(mod, measurement, params):
    config = get_hqt_config(mod).cfg
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    input_scale = calc_maxabs_scale(
        torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    weight_scale = calc_maxabs_scale(
        torch.max(torch.abs(mod.weight.detach())).to(dtype=hp_dtype, device=device), fullscale, weight_backoff
    )
    input_scale = scale_to_pow2_hw(input_scale, device_type=config["device_type"])
    weight_scale = scale_to_pow2_hw(weight_scale, device_type=config["device_type"])
    output_scale = input_scale * weight_scale
    return module_config((input_scale,), output_scale, {"weight": weight_scale})


def linear_act_maxabs_pts_weight_maxabs_pts_pow2_scales(mod, measurement, params):
    config = get_hqt_config(mod).cfg
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    input_scale = calc_maxabs_scale(
        torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    weight_scale = calc_maxabs_scale(
        torch.max(torch.abs(mod.weight.detach())).to(dtype=hp_dtype, device=device), fullscale, weight_backoff
    )
    input_scale = scale_to_pow2(input_scale)
    weight_scale = scale_to_pow2(weight_scale)
    output_scale = input_scale * weight_scale
    return module_config((input_scale,), output_scale, {"weight": weight_scale})


def matmul_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales(mod, measurement, params):
    config = get_hqt_config(mod).cfg
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    input_scale = [
        calc_maxabs_scale(torch.tensor(x, dtype=hp_dtype, device=device).max(), fullscale, input_backoff)
        for x in measurement.inputs
    ]
    input_scale = [scale_to_pow2_hw(x, device_type=config["device_type"]) for x in input_scale]
    output_scale = input_scale[0] * input_scale[1]
    return module_config(input_scale, output_scale, {})


def matmul_act_maxabs_pts_weight_maxabs_pts_pow2_scales(mod, measurement, params):
    config = get_hqt_config(mod).cfg
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    input_scale = [
        calc_maxabs_scale(torch.tensor(x, dtype=hp_dtype, device=device).max(), fullscale, input_backoff)
        for x in measurement.inputs
    ]
    input_scale = [scale_to_pow2(x) for x in input_scale]
    output_scale = input_scale[0] * input_scale[1]
    return module_config(input_scale, output_scale, {})


def linear_act_maxabs_pts_weight_opt_pts_pow2_scales(mod, measurement, params):
    config = get_hqt_config(mod).cfg
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    scales = params["weight_scales"]
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    input_scale = calc_maxabs_scale(
        torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    weight_scale = mmse_scale(mod.weight, scales, lp_dtype, hp_dtype)
    input_scale = scale_to_pow2(input_scale)
    weight_scale = scale_to_pow2(weight_scale)
    output_scale = input_scale * weight_scale
    return module_config((input_scale,), output_scale, {"weight": weight_scale})


def linear_act_maxabs_pts_weight_opt_pts_hw_scales(mod, measurement, params):
    config = get_hqt_config(mod).cfg
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    scales = params["weight_scales"]
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    input_scale = calc_maxabs_scale(
        torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    weight_scale = mmse_scale(mod.weight, scales, lp_dtype, hp_dtype)
    input_scale = scale_to_pow2_hw(input_scale, device_type=config["device_type"])
    weight_scale = scale_to_pow2_hw(weight_scale, device_type=config["device_type"])
    output_scale = input_scale * weight_scale
    return module_config((input_scale,), output_scale, {"weight": weight_scale})


def kv_cache_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales(mod, measurement, params):
    config = get_hqt_config(mod).cfg
    # calc the scale per layer tensor
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    input_scale = calc_maxabs_scale(
        torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    input_scale_list = [scale_to_pow2_hw(input_scale, device_type=config["device_type"])]
    output_scale = input_scale_list[0]  # output scale is same as the first input (current data) since range is same
    return module_config(input_scale_list, output_scale, {})


def kv_cache_act_maxabs_pts_pow2_weight_opt_pcs_pow2_scales(mod, measurement, params):
    # calc the scale per layer tensor
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    input_scale = calc_maxabs_scale(
        torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    input_scale_list = [scale_to_pow2(input_scale)]
    output_scale = input_scale_list[0]  # output scale is same as the first input (current data) since range is same
    return module_config(input_scale_list, output_scale, {})


def softmax_input_unit_output_maxabs_pts_hw_scales(mod, measurement, params):
    config = get_hqt_config(mod).cfg
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    output_scale = calc_maxabs_scale(
        torch.tensor(measurement.outputs, dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    output_scale = scale_to_pow2_hw(output_scale, device_type=config["device_type"])
    return module_config((), output_scale, {})


def linear_act_maxabs_pts_pow2_hw_weights_maxabs_pcs_pow2_scales(mod, measurement, params):
    config = get_hqt_config(mod).cfg
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    input_scale = calc_maxabs_scale(
        torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    input_scale = scale_to_pow2_hw(input_scale, device_type=config["device_type"])
    weight_scale_in_ch = torch.ones([mod.weight.shape[1], 1], dtype=hp_dtype, device=device)

    weight_range_out_ch = torch.max(torch.abs(mod.weight), dim=1)[0].reshape([-1, 1])
    weight_maxabs_scale_out_ch = calc_maxabs_scale(weight_range_out_ch, fullscale, weight_backoff)
    weight_maxabs_scale_out_ch = scale_to_pow2(weight_maxabs_scale_out_ch)
    output_scale = weight_maxabs_scale_out_ch * input_scale
    return module_config(
        (input_scale.flatten(),),
        output_scale.flatten(),
        {"weight": {0: weight_maxabs_scale_out_ch.flatten(), 1: weight_scale_in_ch.flatten()}},
    )


def linear_act_maxabs_pts_pow2_weights_maxabs_pcs_pow2_scales(mod, measurement, params):
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    input_scale = calc_maxabs_scale(
        torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    input_scale = scale_to_pow2(input_scale)
    weight_scale_in_ch = torch.ones([mod.weight.shape[1], 1], dtype=hp_dtype, device=device)

    weight_range_out_ch = torch.max(torch.abs(mod.weight), dim=1)[0].reshape([-1, 1])
    weight_maxabs_scale_out_ch = calc_maxabs_scale(weight_range_out_ch, fullscale, weight_backoff)
    weight_maxabs_scale_out_ch = scale_to_pow2(weight_maxabs_scale_out_ch)
    output_scale = weight_maxabs_scale_out_ch * input_scale
    return module_config(
        (input_scale.flatten(),),
        output_scale.flatten(),
        {"weight": {0: weight_maxabs_scale_out_ch.flatten(), 1: weight_scale_in_ch.flatten()}},
    )


def linear_act_maxabs_pts_pow2_hw_weights_opt_pcs_pow2_scales(mod, measurement, params):
    config = get_hqt_config(mod).cfg
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    weight_scales = params["weight_scales"]
    input_scale = calc_maxabs_scale(
        torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    input_scale = scale_to_pow2_hw(input_scale, device_type=config["device_type"])
    weight_scale_in_ch = torch.ones([mod.weight.shape[1], 1], dtype=hp_dtype, device=device)

    weight_range_out_ch = torch.max(torch.abs(mod.weight), dim=1)[0].reshape([-1, 1])
    weight_maxabs_scale_out_ch = calc_maxabs_scale(weight_range_out_ch, fullscale, weight_backoff)
    weight_maxabs_scale_out_ch = scale_to_pow2(weight_maxabs_scale_out_ch)
    weight_opt_scale_out_ch = mmse_scale_multi(
        torch.transpose(mod.weight, 0, 1), weight_maxabs_scale_out_ch.squeeze(), weight_scales, lp_dtype, hp_dtype
    ).unsqueeze(1)
    weight_maxabs_scale_out_ch = weight_opt_scale_out_ch
    weight_maxabs_scale_out_ch = scale_to_pow2(weight_maxabs_scale_out_ch)  # should be power of 2, just making sure
    output_scale = weight_maxabs_scale_out_ch * input_scale
    return module_config(
        (input_scale.flatten(),),
        output_scale.flatten(),
        {"weight": {0: weight_maxabs_scale_out_ch.flatten(), 1: weight_scale_in_ch.flatten()}},
    )


def linear_act_maxabs_pts_pow2_weights_opt_pcs_pow2_scales(mod, measurement, params):
    device = torch.device("hpu")
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    fullscale = MAX_RANGE[lp_dtype]
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    weight_scales = params["weight_scales"]
    input_scale = calc_maxabs_scale(
        torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max(), fullscale, input_backoff
    )
    input_scale = scale_to_pow2(input_scale)
    weight_scale_in_ch = torch.ones([mod.weight.shape[1], 1], dtype=hp_dtype, device=device)

    weight_range_out_ch = torch.max(torch.abs(mod.weight), dim=1)[0].reshape([-1, 1])
    weight_maxabs_scale_out_ch = calc_maxabs_scale(weight_range_out_ch, fullscale, weight_backoff)
    weight_maxabs_scale_out_ch = scale_to_pow2(weight_maxabs_scale_out_ch)
    weight_opt_scale_out_ch = mmse_scale_multi(
        torch.transpose(mod.weight, 0, 1), weight_maxabs_scale_out_ch.squeeze(), weight_scales, lp_dtype, hp_dtype
    ).unsqueeze(1)
    weight_maxabs_scale_out_ch = weight_opt_scale_out_ch
    weight_maxabs_scale_out_ch = scale_to_pow2(weight_maxabs_scale_out_ch)  # should be power of 2, just making sure
    output_scale = weight_maxabs_scale_out_ch * input_scale
    return module_config(
        (input_scale.flatten(),),
        output_scale.flatten(),
        {"weight": {0: weight_maxabs_scale_out_ch.flatten(), 1: weight_scale_in_ch.flatten()}},
    )