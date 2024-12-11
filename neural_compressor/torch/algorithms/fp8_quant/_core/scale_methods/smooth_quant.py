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
from tqdm import tqdm

from ..common import *
from ..fp_utils import *

def linear_smoothquant_weights_opt_pow2_scales(mod, measurement, params, device = torch.device("hpu")):
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    device_type = get_device_type_for_scales(mod)
    fullscale = get_fullscale(lp_dtype, device_type)
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    alpha = params["alpha"]
    transformed_weight_scales = params["transformed_weight_scales"]
    input_range = torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device)
    weight_range_in_ch = torch.max(torch.abs(mod.weight), dim=0)[0].reshape([-1, 1])
    input_scale = calc_maxabs_scale(input_range, fullscale, input_backoff)
    weight_scale_in_ch = calc_maxabs_scale(weight_range_in_ch, fullscale, weight_backoff)
    input_scale = (input_scale**alpha) / (weight_scale_in_ch ** (1 - alpha))
    input_scale = scale_to_pow2(input_scale)
    weight_scale_in_ch = 1 / input_scale
    trans_weight = scale_fcn(mod.weight, weight_scale_in_ch.reshape([1, -1]))
    trans_weight_range_out_ch = torch.max(torch.abs(trans_weight), dim=1)[0].reshape([-1, 1])
    trans_weight_maxabs_scale_out_ch = calc_maxabs_scale(trans_weight_range_out_ch, fullscale, weight_backoff)
    trans_weight_maxabs_scale_out_ch = scale_to_pow2(trans_weight_maxabs_scale_out_ch)
    trans_weight_scale_out_ch = torch.zeros(mod.weight.shape[0])
    for k in tqdm(range(trans_weight_scale_out_ch.shape[0])):
        trans_weight_scale_out_ch[k] = mmse_scale(
            trans_weight[k, :],
            [s * trans_weight_maxabs_scale_out_ch[k] for s in transformed_weight_scales],
            lp_dtype,
            hp_dtype,
        )
    weight_scale_out_ch = scale_to_pow2(trans_weight_scale_out_ch)
    output_scale = torch.tensor(weight_scale_out_ch, dtype=hp_dtype, device=device)
    return ModuleConfig(
        (input_scale.flatten(),),
        (output_scale.flatten(),),
        {"weight": {0: weight_scale_out_ch.flatten(), 1: weight_scale_in_ch.flatten()}},
    )


def linear_smoothquant_weights_maxabs_pow2_scales(mod, measurement, params, device = torch.device("hpu")):
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    device_type = get_device_type_for_scales(mod)
    fullscale = get_fullscale(lp_dtype, device_type)
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    alpha = params["alpha"]
    input_range = torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device)
    weight_range_in_ch = torch.max(torch.abs(mod.weight), dim=0)[0].reshape([-1, 1])
    input_scale = calc_maxabs_scale(input_range, 1.0, 1.0)
    weight_scale_in_ch = calc_maxabs_scale(weight_range_in_ch, 1.0, 1.0)
    input_scale = (input_scale**alpha) / (weight_scale_in_ch ** (1 - alpha))
    input_scale = scale_to_pow2(input_scale)
    input_range_post = input_range / input_scale
    input_scale_post = calc_maxabs_scale(input_range_post.max(), fullscale, input_backoff)
    input_scale_post = scale_to_pow2(input_scale_post)
    input_scale = input_scale * input_scale_post
    weight_scale_in_ch = 1 / input_scale
    trans_weight = scale_fcn(mod.weight, weight_scale_in_ch.reshape([1, -1]))
    trans_weight_range_out_ch = torch.max(torch.abs(trans_weight), dim=1)[0].reshape([-1, 1])
    trans_weight_maxabs_scale_out_ch = calc_maxabs_scale(trans_weight_range_out_ch, fullscale, weight_backoff)
    trans_weight_maxabs_scale_out_ch = scale_to_pow2(trans_weight_maxabs_scale_out_ch)
    weight_scale_out_ch = scale_to_pow2(trans_weight_maxabs_scale_out_ch)
    output_scale = torch.tensor(weight_scale_out_ch, dtype=hp_dtype, device=device)
    return ModuleConfig(
        (input_scale.flatten(),),
        (output_scale.flatten(),),
        {"weight": {0: weight_scale_out_ch.flatten(), 1: weight_scale_in_ch.flatten()}},
    )


def linear_weaksmoothquant_weights_maxabs_pow2_scales(mod, measurement, params, device = torch.device("hpu")):
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    device_type = get_device_type_for_scales(mod)
    fullscale = get_fullscale(lp_dtype, device_type)
    input_backoff = params["input_backoff"]
    weight_backoff = params["weight_backoff"]
    alpha = params["alpha"]
    input_range = torch.tensor(measurement.inputs[0], dtype=hp_dtype, device=device).max().clamp(min=1e-5)
    input_range_mid = input_range.max() / torch.sqrt(input_range.max() / input_range.min().clamp(min=1e-5))
    input_scale_pcs = calc_maxabs_scale(input_range.clamp(min=1e-5), input_range_mid, 1.0).clamp(min=1e-5)
    weight_range_in_ch = torch.max(torch.abs(mod.weight), dim=0)[0].reshape([-1, 1]).clamp(min=1e-5)
    weight_range_in_ch_mid = weight_range_in_ch.max() / torch.sqrt(
        weight_range_in_ch.max() / weight_range_in_ch.min().clamp(min=1e-5)
    ).clamp(min=1e-5)
    weight_scale_pcs = calc_maxabs_scale(weight_range_in_ch.clamp(min=1e-5), weight_range_in_ch_mid, 1.0).clamp(
        min=1e-5
    )

    input_scale = ((input_scale_pcs**alpha) / (weight_scale_pcs ** (1 - alpha))).clamp(min=1e-5)
    input_scale = scale_to_pow2(input_scale)
    input_scale_post = calc_maxabs_scale((input_range / input_scale).max(), fullscale, input_backoff)
    input_scale_post = scale_to_pow2(input_scale_post)

    weight_scale_in_ch = torch.ones([mod.weight.shape[1], 1], dtype=hp_dtype, device=device) * (1 / input_scale)

    trans_weight = scale_fcn(mod.weight, weight_scale_in_ch.reshape([1, -1]))
    weight_range_out_ch = torch.max(torch.abs(trans_weight), dim=1)[0].reshape([-1, 1])

    weight_maxabs_scale_out_ch = calc_maxabs_scale(weight_range_out_ch, fullscale, weight_backoff)
    weight_maxabs_scale_out_ch = scale_to_pow2(weight_maxabs_scale_out_ch)
    output_scale = weight_maxabs_scale_out_ch * input_scale_post
    return ModuleConfig(
        (input_scale.flatten() * input_scale_post,),
        (output_scale.flatten(),),
        {
            "weight": {
                0: weight_maxabs_scale_out_ch.flatten(),
                1: weight_scale_in_ch.flatten(),
            }
        },
    )
