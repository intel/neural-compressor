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

import numpy as np
import torch

from .._quant_common.quant_config import ScaleMethod, set_hqt_config, get_hqt_config
from ..utils.logger import logger
from .common import *
from .fp_utils import *
from .quant_dequant import *
from .scale_methods import *


def init_mod_config(mod, scales, params):
    scales_inv = invert_scales(scales)
    scale_format = get_hqt_config(mod).cfg["scale_format"]
    use_qdq = get_hqt_config(mod).cfg["use_qdq"]
    fake_quant = get_hqt_config(mod).cfg["fake_quant"]
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    return scales_inv, scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype


def init_input_config(scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant):
    if use_qdq or fake_quant:
        input_config = [QuantDequant(s_inv, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq) for s_inv in scales_inv.inputs]
    else:
        input_config = [QuantInput(s_inv, lp_dtype, hp_dtype, scale_format=scale_format) for s_inv in scales_inv.inputs]
    return input_config


def init_weight_config(scales, scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant):
    if use_qdq:
        # to ensure the weights to be loaded to the device in fp8
        weight_config = [QuantInput(scales_inv, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq),
                         DequantOutput(scales, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq)]
    elif fake_quant:
        weight_config = [QuantDequant(scales_inv, lp_dtype, hp_dtype, scale_format=scale_format)]
    else:
        weight_config = [QuantInput(scales_inv, lp_dtype, hp_dtype, scale_format=scale_format)]
    return weight_config


def matmul_scales_to_mod_config(mod, scales, params):
    scales_inv, scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = init_mod_config(mod, scales, params)
    input_config = init_input_config(scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
    # outputs as bf16, and descaled in gemm under PatchedMatmul, so no need to work here
    output_config = [QuantDequantNone(lp_dtype, hp_dtype, scale_format=scale_format)]
    config = ModuleConfig(input_config, output_config, {})
    return config


def fsdpa_scales_to_mod_config(mod, scales, params):
    scales_inv, scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = init_mod_config(mod, scales, params)
    input_config = init_input_config(scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
    output_config = [DequantOutput(scales.outputs[0], lp_dtype, hp_dtype, scale_format=scale_format)]
    config = ModuleConfig(input_config, output_config, {})
    return config


def linear_scales_to_mod_config(mod, scales, params):
    scales_inv, scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = init_mod_config(mod, scales, params)
    input_config = init_input_config(scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
    # outputs as bf16, and descaled in gemm under PatchedLinear, so no need to work here
    output_config = [QuantDequantNone(lp_dtype, hp_dtype, scale_format=scale_format)]

    if isinstance(scales_inv.params["weight"], (torch.Tensor, float)):
        weight_config = init_weight_config(scales.params["weight"], scales_inv.params["weight"], lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
    elif isinstance(scales_inv.params["weight"], dict):
        weight_scale_inv_out_ch = scales_inv.params["weight"][0]
        weight_scale_inv_in_ch = scales_inv.params["weight"][1]
        if isinstance(weight_scale_inv_out_ch, torch.Tensor):
            scale_inv = [weight_scale_inv_in_ch.reshape([1, -1]), weight_scale_inv_out_ch.reshape([-1, 1])]
        else:
            # TODO SW-169781: Handle here scalar weight for PCQ
            raise TypeError(f"Unknown weight scales type: {type(weight_scale_inv_out_ch)}.")
        weight_config = init_weight_config(scales, scale_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
    else:
        logger.error("Unknown weight scales format.")
    params_config = {"weight": weight_config}
    if hasattr(mod, "bias") and (getattr(mod, "bias") is not None):
        # In PatchedLinear the bias is added to the output of gemm.
        # The output is expected to be descaled and in bf16, so we don't need to touch the bias.
        bias_config = [QuantDequantNone(lp_dtype, hp_dtype)]
        params_config.update({"bias": bias_config})
    config = ModuleConfig(input_config, output_config, params_config)
    return config


def kv_cache_scales_to_mod_config(mod, scales, params):
    # how quant/dequant will be applied on layer tensors
    scales_inv, scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = init_mod_config(mod, scales, params)
    input_config = init_input_config(scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
    output_config = [DequantOutput(scales.outputs[0], lp_dtype, hp_dtype, scale_format=scale_format)]
    config = ModuleConfig(input_config, output_config)
    return config


def softmax_scales_to_mod_config(mod, scales, params):
    scales_inv, scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = init_mod_config(mod, scales, params)
    output_config = [DequantOutput(scales.outputs[0], lp_dtype, hp_dtype, scale_format=scale_format)]
    return ModuleConfig(None, output_config)


def load_layer_scales(mod, mod_name, config, mod_type_str, measurement, scales, scale_file,
                      scales_file_format, scales_obj, scaling_method, scale_config, save_file):
    module_type = mod_default_dict[mod_type_str].type
    logger.debug(
        "Preparing quantization functions for module %s module_type=%s",
        mod_name,
        module_type,
    )
    mod_extra_config = None
    if mod_name in scales or not config.cfg["use_stats_files"] or mod_name in measurement:
        if mod_name not in scales:
            logger.debug("Calculating scales for module %s", mod_name)
            layer_measure = measurement.get(mod_name, None)  # ModuleConfig of measurements
            # calculates scales for current module according to scalling_methods
            scales[mod_name] = scaling_method[module_type][0](mod, layer_measure, scale_config)  # ModuleConfig of scales
            if scale_file is not None:
                scales_obj[mod_name] = ModuleConfig(
                    **format_functions_rec((torch.Tensor, scales_file_format))(scales[mod_name].__dict__)
                )
                save_file = True
        # calculates QuantDequant config for current module according to scalling_methods
        mod_config = scaling_method[module_type][1](mod, scales[mod_name], scale_config)  # ModuleConfig of QuantDequant
        mod_extra_config = ModuleExtraConfig(
                mod_config.inputs,
                mod_config.outputs,
                mod_config.params,
                scales[mod_name],
                scale_config,
                )
    return mod_extra_config, save_file

scaling_methods = {
    "unit_scale": {
        "linear": (linear_single_scale_scales, linear_scales_to_mod_config),
        "matmul": (matmul_single_scale_scales, matmul_scales_to_mod_config),
        "softmax": (softmax_single_scale_scales, softmax_scales_to_mod_config),
        "kv_cache": (kv_cache_single_scale_scales, kv_cache_scales_to_mod_config),
        "fused_sdpa": (fsdpa_single_scale_scales, fsdpa_scales_to_mod_config),
    },
    "hw_aligned_single_scale": {
        "linear": (linear_hw_aligned_single_scale_scales, linear_scales_to_mod_config),
        "matmul": (matmul_hw_aligned_single_scale_scales, matmul_scales_to_mod_config),
        "softmax": (softmax_hw_aligned_single_scale_scales, softmax_scales_to_mod_config),
        "kv_cache": (kv_cache_hw_aligned_single_scale_scales, kv_cache_scales_to_mod_config),
        "fused_sdpa": (fsdpa_hw_aligned_single_scale_scales, fsdpa_scales_to_mod_config),
    },
    "act_maxabs_pts_weight_maxabs_pts_arbitrary": {
        "linear": (
            linear_act_maxabs_pts_weight_maxabs_pts_arbitrary_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_arbitrary_scales,
            matmul_scales_to_mod_config,
        ),
        "kv_cache": (
            kv_cache_act_maxabs_pts_weight_maxabs_pts_arbitrary_scales,
            kv_cache_scales_to_mod_config,
        ),
        "softmax": (
            softmax_input_unit_output_maxabs_pts_arbitrary_scales,
            softmax_scales_to_mod_config,
        ),
        "fused_sdpa": (
            fsdpa_act_maxabs_pts_weight_maxabs_pts_arbitrary_scales,
            fsdpa_scales_to_mod_config,
        ),
    },
    "act_maxabs_pts_weight_maxabs_pts_pow2_hw": {
        "linear": (
            linear_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            matmul_scales_to_mod_config,
        ),
        "kv_cache": (
            kv_cache_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            kv_cache_scales_to_mod_config,
        ),
        "softmax": (
            softmax_input_unit_output_maxabs_pts_hw_scales,
            softmax_scales_to_mod_config,
        ),
        "fused_sdpa": (
            fsdpa_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            fsdpa_scales_to_mod_config,
        ),
    },
    "act_maxabs_pts_weight_maxabs_pts_pow2": {
        "linear": (
            linear_act_maxabs_pts_weight_maxabs_pts_pow2_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_scales,
            matmul_scales_to_mod_config,
        ),
        "kv_cache": (
            kv_cache_act_maxabs_pts_pow2,
            kv_cache_scales_to_mod_config,
        ),
        "softmax": (
            softmax_input_unit_output_maxabs_pts_pow2,
            softmax_scales_to_mod_config,
        ),
        "fused_sdpa": (
            fsdpa_act_maxabs_pts_pow2_weight_maxabs_pts_pow2,
            fsdpa_scales_to_mod_config,
        ),
    },
    "act_maxabs_pts_pow2_hw_weights_maxabs_pcs_pow2": {
        "linear": (
            linear_act_maxabs_pts_pow2_hw_weights_maxabs_pcs_pow2_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            matmul_scales_to_mod_config,
        ),
        # kv_cache is pts as op in hw doesn't work in pcs
        "kv_cache": (
            kv_cache_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            kv_cache_scales_to_mod_config,
        ),
        "fused_sdpa": (
            fsdpa_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            fsdpa_scales_to_mod_config,
        ),
        "softmax": (
            softmax_input_unit_output_maxabs_pts_pow2,
            softmax_scales_to_mod_config,
        ),
    },
    "act_maxabs_pts_weight_opt_pts_pow2": {
        "linear": (
            linear_act_maxabs_pts_weight_opt_pts_pow2_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_scales,
            matmul_scales_to_mod_config,
        ),
        "softmax": (
            softmax_input_unit_output_maxabs_pts_pow2,
            softmax_scales_to_mod_config,
        ),
    },
    "act_maxabs_pts_weight_opt_pts_hw": {
        "linear": (
            linear_act_maxabs_pts_weight_opt_pts_hw_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            matmul_scales_to_mod_config,
        ),
        "softmax": (
            softmax_input_unit_output_maxabs_pts_hw_scales,
            softmax_scales_to_mod_config,
        ),
        "fused_sdpa": (
            fsdpa_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            fsdpa_scales_to_mod_config,
        ),
        "softmax": (
            softmax_input_unit_output_maxabs_pts_hw_scales,
            softmax_scales_to_mod_config,
        ),
    },
    "act_maxabs_pts_pow2_hw_weights_opt_pcs_pow2": {
        "linear": (
            linear_act_maxabs_pts_pow2_hw_weights_opt_pcs_pow2_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            matmul_scales_to_mod_config,
        ),
        # kv_cache is pts as op in hw doesn't work in pcs
        "kv_cache": (
            kv_cache_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            kv_cache_scales_to_mod_config,
        ),
        "fused_sdpa": (
            fsdpa_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            fsdpa_scales_to_mod_config,
        ),
        "softmax": (
            softmax_input_unit_output_maxabs_pts_hw_scales,
            softmax_scales_to_mod_config,
        ),
    },
    "act_maxabs_pts_pow2_weights_maxabs_pcs_pow2": {
        "linear": (
            linear_act_maxabs_pts_pow2_weights_maxabs_pcs_pow2_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_scales,
            matmul_scales_to_mod_config,
        ),
        # kv_cache is pts as op in hw doesn't work in pcs
        "kv_cache": (
            kv_cache_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            kv_cache_scales_to_mod_config,
        ),
        "fused_sdpa": (
            fsdpa_act_maxabs_pts_weight_maxabs_pts_pow2_scales,
            fsdpa_scales_to_mod_config,
        ),
        "softmax": (
            softmax_input_unit_output_maxabs_pts_pow2,
            softmax_scales_to_mod_config,
        ),
    },
    "act_maxabs_pts_pow2_weights_opt_pcs_pow2": {
        "linear": (
            linear_act_maxabs_pts_pow2_weights_opt_pcs_pow2_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_scales,
            matmul_scales_to_mod_config,
        ),
        # kv_cache is pts as op in hw doesn't work in pcs
        "kv_cache": (
            kv_cache_act_maxabs_pts_pow2,
            kv_cache_scales_to_mod_config,
        ),
        "fused_sdpa": (
            fsdpa_act_maxabs_pts_weight_maxabs_pts_pow2_scales,
            fsdpa_scales_to_mod_config,
        ),
        "softmax": (
            softmax_input_unit_output_maxabs_pts_pow2,
            softmax_scales_to_mod_config,
        ),
    },
    "smoothquant_weights_opt_pow2": {
        "linear": (
            linear_smoothquant_weights_opt_pow2_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            matmul_scales_to_mod_config,
        ),
    },
    "smoothquant_weights_maxabs_pow2": {
        "linear": (
            linear_smoothquant_weights_maxabs_pow2_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            matmul_scales_to_mod_config,
        ),
    },
    "weaksmoothquant_weights_maxabs_pow2": {
        "linear": (
            linear_weaksmoothquant_weights_maxabs_pow2_scales,
            linear_scales_to_mod_config,
        ),
        "matmul": (
            matmul_act_maxabs_pts_weight_maxabs_pts_pow2_hw_scales,
            matmul_scales_to_mod_config,
        ),
    },
}

scale_method_mapping = {
    (ScaleMethod.UNIT_SCALE, "maxabs"): "unit_scale",
    (ScaleMethod.UNIT_SCALE, "maxabs_per_channel"): "unit_scale",
    (ScaleMethod.HW_ALIGNED_SINGLE_SCALE, "maxabs"): "hw_aligned_single_scale",
    (ScaleMethod.HW_ALIGNED_SINGLE_SCALE, "maxabs_per_channel"): "hw_aligned_single_scale",
    (ScaleMethod.MAXABS_HW, "maxabs"): "act_maxabs_pts_weight_maxabs_pts_pow2_hw",
    (ScaleMethod.MAXABS_POW2, "maxabs"): "act_maxabs_pts_weight_maxabs_pts_pow2",
    (ScaleMethod.MAXABS_ARBITRARY, "maxabs"): "act_maxabs_pts_weight_maxabs_pts_arbitrary",
    (ScaleMethod.MAXABS_HW_OPT_WEIGHT, "maxabs"): "act_maxabs_pts_weight_opt_pts_hw",
    (
        ScaleMethod.MAXABS_POW2_OPT_WEIGHT,
        "maxabs",
    ): "act_maxabs_pts_weight_opt_pts_pow2",
    (
        ScaleMethod.ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2,
        "maxabs",
    ): "act_maxabs_pts_pow2_hw_weights_maxabs_pcs_pow2",
    (
        ScaleMethod.ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2,
        "maxabs_per_channel",
    ): "act_maxabs_pts_pow2_hw_weights_maxabs_pcs_pow2",
    (
        ScaleMethod.SMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2,
        "maxabs_per_channel",
    ): "smoothquant_weights_maxabs_pow2",
    (
        ScaleMethod.WEAKSMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2,
        "maxabs_per_channel",
    ): "weaksmoothquant_weights_maxabs_pow2",
    (
        ScaleMethod.ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2,
        "maxabs",
    ): "act_maxabs_pts_pow2_hw_weights_opt_pcs_pow2",
    (
        ScaleMethod.ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2,
        "maxabs_per_channel",
    ): "act_maxabs_pts_pow2_hw_weights_opt_pcs_pow2",
    (
        ScaleMethod.ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2,
        "maxabs",
    ): "act_maxabs_pts_pow2_weights_maxabs_pcs_pow2",
    (
        ScaleMethod.ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2,
        "maxabs_per_channel",
    ): "act_maxabs_pts_pow2_weights_maxabs_pcs_pow2",
    (
        ScaleMethod.ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2,
        "maxabs",
    ): "act_maxabs_pts_pow2_weights_opt_pcs_pow2",
    (ScaleMethod.SMOOTHQUANT_OPT, "maxabs_per_channel"): "smoothquant_weights_opt_pow2",
}

scaling_params = {
    "unit_scale": {},
    "hw_aligned_single_scale": {},
    "act_maxabs_pts_weight_maxabs_pts_arbitrary": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
    },
    "act_maxabs_pts_weight_maxabs_pts_pow2_hw": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
    },
    "act_maxabs_pts_weight_maxabs_pts_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
    },
    "act_maxabs_pts_weight_opt_pts_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
        "weight_scales": [2.0**s for s in range(-10, 10)],
    },
    "act_maxabs_pts_weight_opt_pts_hw": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
        "weight_scales": [2.0**s for s in [4, 0, -4, -8]],
    },
    "smoothquant_weights_maxabs_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
        "alpha": 0.5,
    },
    "weaksmoothquant_weights_maxabs_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
        "alpha": 0.5,
    },
    "act_maxabs_pts_pow2_hw_weights_maxabs_pcs_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
    },
    "act_maxabs_pts_pow2_hw_weights_opt_pcs_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
        "weight_scales": [2.0**s for s in range(-3, 5)],
    },
    "act_maxabs_pts_pow2_weights_maxabs_pcs_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
    },
    "act_maxabs_pts_pow2_weights_opt_pcs_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
        "weight_scales": [2.0**s for s in range(-3, 5)],
    },
    "smoothquant_weights_opt_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
        "alpha": 0.5,
        "transformed_weight_scales": [2.0**s for s in range(-3, 5)],
    },
}
