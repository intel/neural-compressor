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

from .._quant_common.quant_config import ScaleMethod, set_hqt_config
from ..utils.logger import logger
from .common import *
from .fp_utils import *
from .quant_dequant import *
from .scale_methods import *


def matmul_scales_to_mod_config(mod, scales, params):
    scales_inv = invert_scales(scales)
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    input_config = [QuantInput(s_inv, lp_dtype, hp_dtype) for s_inv in scales_inv.inputs]
    # outputs as bf16, and descaled in gemm under PatchedMatmul, so no need to work here
    output_config = [QuantDequantNone(lp_dtype, hp_dtype)]
    config = ModuleConfig(input_config, output_config, {})
    return config


def fsdpa_scales_to_mod_config(mod, scales, params):
    scales_inv = invert_scales(scales)
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    input_config = [QuantInput(s_inv, lp_dtype, hp_dtype) for s_inv in scales_inv.inputs]
    output_config = [DequantOutput(scales.outputs[0], lp_dtype, hp_dtype)]
    config = ModuleConfig(input_config, output_config, {})
    return config


def linear_scales_to_mod_config(mod, scales, params):
    scales_inv = invert_scales(scales)
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    input_config = [QuantInput(scales_inv.inputs[0], lp_dtype, hp_dtype)]
    # outputs as bf16, and descaled in gemm under PatchedLinear, so no need to work here
    output_config = [QuantDequantNone(lp_dtype, hp_dtype)]
    if isinstance(scales_inv.params["weight"], (torch.Tensor, float)):
        weight_config = QuantInput(scales_inv.params["weight"], lp_dtype, hp_dtype)
    elif isinstance(scales_inv.params["weight"], dict):
        weight_scale_inv_out_ch = scales_inv.params["weight"][0]
        weight_scale_inv_in_ch = scales_inv.params["weight"][1]
        if isinstance(weight_scale_inv_out_ch, torch.Tensor):
            scale_inv = torch.mul(
                weight_scale_inv_in_ch.reshape([1, -1]),
                weight_scale_inv_out_ch.reshape([-1, 1]),
            )
        else:
            # TODO SW-169781: Handle here scalar weight for PCQ
            raise TypeError(f"Unknown weight scales type: {type(weight_scale_inv_out_ch)}.")
        weight_config = QuantInput(scale_inv, lp_dtype, hp_dtype)
    else:
        logger.error("Unknown weight scales format.")
    params_config = {"weight": weight_config}
    if hasattr(mod, "bias") and (getattr(mod, "bias") is not None):
        # In PatchedLinear the bias is added to the output of gemm.
        # The output is expected to be descaled and in bf16, so we don't need to touch the bias.
        bias_config = QuantDequantNone(lp_dtype, hp_dtype)
        params_config.update({"bias": bias_config})
    config = ModuleConfig(input_config, output_config, params_config)
    return config


def kv_cache_scales_to_mod_config(mod, scales, params):
    # how quant/dequant will be applied on layer tensors
    scales_inv = invert_scales(scales)
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    input_config = [QuantInput(scales_inv.inputs[0], lp_dtype, hp_dtype)]
    output_config = [DequantOutput(scales.outputs[0], lp_dtype, hp_dtype)]
    config = ModuleConfig(input_config, output_config)
    return config


def softmax_scales_to_mod_config(mod, scales, params):
    lp_dtype = params["lp_dtype"]
    hp_dtype = params["hp_dtype"]
    output_config = [DequantOutput(scales.outputs[0], lp_dtype, hp_dtype)]
    return ModuleConfig(None, output_config)


def get_config(
    model,
    measurement,
    mod_dict,
    method,
    params,
    scales_file=None,
    recalc_scales=False,
    mod_list=None,
):
    with torch.no_grad():
        top_level_config = get_hqt_config(model)
        qconfig = {UNMEASURED_MODELS: []}
        scales_file_format = np.ndarray  # file_functions[os.path.splitext(scales_file)[1]][0]
        scales_obj = (
            load_scales(scales_file + ".npz", scales_file_format)
            if (scales_file is not None) and not recalc_scales
            else {}
        )
        scales = convert_scales_to_tensors_dict(scales_obj, scales_file_format, params["hp_dtype"])
        model_dict = dict(model.named_modules())
        for mname in mod_list:
            mod = model_dict[mname]
            set_hqt_config(mod, top_level_config)  # set config in the module, as it consumed by the patched module
            mod_type_str = mod.__class__.__name__
            layer_type = mod_dict[mod_type_str].type
            if mname not in scales:
                logger.debug("Calculating scales for layer %s", mname)
                if mname not in measurement:
                    qconfig[UNMEASURED_MODELS].append(mname)
                    logger.debug(
                        "Layer '%s' has no measurements therefore it can't be quantized.",
                        mname,
                    )
                    continue
                layer_measure = measurement[mname]  # ModuleConfig() of measurements
                scales[mname] = method[layer_type][0](mod, layer_measure, params)  # ModuleConfig() of scales
                if scales_file is not None:
                    scales_obj[mname] = ModuleConfig(
                        **format_functions_rec((torch.Tensor, scales_file_format))(scales[mname].__dict__)
                    )

            logger.debug(
                "Preparing quantization functions for layer %s layer_type=%s",
                mname,
                layer_type,
            )
            mod_config = method[layer_type][1](mod, scales[mname], params)  # ModuleConfig() of QuantDequant
            mod_extra_config = ModuleExtraConfig(
                mod_config.inputs,
                mod_config.outputs,
                mod_config.params,
                scales[mname],
                params,
            )
            qconfig[mname] = mod_extra_config
        if scales_file is not None:
            save_scales(model, scales_obj, scales_file_format, scales_file + ".npz")
            save_scales(model, scales_obj, scales_file_format, scales_file + ".json")
    return qconfig


scaling_methods = {
    "unit_scale": {
        "linear": (linear_unit_scale_scales, linear_scales_to_mod_config),
        "matmul": (matmul_unit_scale_scales, matmul_scales_to_mod_config),
        "softmax": (softmax_unit_scale_scales, softmax_scales_to_mod_config),
        "kv_cache": (kv_cache_unit_scale_scales, kv_cache_scales_to_mod_config),
        "fused_sdpa": (fsdpa_unit_scale_scales, fsdpa_scales_to_mod_config),
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
            kv_cache_act_maxabs_pts_pow2_weight_opt_pcs_pow2_scales,
            kv_cache_scales_to_mod_config,
        ),
        "fused_sdpa": (
            fsdpa_act_maxabs_pts_weight_maxabs_pts_pow2_scales,
            fsdpa_scales_to_mod_config,
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
    (ScaleMethod.MAXABS_HW, "maxabs"): "act_maxabs_pts_weight_maxabs_pts_pow2_hw",
    (ScaleMethod.MAXABS_POW2, "maxabs"): "act_maxabs_pts_weight_maxabs_pts_pow2",
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
