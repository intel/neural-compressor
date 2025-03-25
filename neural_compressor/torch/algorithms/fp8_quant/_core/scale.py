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

from .common import format_functions_rec, logger
from .patching_common import mod_default_dict
from ..model_configs import ModuleConfig, ModuleExtraConfig
from .scale_methods import ops_quantizer
from .._quant_common.quant_config import ScaleMethod
import torch


def load_layer_scales(mod, mod_name, config, mod_type_str, measurement, scales, scale_file,
                      scales_file_format, scales_obj, scaling_method_name, scale_config, save_file):
    module_type = mod_default_dict[mod_type_str].type
    logger.debug(
        "Preparing quantization functions for module %s module_type=%s",
        mod_name,
        module_type,
    )
    mod_extra_config = None
    if mod_name in scales or not config.cfg["use_stats_files"] or mod_name in measurement:
        op_for_scale_obj = ops_quantizer.get_op_quantizer(scaling_method_name, mod, measurement.get(mod_name, None),
                                                          scale_config, module_type)
        if mod_name not in scales:
            logger.debug("Calculating scales for module %s", mod_name)
            # calculates scales for current module according to scalling_methods
            scales[mod_name] = op_for_scale_obj.get_scales_module_config()  # ModuleConfig of scales
            if scale_file is not None:
                scales_obj[mod_name] = ModuleConfig(
                    **format_functions_rec((torch.Tensor, scales_file_format))(scales[mod_name].__dict__)
                )
                save_file = True
        # calculates QuantDequant config for current module according to scalling_methods
        mod_config = op_for_scale_obj.scales_module_config_to_q_and_dq(scales[mod_name])  # ModuleConfig of QuantDequant
        mod_extra_config = ModuleExtraConfig(
                mod_config.inputs,
                mod_config.outputs,
                mod_config.params,
                scales[mod_name],
                scale_config,
                )
    return mod_extra_config, save_file

def prepare_layer_scales(mod, mod_name, config, mod_type_str, measurement, scales, scale_file,
                      scales_file_format, scales_obj, scaling_method_name, scale_config, save_file):
    module_type = mod_default_dict[mod_type_str].type
    logger.debug(
        "Preparing quantization functions for module %s module_type=%s",
        mod_name,
        module_type,
    )
    mod_extra_config = None
    op_obj = ops_quantizer.get_op_quantizer(scaling_method_name, mod, None, scale_config, module_type)
    logger.debug("Preparing dynamic scales for module %s", mod_name)
    # calculates scales for current module according to scaling_methods
    scales[mod_name] = op_obj.get_scales_module_config()  # ModuleConfig of scales

    # Creates QuantDequant config for current module according to scaling_methods
    mod_config =  op_obj.scales_module_config_to_q_and_dq(scales[mod_name])  # ModuleConfig of QuantDequant
    mod_extra_config = ModuleExtraConfig(
            mod_config.inputs,
            mod_config.outputs,
            mod_config.params,
            scales[mod_name],
            scale_config,
            )
    return mod_extra_config, save_file


scale_method_mapping = {
    (ScaleMethod.UNIT_SCALE, "maxabs"): "unit_scale",
    (ScaleMethod.UNIT_SCALE, "maxabs_per_channel"): "unit_scale",
    (ScaleMethod.HW_ALIGNED_SINGLE_SCALE, "maxabs"): "hw_aligned_single_scale",
    (ScaleMethod.HW_ALIGNED_SINGLE_SCALE, "maxabs_per_channel"): "hw_aligned_single_scale",
    (ScaleMethod.MAXABS_HW, "maxabs"): "act_maxabs_pts_pow2_hw_weight_maxabs_pts_pow2_hw",
    (ScaleMethod.MAXABS_POW2, "maxabs"): "act_maxabs_pts_pow2_weight_maxabs_pts_pow2",
    (ScaleMethod.MAXABS_ARBITRARY, "maxabs"): "act_maxabs_pts_weight_maxabs_pts_arbitrary",
    (ScaleMethod.MAXABS_POW2_DYNAMIC, "maxabs"): "act_maxabs_pcs_dyn_pow2_weight_maxabs_pts_pow2_hw", # TODO: remove when changing config parsing
    (ScaleMethod.MAXABS_HW_OPT_WEIGHT, "maxabs"): "act_maxabs_pts_hw_weight_opt_pts_hw",
    (
        ScaleMethod.MAXABS_POW2_OPT_WEIGHT,
        "maxabs",
    ): "act_maxabs_pts_pow2_weight_opt_pts_pow2",
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
    "act_maxabs_pcs_dyn_pow2_weight_maxabs_pts_pow2_hw": {
        "input_backoff": 1.0,
        "weight_backoff": 0.5,
    },
    "act_maxabs_pts_pow2_hw_weight_maxabs_pts_pow2_hw": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
    },
    "act_maxabs_pts_pow2_weight_maxabs_pts_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
    },
    "act_maxabs_pts_pow2_weight_opt_pts_pow2": {
        "input_backoff": 0.25,
        "weight_backoff": 0.5,
        "weight_scales": [2.0**s for s in range(-10, 10)],
    },
    "act_maxabs_pts_hw_weight_opt_pts_hw": {
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
