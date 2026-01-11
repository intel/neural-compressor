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
from .scale_methods.scale_method_factory import QuantTensorName
from .scale_methods.scale_method_config import CfgStr

import torch



def load_layer_scales(mod, mod_name, config, mod_type_str, measurement, scales, scale_file, scales_file_format,
                      scales_obj, scaling_method_config, scale_config, save_file, scale_method_config_by_mod_map):
    module_type = mod_default_dict[mod_type_str].type
    logger.debug(
        "Preparing quantization functions for module %s module_type=%s",
        mod_name,
        module_type,
    )
    mod_extra_config = None

    if (mod_name in scales or not config.cfg["use_stats_files"] or mod_name in measurement) and mod_default_dict[mod_type_str].should_measure_and_quant:
        op_for_scale_obj = ops_quantizer.get_op_quantizer(scaling_method_config, mod, measurement.get(mod_name, None), scale_config, mod_type_str)
        # save mapping of current module name to scale method config
        scale_method_config_by_mod_map[mod_name] ={
            CfgStr.ACTIVATION: op_for_scale_obj.scales_method_factory.scale_method_config_map[QuantTensorName.INPUT], 
            CfgStr.WEIGHT: op_for_scale_obj.scales_method_factory.scale_method_config_map[QuantTensorName.WEIGHT_IN_CH]
        }
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
