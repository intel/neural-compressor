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

import os
from neural_compressor.torch.algorithms.habana_fp8 import get_mod_list, update_mode, restore_patched_module, with_patched_module
from neural_compressor.torch.algorithms import algo_quantizer_register, AlgoBase
from neural_compressor.common.utils import FP8_QUANT

@algo_quantizer_register(name=FP8_QUANT)
class FP8Quantizer(AlgoBase):
    def __init__(self, config):
        super().__init__(config)
        if isinstance(config, dict):
            json_file = [cfg.json_file for cfg in config.values()]
            assert len(json_file) > 0, "Cannot get json file from config."
            self.quant_config = json_file[0]

    def prepare(self, model):
        # set environment
        os.environ['QUANT_CONFIG'] = self.quant_config
        _prepare(model)
        return model

    def convert(self, model):
        # set environment

        os.environ['QUANT_CONFIG'] = self.quant_config
        if with_patched_module(model):
            restore_patched_module(model)
        _convert(model)
        return model

def _convert(model):
    from habana_quantization_toolkit._hook_method import config, quantize_hooks, scale_method_mapping, scaling_params

    # update mode to QUANTIZE
    update_mode(quant_step=True)

    mod_list = get_mod_list(model)
    scaling_method_name = scale_method_mapping[(config.cfg['scale_method'], config.cfg['observer'])]
    scaling_params[scaling_method_name].update(config.cfg['scale_params'])
    config.cfg['scale_params'] = scaling_params[scaling_method_name]

    return quantize_hooks(model, mod_list)

def _prepare(model):
    from habana_quantization_toolkit._hook_method import prepare_model_for_measure

    # update mode to MEASURE
    update_mode(calib_step=True)

    mod_list = get_mod_list(model)

    return prepare_model_for_measure(model, mod_list)
