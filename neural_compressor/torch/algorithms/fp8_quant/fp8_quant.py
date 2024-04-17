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

from neural_compressor.common.utils import FP8_QUANT
from neural_compressor.torch.algorithms import AlgoBase, algo_quantizer_register
from neural_compressor.torch.algorithms.fp8_quant import (
    restore_patched_module,
    update_mode,
    with_patched_module,
)


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
        # os.environ["QUANT_CONFIG"] = self.quant_config
        _prepare(model, self.quant_config)
        return model

    def convert(self, model):
        # set environment

        # os.environ["QUANT_CONFIG"] = self.quant_config
        if with_patched_module(model):
            restore_patched_module(model)
        _convert(model, self.quant_config)
        return model


def _convert(model, config_path):
    from neural_compressor.torch.algorithms.fp8_quant import habana_quantization_toolkit

    # update mode to QUANTIZE
    config_path = update_mode(config_path, quant_step=True)

    return habana_quantization_toolkit.prep_model(model, config_path)


def _prepare(model, config_path):
    from neural_compressor.torch.algorithms.fp8_quant import habana_quantization_toolkit

    # update mode to MEASURE
    config_path = update_mode(config_path, calib_step=True)

    return habana_quantization_toolkit.prep_model(model, config_path)
