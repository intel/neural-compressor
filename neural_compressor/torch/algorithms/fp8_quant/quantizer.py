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
from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.algorithms.fp8_quant import (
    finish_measurements,
    prep_model,
    restore_patched_module,
    update_mode,
    with_patched_module,
)


class FP8Quantizer(Quantizer):
    def __init__(self, quant_config):
        super().__init__(quant_config)
        if isinstance(quant_config, dict):
            json_file = [cfg.json_file for cfg in quant_config.values()]
            assert len(json_file) > 0, "Cannot get json file from config."
            self.quant_config = json_file[0]

    def prepare(self, model):
        _prepare(model, self.quant_config)
        return model

    def convert(self, model):
        if with_patched_module(model):  # if model was calibrated on hpu
            finish_measurements(model)  # dump the measurements into files to be loaded in _convert
            # for INC flow, it calls `prepare` and then `convert` user-facing API in one run
            restore_patched_module(model)
        _convert(model, self.quant_config)
        return model


def _convert(model, config_path):
    # update mode to QUANTIZE
    config_path = update_mode(config_path, quant_step=True)

    return prep_model(model, config_path)


def _prepare(model, config_path):
    # update mode to MEASURE
    config_path = update_mode(config_path, measure_step=True)

    return prep_model(model, config_path)