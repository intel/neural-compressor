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

from typing import Callable, Dict
from collections import OrderedDict

import tensorflow as tf

from neural_compressor.tensorflow.quantization.config import StaticQuantConfig

class ParseKerasConfig:
    """The class that parse StaticQuantConfig to tunning config"""
    support_int8_weight = {"Dense", "Conv2d", "DepthwiseConv2D", "SeparableConv2D"}

    def __init__(self,
                 quant_config: StaticQuantConfig, 
                 calib_iteration: int):
        """Init parser for keras static quant config.

        Args:
            quant_config: the keras static quant config.
            calib_iteration: the iteration of calibration.
        """
        self.quant_config = quant_config
        self.calib_iteration = calib_iteration

    def update_config(self, quant_config, op_key):
        """Update op-wise config.
        
            Args:
                quant_config: the keras static quant config.
                op_key: a tuple such as (layer type, layer name).
        """
        op_value = {"activation": {}}
        op_value["activation"].update(
            {
                "dtype": quant_config.act_dtype,
                "quant_mode": "static",
                "scheme": ("sym" if quant_config.act_sym else "asym"),
                "granularity": quant_config.act_granularity,
                "algorithm": "minmax",
            }
        )
        if op_key[1] not in self.support_int8_weight:
            return
        op_value["weight"] = {
            "dtype": quant_config.weight_dtype,
            "scheme": "sym" if quant_config.weight_sym else "asym",
            "granularity": quant_config.weight_granularity,
            "algorithm": "minmax",
        }
        return op_value

    def parse_to_tune_cfg(self) -> Dict:
        """The function that parses StaticQuantConfig to keras tuning config."""
        tune_cfg = {"op": OrderedDict()}
        for op_key, config in self.quant_config.items():
            op_value = self.update_config(config, op_key)
            tune_cfg["op"].update({op_key: op_value})
            tune_cfg["calib_iteration"] = self.calib_iteration

        return tune_cfg
