#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The basic tuning strategy."""
from .strategy import strategy_registry, TuneStrategy
from ..utils import logger


@strategy_registry
class FWKQuantTuneStrategy(TuneStrategy):
    """Quantize model with default config.
    
    Quantize the model with the default config which is consistent with the framework behavior.
    """
    
    def next_tune_cfg(self):
        """Generate and yield the next tuning config.

        Returns:
            tune_config (dict): A dict containing the tuning configuration for quantization.
        """
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        # Initialize the tuning config for each op according to the quantization approach.
        _, _, op_tuning_cfg = self.initial_tuning_cfg()
        op_tuning_cfg['calib_sampling_size'] = calib_sampling_size_lst[0]
        logger.info(f"Quantize the model with default config.")
        yield op_tuning_cfg