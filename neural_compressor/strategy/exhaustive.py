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
"""The exhaustive tuning strategy."""
from .strategy import TuneStrategy, strategy_registry
from .utils.tuning_sampler import OpWiseTuningSampler


@strategy_registry
class ExhaustiveTuneStrategy(TuneStrategy):
    """The exhaustive tuning strategy."""

    def next_tune_cfg(self):
        """Generate and yield the next tuning config using exhaustive search in tuning space.

        It sequentially traverse all possible quantization tuning configurations
        in a tuning space. From the perspective of the impact on performance,
        we currently only traverse all possible quantization tuning configs.
        Same reason as Bayesian, fallback datatypes are not included for now.

        Returns:
            tune_config (dict): A dict containing the tuning configuration for quantization.
        """
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name("calib_sampling_size").options
        for calib_sampling_size in calib_sampling_size_lst:
            op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg = self.initial_tuning_cfg()
            op_wise_tuning_sampler = OpWiseTuningSampler(
                tuning_space, [], [], op_item_dtype_dict, initial_op_tuning_cfg
            )
            for op_tuning_cfg in op_wise_tuning_sampler:
                op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
                yield op_tuning_cfg
