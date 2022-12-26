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

from collections import OrderedDict
from .strategy import strategy_registry, TuneStrategy

from .utils.tuning_sampler import OpWiseTuningSampler, FallbackTuningSampler, ModelWiseTuningSampler
from .utils.tuning_structs import OpTuningConfig
from ..utils import logger

@strategy_registry
class ExhaustiveTuneStrategy(TuneStrategy):
    """The exhaustive tuning strategy."""

    def next_tune_cfg(self):
        """Generate and yield the next tuning config using exhaustive search in tuning space.
        
        It sequentially traverse all possible quantization tuning configurations in a tuning space.
        
        Yields:
            tune_config (dict): A dict containing the tuning configuration for quantization.
        """
        from copy import deepcopy
        tuning_space = self.tuning_space
        initial_op_tuning_cfg = {}
        for item in tuning_space.root_item.options:
            if item.item_type == 'op':
                op_name, op_type = item.name
                initial_op_tuning_cfg[item.name] = OpTuningConfig(op_name, op_type, 'fp32', tuning_space)
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        for calib_sampling_size in calib_sampling_size_lst:
            # step1. collect the ops that support static and dynamic
            quant_mode_wise_items = OrderedDict()
            query_order = ['static', 'dynamic', 'bf16', 'fp32']
            pre_items = set()
            for quant_mode in query_order:
                items = tuning_space.query_items_by_quant_mode(quant_mode)
                filtered_items = [item for item in items if item not in pre_items]
                pre_items = pre_items.union(set(items))
                quant_mode_wise_items[quant_mode] = filtered_items

            def initial_op_quant_mode(items_lst, target_quant_mode, op_item_dtype_dict):
                for item in items_lst:
                    op_item_dtype_dict[item.name] = target_quant_mode

            op_item_dtype_dict = OrderedDict()
            for quant_mode, quant_mode_items in quant_mode_wise_items.items():
                initial_op_quant_mode(quant_mode_items, quant_mode, op_item_dtype_dict)

            # step3. optype-wise tuning tuning items: the algorithm/scheme/granularity of activation(weight)
            early_stop_tuning = False
            stage1_cnt = 0
            int8_ops = quant_mode_wise_items['dynamic'] + quant_mode_wise_items['static']
            stage1_max = min(5, len(int8_ops))  # TODO set a more appropriate value
            op_wise_tuning_sampler = OpWiseTuningSampler(tuning_space, [], [], 
                                                         op_item_dtype_dict, initial_op_tuning_cfg)
            # TODO combine with op wise 
            model_wise_tuning_sampler = ModelWiseTuningSampler(tuning_space, [], [], 
                                                               op_item_dtype_dict, initial_op_tuning_cfg)
            for op_tuning_cfg in op_wise_tuning_sampler:
                stage1_cnt += 1
                if early_stop_tuning and stage1_cnt > stage1_max:
                    logger.info("Early stopping the stage 1.")
                    break
                op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
                yield op_tuning_cfg
        return
