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

from copy import deepcopy
import numpy as np
from collections import OrderedDict, deque

from .strategy import strategy_registry, TuneStrategy
from ..utils import logger

from .st_utils.tuning_sampler import OpTypeWiseTuningSampler, FallbackTuningSampler, ModelWiseTuningSampler
from .st_utils.tuning_structs import OpTuningConfig
from .st_utils.tuning_space import TUNING_ITEMS_LST

@strategy_registry
class ConservativeTuneStrategy(TuneStrategy):
    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None, q_hooks=None):
        super(
            ConservativeTuneStrategy,
            self).__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts,
            q_hooks)

    def next_tune_cfg(self):
        """
        Conservative tuning: accuracy first, performance second
        
        1. Query all quantifiable ops and save as a list: quantifiable_ops = [(op_name, op_type), ...]
        2. Classify the op by its op type
        3. Add op to quant_queue according to the op type priority
        4. Go through the quant_queue and replace it with the fp32 config in tune_cfg if
           accuracy meet the requirements else continue
        
        For bf16 and fp16, do the same thing as int8
        Note:
        1) other tunable items will using the first option as the default value.
        2) If auto: for op support both dynamic and static, use dynamic.

        Yields:
            tune_config (dict): It's a dict containing the tuning configuration to run.
        """
        
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        calib_sampling_size = calib_sampling_size_lst[0]
        tune_cfg, quant_queue = self._initialize_tune_cfg()
        tune_cfg['calib_sampling_size'] = calib_sampling_size
        # Try to add quantized ops.
        import pdb
        pdb.set_trace()
        logger.debug(f"*** Quantized op list: {[(pair[0].name, pair[1]) for pair in quant_queue]}")
        while quant_queue:
            op_item, target_dtype = quant_queue.pop()
            op_info = op_item.name
            op_name, op_type = op_info
            op_config = tuning_space.set_deafult_config(op_info, target_dtype)
            tmp_tune_cfg = deepcopy(tune_cfg)
            tmp_tune_cfg[op_info] = op_config
            yield tmp_tune_cfg
            acc, _ = self.last_tune_result
            acc_meet_flag = self.acc_meet(acc)
            if acc_meet_flag:
                logger.info(f"*** Quantized op {op_name} and accuracy still meet the requiments")
                tune_cfg[op_info] = op_config
            else:
                logger.info(f"*** Skip quantize op {op_name}.")
        logger.info(f"*** Ending tuning process due to no quantifiable op left.")
        
    def acc_meet(self, acc):
        return True
            
    def _sorted_item_by_op_type(self, items_lst, op_type_priority):
        priority_val_lst = range(len(op_type_priority))
        max_priority_val = len(op_type_priority) + 1
        priority_map = dict(zip(op_type_priority, priority_val_lst))
        item_priority = []
        for item in items_lst:
            item_op_type = item[0].name[0]
            priority_val =  op_type_priority[item_op_type] if item_op_type in op_type_priority else max_priority_val
            item_priority.append((item, priority_val))
        sorted_item_priority = sorted(item_priority, key=lambda x: x[1])
        sorted_items_lst = [item[0] for item in sorted_item_priority]
        import pdb
        pdb.set_trace()
        return sorted_items_lst
            
    def _initialize_tune_cfg(self):
        tuning_space = self.tuning_space
        quant_mode_wise_items = tuning_space.quant_mode_wise_items
        # Initialize the tuning config
        initial_tuning_cfg = {}
        all_ops = set()
        fp32_ops = []
        for quant_mode, items_lst in quant_mode_wise_items.items():
            items_name_lst = [item.name for item in items_lst]
            all_ops = all_ops.union(set(items_name_lst))
            if quant_mode == "fp32":
                fp32_ops += [item.name for item in items_lst]
        non_fp32_ops_dtype = {}
        fp32_ops_set = set(fp32_ops)
        for quant_mode, items_lst in quant_mode_wise_items.items():
            items_name_set = set([item.name for item in items_lst])
            tmp_non_fp32_ops = items_name_set.difference(fp32_ops_set)
            if tmp_non_fp32_ops:
                for op_info in tmp_non_fp32_ops:
                    non_fp32_ops_dtype[op_info] = quant_mode

        for op_info in fp32_ops:
            initial_tuning_cfg[op_info] = tuning_space.set_deafult_config(op_info, "fp32")
        for op_info, quant_mode in non_fp32_ops_dtype.items():
            initial_tuning_cfg[op_info] = tuning_space.set_deafult_config(op_info, quant_mode)
        
        # Add all quantized pair into queue
        quant_ops_queue = deque([])
        for quant_mode in  ['bf16', 'fp16']:
            if quant_mode in quant_mode_wise_items:
                op_item_pairs = [(op_item, quant_mode) for op_item in quant_mode_wise_items[quant_mode]]
                op_item_pairs = self._sorted_item_by_op_type(op_item_pairs, ['conv2d', 'linear'])
                quant_ops_queue.extend(op_item_pairs)
        op_item_pairs = []
        quant_ops_name_set = set()
        for quant_mode, items_lst in quant_mode_wise_items.items():
            if "static" in quant_mode or 'dynamic' in quant_mode:
                op_item_pairs += [(item, quant_mode) for item in items_lst if item.name not in quant_ops_name_set]
                quant_ops_name_set = quant_ops_name_set.union([item.name for item in items_lst])
                op_item_pairs = self._sorted_item_by_op_type(op_item_pairs, ['conv2d', 'linear'])
                quant_ops_queue.extend(op_item_pairs)
        return initial_tuning_cfg, quant_ops_queue
        
        
            
            
            
            
            
                
            
        
         
         
        

            
            
