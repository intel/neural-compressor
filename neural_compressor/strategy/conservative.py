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
"""The conservative tuning strategy for quantization level 0."""
import copy
import os
import numpy as np

from collections import deque
from collections import OrderedDict as COrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, OrderedDict

from .strategy import strategy_registry, TuneStrategy
from .utils.tuning_space import TuningItem
from ..utils import logger
from ..utils.utility import Statistics
from ..algorithm import AlgorithmScheduler

@strategy_registry
class ConservativeTuneStrategy(TuneStrategy):
    """Tuning strategy with accuracy first, performance second.
    
    The quantization level O0 is designed for user who want to keep the accuracy
    of the model after quantization. It starts with the original(fp32) model,
    and then quantize the OPs to lower precision OP type wisely and OP wisely.
    """

    def __init__(self, model, conf, q_dataloader, q_func=None, eval_dataloader=None, 
                 eval_func=None, dicts=None, q_hooks=None):
        """Init conservative tuning strategy."""
        super().__init__(model, conf, q_dataloader, q_func, eval_dataloader, 
                         eval_func, dicts, q_hooks)
        self.acc_meet_flag = False
        self.quant_op_type_lst = ['conv', 'matmul', 'linear']
        res_lst = [None] * len(self.quant_op_type_lst)
        self.quant_status = {k : v for k, v in zip(self.quant_op_type_lst, res_lst)}

    def next_tune_cfg(self):
        """Generate and yield the next tuning config with below order.

        1. Query all quantifiable ops and save as a list of [(op_name, op_type), ...]
        2. Classify the op by its op type
        3. Add op to quant_queue according to the op type priority
        4. Go through the quant_queue and replace it with the fp32 config in tune_cfg if
        accuracy meets the requirements else continue
        5. For bf16 and fp16 operators, do the same as int8 operators.

        Returns:
            tune_config (dict): It's a dict containing the tuning configuration to run.
        """
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        calib_sampling_size = calib_sampling_size_lst[0]
        tune_cfg = self._initialize_tune_cfg()
        tune_cfg['calib_sampling_size'] = calib_sampling_size
        op_type_priority = self._get_op_type_priority()
        quant_items_pool = self._quant_items_pool(op_type_priority)
        self.re_quant = True
        logger.info(f"*** Try to convert op into lower precision to improve performance.")
        for dtype, op_items in quant_items_pool.items():
            logger.info(f"*** Start to convert op into {dtype}.")
            for op_type, items_lst in op_items.items():
                logger.info(f"*** Try to convert all {op_type} ops into {dtype}.")
                tmp_tune_cfg = deepcopy(tune_cfg)
                for item, quant_mode in items_lst:
                    op_info = item.name
                    op_config = tuning_space.get_default_config(op_info, quant_mode)
                    tmp_tune_cfg[op_info] = op_config
                yield tmp_tune_cfg
                if self.objectives.accuracy_meets():
                    self.quant_status[op_type] = dtype
                    logger.info(f"*** Convert all {op_type} ops to {dtype} and accuracy still meet the requirements")
                    tune_cfg = deepcopy(tmp_tune_cfg)
                else:
                    # tmp_tune_cfg = deepcopy(tune_cfg)
                    self.quant_status[op_type] = 'fp32'
                    logger.info(f"*** Convert all {op_type} ops to {dtype} but accuracy not meet the requirements")
                logger.info(f"***Current result {self.quant_status.items()}")
        logger.info(f"*** Ending tuning process due to no quantifiable op left.")
        self.re_quant = False

            
    def _compare_performace(self, last_tune_result, best_tune_result): # pragma: no cover
        """Compare the tuning result with performance only.

        Args:
            last_tune_result (list): The list of last tuning result.
            best_tune_result (list): The list of best tuning result.

        Returns:
            bool: whether the best tuning result is better than last tuning result
              in performance.
        """
        _, last_perf = last_tune_result
        _, best_perf = best_tune_result
        return last_perf[0] < best_perf[0]
    
    def _dump_tuning_state(self, trials_count, last_tune_result, best_tune_result, baseline):
        if last_tune_result:
            last_tune = last_tune_result[0] if \
                isinstance(last_tune_result[0], list) else [last_tune_result[0]]
            for name, data in zip(self.metric_name, last_tune):
                if len(self.tune_data[name]) == 1:
                    self.tune_data[name].append(data)
                else:
                    self.tune_data[name][1] = data

            if self.metric_weight and len(last_tune) > 1:
                weighted_acc = np.mean(np.array(last_tune) * self.metric_weight)                    
                if len(self.tune_data['Weighted accuracy']) == 1:
                    self.tune_data['Weighted accuracy'].append(weighted_acc)
                else:
                    self.tune_data['Weighted accuracy'][1] = weighted_acc
                last_tune = [weighted_acc]

            last_tune_msg = '[Accuracy (int8|fp32):' + \
                ''.join([' {:.4f}|{:.4f}'.format(last, base) for last, base in \
                zip(last_tune, self.tune_data['baseline'])]) + \
                ''.join([', {} (int8|fp32): {:.4f}|{:.4f}'.format( \
                x, y, z) for x, y, z in zip( \
                self.objectives.representation, last_tune_result[1], baseline[1]) \
                if x != 'Accuracy']) + ']'
        else: # pragma: no cover
            last_tune_msg = 'n/a'
            for name in self.tune_data.keys() - {'baseline'}:
                if len(self.tune_data[name]) == 1:
                    self.tune_data[name].append('n/a')
                else:
                    self.tune_data[name][1] = 'n/a'

        if best_tune_result:
            best_tune = best_tune_result[0] if isinstance(best_tune_result[0], list) \
                        else [best_tune_result[0]]
 
            for name, data in zip(self.metric_name, best_tune):
                if len(self.tune_data[name]) == 2:
                     self.tune_data[name].append(data)
                else:
                    self.tune_data[name][2] = data

            if self.metric_weight and len(best_tune) > 1:
                weighted_acc = np.mean(np.array(best_tune) * self.metric_weight)

                if len(self.tune_data['Weighted accuracy']) == 2:
                    self.tune_data['Weighted accuracy'].append(weighted_acc)
                else: # pragma: no cover
                    self.tune_data['Weighted accuracy'][2] = weighted_acc

                best_tune = [weighted_acc]

            best_tune_msg = '[Accuracy:' + ''.join([' {:.4f}'.format(best) \
                for best in best_tune]) + ''.join([', {}: {:.4f}'.format(x,y) \
                for x,y in zip(self.objectives.representation, \
                best_tune_result[1]) if x != 'Accuracy']) + ']'

        else:
            best_tune_msg = 'n/a'
            for name in self.tune_data.keys() - {'baseline'}:
                if len(self.tune_data[name]) == 2:
                    self.tune_data[name].append('n/a')
                else:
                    self.tune_data[name][2] = 'n/a'

        logger.info("Tune {} result is: {}, Best tune result is: {}".format(trials_count,
                                                                            last_tune_msg,
                                                                            best_tune_msg))
        output_data = [[info_type, 
            '{:.4f} '.format(self.tune_data[info_type][0]) if \
            not isinstance(self.tune_data[info_type][0], str) else self.tune_data[info_type][0], 
            '{:.4f} '.format(self.tune_data[info_type][1]) if \
            not isinstance(self.tune_data[info_type][1], str) else self.tune_data[info_type][1],
            '{:.4f} '.format(self.tune_data[info_type][2]) if \
            not isinstance(self.tune_data[info_type][2], str) else self.tune_data[info_type][2]] \
            for info_type in self.tune_data.keys() if info_type != 'baseline']

        output_data.extend([[obj, 
            '{:.4f} '.format(baseline[1][i]) if baseline else 'n/a',
            '{:.4f} '.format(last_tune_result[1][i]) if last_tune_result else 'n/a',
            '{:.4f} '.format(best_tune_result[1][i]) if best_tune_result else 'n/a'] \
            for i, obj in enumerate(self.objectives.representation)])

        Statistics(output_data,
                   header='Tune Result Statistics',
                   field_names=['Info Type', 'Baseline', 'Tune {} result'.format(trials_count), \
                                                                'Best tune result']).print_stat()

    def _get_op_type_priority(self):
        optypewise_cap = self.capability['optypewise']
        op_type_priority = list(optypewise_cap.keys())
        return op_type_priority

    def _sorted_item_by_op_type(self, 
                                items_lst, 
                                op_type_priority: List[str]) -> OrderedDict[str, List]:
        """Scoring the tuning items according to its op type.
        
        Args:
            items_lst: The tuning item list. # [(op_item, quant_mode), ... ]
            op_type_priority: The op type list with the order. # [optype_1, optype_2]

        Returns:
            The tuning items list that sorted according to its op type.
            OrderDict:
                # op_type: [(TuningItem, quant_mode), ...]
                conv: [(TuningItem, static), (TuningItem, static)] 
                linear: [(TuningItem, static), (TuningItem, static)]
                matmul: [(TuningItem, static), (TuningItem, static)]
        """
        sorted_items = COrderedDict()
        for op_item, quant_mode in items_lst:
            op_name, op_type = op_item.name
            for target_op_type in self.quant_op_type_lst:
                if target_op_type in op_type.lower():
                    if target_op_type not in sorted_items:
                        sorted_items[target_op_type] = []
                    sorted_items[target_op_type].append((op_item, quant_mode))
        return sorted_items
            
    def _initialize_tune_cfg(self):
        """Initialize the tuning config with fp32 AMAP.

        Returns:
            The intialized tuning config.
        """
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
            initial_tuning_cfg[op_info] = tuning_space.get_default_config(op_info, "fp32")
        for op_info, quant_mode in non_fp32_ops_dtype.items():
            initial_tuning_cfg[op_info] = tuning_space.get_default_config(op_info, quant_mode)
        return initial_tuning_cfg
            
    def _quant_items_pool(self, op_type_priority: List[str]) -> OrderedDict[
        str, OrderedDict[str, List[Tuple[TuningItem, str]]]]:
        """Create the op queue to be quantized.
        
        --------------------------------------------------------------------------
        | Level 1 |         bf16       |         fp16       |  static/dynamic    |
        | Level 2 | conv2d, linear, ...| conv2d, linear, ...| conv2d, linear, ...|
        
        Args:
            op_type_priority: The optype list with priority.
            
        Returns:
            The op item pool to convert into lower precision.
            quant_items_pool(OrderDict):
                bf16:
                    OrderDict:
                        conv2d: [(TuningItem, bf16), (TuningItem, bf16)]
                        linear: [(TuningItem, bf16), (TuningItem, bf16)]
                int8:
                    OrderDict:
                        # (TuningItem, quant_mode)
                        conv2d: [(TuningItem, static), (TuningItem, static)] 
                        linear: [(TuningItem, static), (TuningItem, static)]
        """
        quant_mode_wise_items = self.tuning_space.quant_mode_wise_items
        # Add all quantized pair into queue
        quant_items_pool = COrderedDict()
        # collect and sorted all ops that support bf16 and fp16
        for quant_mode in  ['bf16', 'fp16']:
            if quant_mode in quant_mode_wise_items:
                op_item_pairs = [(op_item, quant_mode) for op_item in quant_mode_wise_items[quant_mode]]
                op_item_pairs = self._sorted_item_by_op_type(op_item_pairs, op_type_priority)
                quant_items_pool[quant_mode] = op_item_pairs
        op_item_pairs = []
        quant_ops_name_set = set()
        # collect and sorted all ops that support int8
        for quant_mode, items_lst in quant_mode_wise_items.items():
            if "static" in quant_mode or 'dynamic' in quant_mode:
                _quant_mode = "static" if "static" in quant_mode else "dynamic"
                op_item_pairs += [(item, _quant_mode) for item in items_lst if item.name not in quant_ops_name_set]
                quant_ops_name_set = quant_ops_name_set.union([item.name for item in items_lst])
                op_item_pairs = self._sorted_item_by_op_type(op_item_pairs, op_type_priority)
                quant_items_pool['int8'] = op_item_pairs
        return quant_items_pool
