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

"""Tuning sampler."""

from itertools import product
import copy
from collections import deque, OrderedDict, defaultdict
from typing import List, Dict, Any
from .tuning_space import TuningSpace
from .tuning_structs import OpTuningConfig
from ...utils import logger

TUNING_ITEM_PRIORITY = [('activation','scheme'), ('activation','algorithm'),('activation','granularity'), 
                        ('activation','compute_dtype'), ('weight','scheme'), ('weight','algorithm'), \
                        ('weight','granularity')]


class TuningOrder:
    """Not displayed in API Docs."""

    def __init__(self):
        """For future use."""
        pass


class TuningSampler:
    """Not displayed in API Docs.
    
    Basic class of tuning sampler.
    """

    def __init__(self, 
                 tuning_space: TuningSpace, 
                 tuning_order_lst: List[TuningOrder], 
                 initial_op_tuning_cfg: Dict):
        """Init tuning sampler.

        Args:
            tuning_space: The tuning space.
            tuning_order_lst: The traverse orders.
            initial_op_tuning_cfg: The initialized tuning config.
        """
        self.tuning_space = tuning_space
        self.tuning_order_lst = tuning_order_lst
        self.initial_op_tuning_cfg = initial_op_tuning_cfg
        self.queue = deque()

    def __iter__(self):
        """Interface for generate the next tuning config."""
        pass

class ModelWiseTuningSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(self,
                 tuning_space: TuningSpace,
                 tuning_items_priority: List[str],
                 tuning_order_lst: List[TuningOrder],
                 op_dtype_dict: Dict[tuple, str],
                 initial_op_tuning_cfg: Dict[tuple, OpTuningConfig]):
        """Model type wise tuning sampler.

        step1. create a default tuning config for each op
        step2. collect all tuning items and options, and build the model-wise traverse order
        step3. yield the tuning item with option one by one, query the existence of tuning item 
               and specific option for one op if exist, use the default tuning config if not exist
               
        Args:
            tuning_space: Tuning space.
            tuning_items_priority: The priority to traverse the tuning items.
            tuning_order_lst: The tuning orders.
            op_dtype_dict: The (op name, op type) and its target data type.
            initial_op_tuning_cfg: The initial tuning config.
        
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)

        self.op_dtype_dict = op_dtype_dict
        self.tuning_space = tuning_space
        self.default_op_config = {}
        tuning_items = defaultdict(set) # item name: options
        for op_name_type, quant_mode in op_dtype_dict.items():
            op_name, op_type = op_name_type
            # step1, set the default config for each op
            self.default_op_config[op_name_type] = tuning_space.set_deafult_config(op_name_type, quant_mode)
            # step2, collect all tuning items and their options
            op_item = tuning_space.op_items[op_name_type]
            quant_mode_item = tuning_space.query_quant_mode_item(op_name_type, quant_mode)
            for tuning_item in quant_mode_item.options:
                tuning_items[tuning_item.name] = tuning_items[tuning_item.name].union(tuning_item.options)
        self.tuning_items = tuning_items
    
    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        keys = self.tuning_items.keys()
        for vals in product(*self.tuning_items.values()):
            # traverse all possible combinations by model-wise level
            tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
            for op_name_type, quant_mode in self.op_dtype_dict.items():
                all_exist_flag = True
                for key, val in zip(keys, vals):
                    if not self.tuning_space.query_item_option(op_name_type, quant_mode, key, val):
                        all_exist_flag = False
                        tune_cfg[op_name_type] = self.default_op_config[op_name_type]
                        logger.debug(f"{op_name_type} dose not support {val} for {key}, \
                                     using the default config instead")
                        break
                if all_exist_flag:
                    tune_cfg[op_name_type] = OpTuningConfig(op_name_type[0], 
                                                            op_name_type[1], 
                                                            quant_mode, 
                                                            self.tuning_space,
                                                            dict(zip(keys, vals)))
            yield tune_cfg
                

class OpTypeWiseTuningSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(self,
                 tuning_space: TuningSpace,
                 tuning_items_priority: List[str],
                 tuning_order_lst: List[TuningOrder],
                 op_dtype_dict: Dict[tuple, str],
                 initial_op_tuning_cfg: Dict[tuple, OpTuningConfig]):
        """Op type wise tuning sampler.

        Args:
            tuning_space: Tuning space.
            tuning_items_priority: The priority to traverse the tuning items.
            tuning_order_lst: The tuning orders.
            op_dtype_dict: The (op name, op type) and its target data type.
            initial_op_tuning_cfg: The initial tuning config.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)
        tuning_items_priority = TUNING_ITEM_PRIORITY
        self.optype_quant_mode_option = {}
        self.optype_quant_mode_items_name = defaultdict(list)
        self.op_type_quant_mode_wise_combination = {}
        self.op_dtype_dict = op_dtype_dict
        self.default_op_config = {}
        for op_name_type, quant_mode in op_dtype_dict.items():
            self.default_op_config[op_name_type] = tuning_space.set_deafult_config(op_name_type, quant_mode)
            op_name, op_type = op_name_type
            op_type_quant_mode = (op_type, quant_mode)
            op_item = tuning_space.op_items[op_name_type]
            quant_mode_item = tuning_space.query_quant_mode_item(op_name_type, quant_mode)
            filtered_tuning_items = []
            for item_name in tuning_items_priority:
                item = quant_mode_item.get_option_by_name(item_name)
                if item:
                    if op_type_quant_mode not in self.optype_quant_mode_option:
                        self.optype_quant_mode_option[op_type_quant_mode] = defaultdict(list)
                    self.optype_quant_mode_option[op_type_quant_mode][item_name] += item.options
                    filtered_tuning_items.append(item)
            self.optype_quant_mode_items_name[op_type_quant_mode] = filtered_tuning_items

        for op_type_quant_mode, val in self.optype_quant_mode_option.items():
            options_lst = []
            # remove the duplicate options
            for _, item_options in val.items():
                seen = set()
                filter_options = [option for option in item_options if not (option in seen or seen.add(option))]
                options_lst.append(filter_options)
            op_type_quant_mode_vals = product(*options_lst)
            self.op_type_quant_mode_wise_combination[op_type_quant_mode] = op_type_quant_mode_vals

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
        for options_lst in product(*self.op_type_quant_mode_wise_combination.values()):
            for index, op_type_quant_mode in enumerate(self.op_type_quant_mode_wise_combination.keys()):
                for op_name_type, quant_mode in self.op_dtype_dict.items():
                    if op_name_type[1] == op_type_quant_mode[0] and quant_mode == op_type_quant_mode[1]:
                        op_quant_mode = self.op_dtype_dict[op_name_type]
                        op_tuning_items = [item.name for item in \
                                           self.optype_quant_mode_items_name[op_type_quant_mode]]
                        op_tuning_item_vals = options_lst[index]
                        all_exist_flag = True
                        for key, val in zip(op_tuning_items, op_tuning_item_vals):
                            if not self.tuning_space.query_item_option(op_name_type, quant_mode, key, val):
                                all_exist_flag = False
                                op_tuning_config = self.default_op_config[op_name_type]
                                logger.debug(f"{op_name_type} dose not support {val} for {key}, \
                                             using the default config instead")
                                break
                        if all_exist_flag:
                            op_tuning_config = OpTuningConfig(op_name_type[0], 
                                                              op_name_type[1], 
                                                              quant_mode, 
                                                              self.tuning_space,
                                                              dict(zip(op_tuning_items, op_tuning_item_vals)))
                        new_tune_cfg.update({op_name_type: op_tuning_config})
            yield new_tune_cfg




class OpWiseTuningSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(self,
                 tuning_space: TuningSpace,
                 tuning_items_priority: List[str],
                 tuning_order_lst: List[TuningOrder],
                 op_dtype_dict: Dict[tuple, str],
                 initial_op_tuning_cfg: Dict):
        """Op wise tuning config sampler.

        Args:
            tuning_space: Tuning space.
            tuning_items_priority: The priority to traverse the tuning items.
            tuning_order_lst: The tuning orders.
            op_dtype_dict: The (op name, op type) and its target data type.
            initial_op_tuning_cfg: The initial tuning config.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)
        tuning_items_priority = TUNING_ITEM_PRIORITY
        # query the combination of tuning items with according to the tuning items priority
        self.op_dtype_dict = op_dtype_dict
        self.op_options_combination = OrderedDict()
        self.op_tuning_items = {}
        for op_name_type, op_quant_mode in op_dtype_dict.items():
            op_item = tuning_space.op_items[op_name_type]
            quant_mode_op = tuning_space.query_quant_mode_item(op_name_type, op_quant_mode)
            filtered_tuning_items = []
            for item_name in tuning_items_priority:
                item = quant_mode_op.get_option_by_name(item_name)
                if item:
                    filtered_tuning_items.append(item)
            self.op_tuning_items[op_name_type] = filtered_tuning_items
            op_options_lst = product(*[item.options for item in filtered_tuning_items])
            self.op_options_combination[op_name_type] = op_options_lst

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
        for op_options_lst in product(*self.op_options_combination.values()):
            for index, op_name_type in enumerate(self.op_options_combination.keys()):
                op_quant_mode = self.op_dtype_dict[op_name_type]
                op_tuning_items = [item.name for item in self.op_tuning_items[op_name_type]]
                op_tuning_item_vals = op_options_lst[index]
                op_tuning_config = OpTuningConfig(op_name_type[0], op_name_type[1], 
                                                  op_quant_mode, self.tuning_space,
                                                  dict(zip(op_tuning_items, op_tuning_item_vals)))
                new_tune_cfg.update({op_name_type: op_tuning_config})
            yield new_tune_cfg
    
    def get_opwise_candidate(self):
        """Collect all op-wise setting.

        Returns:
            op_wise_configs: all op-wise setting.
        """
        op_wise_configs = OrderedDict()
        for op_name_type, op_quant_mode in self.op_dtype_dict.items():
            op_tuning_items = [item.name for item in self.op_tuning_items[op_name_type]]
            op_options = self.op_options_combination[op_name_type]
            op_wise_configs[op_name_type] = []
            for op_tuning_item_vals in op_options:
                op_tuning_config = OpTuningConfig(op_name_type[0], op_name_type[1], 
                                                  op_quant_mode, self.tuning_space,
                                                  dict(zip(op_tuning_items, op_tuning_item_vals)))
                op_wise_configs[op_name_type].append(op_tuning_config)
        return op_wise_configs


class FallbackTuningSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(self,
                 tuning_space: TuningSpace,
                 tuning_order_lst: List[TuningOrder],
                 initial_op_tuning_cfg: Dict[tuple, Any],
                 op_dtypes: Dict[str, str],
                 accumulate: bool,
                 skip_first: bool = True
                 ):
        """Sampler for generate the tuning config of fallback stage.

        Args:
            tuning_space: Tuning space.
            tuning_order_lst: The tuning orders.
            initial_op_tuning_cfg: The initial tuning config.
            op_dtypes: The (op name, op type) and its target data type.
            accumulate: Fallback accumulated or not.
            skip_first: Skip fallback the first op or not. Defaults to True.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)
        self.op_dtypes = op_dtypes
        self.accumulate = accumulate
        self.skip_first = skip_first

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
        skip_first = self.skip_first
        for op_name_type, target_dtype in self.op_dtypes.items():
            if not self.accumulate:
                new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
            new_op_config = OpTuningConfig(op_name_type[0], op_name_type[1], target_dtype, self.tuning_space)
            new_tune_cfg.update({op_name_type: new_op_config})
            if self.accumulate and skip_first:  # skip the first one
                skip_first = False
                continue
            logger.debug(f"fallback {op_name_type} to {target_dtype}")
            yield new_tune_cfg  # need to skip the first one
