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

"""The `random` tuning strategy."""

import numpy as np
from .strategy import strategy_registry, TuneStrategy
from collections import OrderedDict

from .utils.tuning_sampler import OpWiseTuningSampler, FallbackTuningSampler
from .utils.tuning_structs import OpTuningConfig
from ..utils import logger

@strategy_registry
class RandomTuneStrategy(TuneStrategy):
    """The `random` tuning strategy, which using random searching in tuning space."""

    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None, q_hooks=None):
        """Construct a random tuning strategy.

        Args:
            model (object): The FP32 model specified for low precision tuning.
            conf (Conf | Config): The configurations for tuning, quantization, evaluation etc.
            q_dataloader (generator[input, label]): Data loader for calibration, mandatory for post-training quantization.
            q_func (function): Training function for quantization aware training. Defaults to None.
            eval_dataloader (generator[input, label]): Data loader for evaluation. Defaults to None.
            eval_func (function(model)->accuracy): The evaluation function provided by user. Defaults to None.
            dicts (dict): The dict containing resume information. Defaults to None.
        """
        super().__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts,
            q_hooks)

    def next_tune_cfg(self):
        """Generate and yield the next tuning config by random searching in tuning space.
        
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
        # collect the ops that support static and dynamic
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

        op_wise_tuning_sampler = OpWiseTuningSampler(tuning_space, [], [], 
                                                            op_item_dtype_dict, initial_op_tuning_cfg)
        op_tuning_cfg_lst = list(op_wise_tuning_sampler)
        op_tuning_cfg_cnt = len(op_tuning_cfg_lst)
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        calib_sampling_size_cnt = len(calib_sampling_size_lst)
        while True:
            calib_index = np.random.choice(calib_sampling_size_cnt)
            calib_sampling_size = calib_sampling_size_lst[calib_index]
            op_tuning_cfg_index = np.random.choice(op_tuning_cfg_cnt)
            op_tuning_cfg = op_tuning_cfg_lst[op_tuning_cfg_index]
            op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
            yield op_tuning_cfg
        return
