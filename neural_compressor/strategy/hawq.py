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

import copy
import numpy as np
from collections import OrderedDict

import torch.nn

from .strategy import strategy_registry, TuneStrategy
from ..utils import logger

from .st_utils.tuning_sampler import OpTypeWiseTuningSampler, FallbackTuningSampler, ModelWiseTuningSampler
from .st_utils.tuning_structs import OpTuningConfig
from .st_utils.tuning_space import TUNING_ITEMS_LST
from torch.quantization.quantize_fx import fuse_fx
import torch.nn.intrinsic.quantized as nniq
from torch.fx import symbolic_trace, graph_module
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Any, Union, Callable, Set
@strategy_registry
class HawqTuneStrategy(TuneStrategy):
    """The basic tuning strategy which tunes the low precision model with below order.

    1. modelwise tuning for all quantizable ops.
    2. fallback tuning from bottom to top to decide the priority of which op has biggest impact
       on accuracy.
    3. incremental fallback tuning by fallbacking multiple ops with the order got from #2.

    Args:
        model (object):                        The FP32 model specified for low precision tuning.
        conf (Class):                          The Conf class instance initialized from user yaml
                                               config file.
        q_dataloader (generator):              Data loader for calibration, mandatory for
                                               post-training quantization.
                                               It is iterable and should yield a tuple (input,
                                               label) for calibration dataset containing label,
                                               or yield (input, _) for label-free calibration
                                               dataset. The input could be a object, list, tuple or
                                               dict, depending on user implementation, as well as
                                               it can be taken as model input.
        q_func (function, optional):           Reserved for future use.
        eval_dataloader (generator, optional): Data loader for evaluation. It is iterable
                                               and should yield a tuple of (input, label).
                                               The input could be a object, list, tuple or dict,
                                               depending on user implementation, as well as it can
                                               be taken as model input. The label should be able
                                               to take as input of supported metrics. If this
                                               parameter is not None, user needs to specify
                                               pre-defined evaluation metrics through configuration
                                               file and should set "eval_func" parameter as None.
                                               Tuner will combine model, eval_dataloader and
                                               pre-defined metrics to run evaluation process.
        eval_func (function, optional):        The evaluation function provided by user.
                                               This function takes model as parameter, and
                                               evaluation dataset and metrics should be
                                               encapsulated in this function implementation and
                                               outputs a higher-is-better accuracy scalar value.

                                               The pseudo code should be something like:

                                               def eval_func(model):
                                                    input, label = dataloader()
                                                    output = model(input)
                                                    accuracy = metric(output, label)
                                                    return accuracy
        dicts (dict, optional):                The dict containing resume information.
                                               Defaults to None.

    """

    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None, q_hooks=None):
        super(
            HawqTuneStrategy,
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
        from copy import deepcopy
        tuning_space = self.tuning_space
        calib_size = tuning_space.root_item.get_option_by_name('calib_sampling_size').options[0]  ##TODO suppoprt list

        # Initialize the tuning config for each op according to the quantization approach
        op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg = self.initial_tuning_cfg()
        # Optype-wise tuning tuning items: the algorithm/scheme/granularity of activation(weight)
        early_stop_tuning = True
        stage1_cnt = 0
        quant_ops = quant_mode_wise_items['static'] if 'static' in quant_mode_wise_items else []
        quant_ops += quant_mode_wise_items['dynamic'] if 'dynamic' in quant_mode_wise_items else []
        stage1_max = 1  # TODO set a more appropriate value
        op_wise_tuning_sampler = OpTypeWiseTuningSampler(tuning_space, [], [],
                                                         op_item_dtype_dict, initial_op_tuning_cfg)
        for op_tuning_cfg in op_wise_tuning_sampler:
            stage1_cnt += 1
            if early_stop_tuning and stage1_cnt > stage1_max:
                logger.info("Early stopping the stage 1.")
                break
            op_tuning_cfg['calib_sampling_size'] = calib_size
            yield op_tuning_cfg
        
        # Start compute the hessian trace
        # Fallback the ops supported both static and dynamic from static to dynamic
        quant_ops = quant_mode_wise_items['static'] if 'static' in quant_mode_wise_items else []
        quant_ops += quant_mode_wise_items['dynamic'] if 'dynamic' in quant_mode_wise_items else []

        target_dtype = "int8"  ##TODO support bf16
        target_type_lst = set(tuning_space.query_items_by_quant_mode(target_dtype))
        fp_op_list = [item.name for item in quant_ops if item in target_type_lst]
        # for n, p in self._fp32_model.named_modules():
        #     print(n)
        # for n, p in self._fp32_model.named_parameters():
        #     print(n)
        # # TODO uncomment it when algo ready.
        criterion=torch.nn.CrossEntropyLoss()
        op_to_traces = self.adaptor.calculate_hessian_trace(fp32_model = self._fp32_model, 
                                                            dataloader = self.calib_dataloader, 
                                                            q_model = self.q_model, 
                                                            criterion =criterion, # TODO replace it with user specify loss
                                                            enable_act = False)
        ordered_ops = sorted(op_to_traces.keys(),
                             key=lambda key: op_to_traces[key],
                             reverse=self.higher_is_better)
        # WA for add op type
        # print("ordered_ops:",ordered_ops)
        op_info_map = {}
        for op_info in list(initial_op_tuning_cfg.keys()):
            op_info_map[op_info[0]] = op_info  # op_name: (op_name, op_type)
        tmp_ordered_ops = [op_info_map[op_name] for op_name in ordered_ops]
        op_dtypes = OrderedDict(zip(tmp_ordered_ops, [target_dtype] * len(ordered_ops)))
        indx=0
        #defautly fallback 5 ops
        for i in op_dtypes.keys():
            op_dtypes[i]="fp32"
            indx=indx+1
            if indx>4:
                break
        print(op_dtypes)
        logger.info("hawq op_config:"+str(op_dtypes))
        logger.info(f"Start to accumulate fallback to {target_dtype}.")
        initial_op_tuning_cfg = deepcopy(op_tuning_cfg)
        fallback_sampler = FallbackTuningSampler(tuning_space, tuning_order_lst=[],
                                                 initial_op_tuning_cfg=op_tuning_cfg,
                                                 op_dtypes=op_dtypes, accumulate=True)
        for op_tuning_cfg in fallback_sampler:
            op_tuning_cfg['calib_sampling_size'] = calib_size
            yield op_tuning_cfg

    def initial_dynamic_cfg_based_on_static_cfg(self, op_static_cfg: OpTuningConfig):
        op_state = op_static_cfg.get_state()
        op_name = op_static_cfg.op_name
        op_type = op_static_cfg.op_type
        op_quant_mode = 'dynamic'
        tuning_space = self.tuning_space
        dynamic_state = {}
        for att in ['weight', 'activation']:
            if att not in op_state:
                continue
            for item_name, item_val in op_state[att].items():
                att_item = (att, item_name)
                if att_item not in TUNING_ITEMS_LST:
                    continue
                if tuning_space.query_item_option((op_name, op_type), op_quant_mode, att_item, item_val):
                    dynamic_state[att_item] = item_val
                else:
                    quant_mode_item = tuning_space.query_quant_mode_item((op_name, op_type), op_quant_mode)
                    tuning_item = quant_mode_item.get_option_by_name(att_item)
                    dynamic_state[att_item] = tuning_item.options[0] if tuning_item else None
        return OpTuningConfig(op_name, op_type, op_quant_mode, tuning_space, kwargs=dynamic_state)
