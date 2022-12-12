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
from copy import deepcopy
import numpy as np
from collections import OrderedDict
from typing import Dict, Any, List
from .strategy import strategy_registry, TuneStrategy
from ..utils import logger
from time import time 

from .utils.tuning_sampler import OpTypeWiseTuningSampler, FallbackTuningSampler
from .utils.tuning_structs import OpTuningConfig
from .utils.helper import tuning_record_msg

@strategy_registry
class MSE_V2TuneStrategy(TuneStrategy):
    """The tuning strategy using MSE policy in tuning space.

       This MSE policy runs fp32 model and int8 model seperately to get all activation tensors,
       and then compares those tensors by MSE algorithm to order all ops with MSE distance for
       deciding the impact of each op to final accuracy.
       It will be used to define opwise tuningspace by priority.

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
        self.ordered_ops = None
        super(
            MSE_V2TuneStrategy,
            self).__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts,
            q_hooks)

    def __getstate__(self):
        for history in self.tuning_history:
            if self._same_yaml(history['cfg'], self.cfg):
                history['ordered_ops'] = self.ordered_ops
        save_dict = super().__getstate__()
        return save_dict

    def next_tune_cfg(self):
        """The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.

        Yields:
            tune_config (dict): It's a dict containing the tuning configuration to run.
        """

        best_op_tuning_cfg = None
        if len(self.metric_name) == 1 or self.metric_weight is not None:
            best_acc = float('-inf') if self.higher_is_better else float('inf')
        else:
            best_acc = [float('-inf') if higher_is_better else float('inf') for \
                higher_is_better in self.metric_criterion]

        from copy import deepcopy
        tuning_space = self.tuning_space
        initial_op_tuning_cfg = {}
        for item in tuning_space.root_item.options:
            if item.item_type == 'op':
                op_name, op_type = item.name
                initial_op_tuning_cfg[item.name] = OpTuningConfig(op_name, op_type, 'fp32', tuning_space)
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        for calib_sampling_size in calib_sampling_size_lst:
            # Collect the ops that support static and dynamic
            quant_mode_wise_items = OrderedDict()
            query_order = ['static', 'dynamic', 'bf16', 'fp16', 'fp32']
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

            # Optype-wise tuning 
            early_stop_tuning = True
            stage1_cnt = 0
            int8_ops = quant_mode_wise_items['dynamic'] + quant_mode_wise_items['static']
            stage1_max = 2  # TODO set a more appropriate value
            op_wise_tuning_sampler = OpTypeWiseTuningSampler(tuning_space, [], [], 
                                                             op_item_dtype_dict, initial_op_tuning_cfg)
            for op_tuning_cfg in op_wise_tuning_sampler:
                stage1_cnt += 1
                if early_stop_tuning and stage1_cnt > stage1_max:
                    logger.info("Early stopping the stage 1.")
                    break
                op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
                yield op_tuning_cfg

            # Fallback the ops supported both static and dynamic from static to dynamic
            static_dynamic_items = [item for item in tuning_space.query_items_by_quant_mode('static') if
                                    item in tuning_space.query_items_by_quant_mode('dynamic')]
            if static_dynamic_items:
                logger.info("Fallback all ops that support both dynamic and static to dynamic.")
            else:
                logger.info("No op support both dynamic and static")

            def dynamic_op_tuning_cfg_from_static(op_tuning_cfg: OpTuningConfig):
                new_op_tuning_cfg = deepcopy(op_tuning_cfg)
                new_op_tuning_cfg.op_quant_mode = 'dynamic'
                return new_op_tuning_cfg

            new_op_tuning_cfg = deepcopy(self.cur_best_tuning_cfg)
            for item in static_dynamic_items:
                new_op_tuning_cfg[item.name] = dynamic_op_tuning_cfg_from_static(new_op_tuning_cfg[item.name])
            new_op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
            yield new_op_tuning_cfg

            # Fallback one by one by op sensitivity(mse)
            # 1. while the accuracy requirements not met:  # to improve the accuracy
            #     1) calculate the sensitivity of int8 ops in current state. 
            #     2) fallback the op with higher sensitivity accumulatively
            # 2. after the accuracy requirements met:  # to improve the performance 
            #     1) calculate the sensitivity of fp32 ops in the current state
            #     2) re-quantize the op with lower sensitivity accumulatively
            tune_cfg = deepcopy(self.cur_best_tuning_cfg)
            requantize_cfg = deepcopy(self._tune_cfg_converter(self.cur_best_tuning_cfg))
            self.output_op_names = self.adaptor.get_output_op_names(self.cur_best_qmodel)
            self.confidence_batches = (self.cfg.tuning.strategy.confidence_batches
                                       if self.cfg.tuning.strategy.confidence_batches != None else 2)
            tune_cfg_backup = deepcopy(tune_cfg)
            quant_ops_in_tune_cfg = self._collect_ops_by_quant_mode(tune_cfg, 'dynamic') + \
                                    self._collect_ops_by_quant_mode(tune_cfg, 'static')
            op_quant_cfgs = {op_info: tune_cfg_backup[op_info] for op_info in quant_ops_in_tune_cfg}
            fallback_records = []
            self.re_quant = True
            while not self.objectives.compare(self.last_tune_result, self.baseline):
                # Record the time of calcutating the sensitivity
                start = time()
                ops_lst = self.adaptor.calculate_op_sensitivity(self.model, 
                                                                self.calib_dataloader, 
                                                                deepcopy(self._tune_cfg_converter(tune_cfg)), 
                                                                self.output_op_names,
                                                                self.confidence_batches,
                                                                fallback=True)
                logger.debug(f"*** The op sensitivity analysis took {time() - start:.2f}s.")
                select_op_info = ops_lst[0]
                logger.info(f"*** The op {select_op_info} have the highest sensitivity in the current state, \
                    fallback it to fp32.")
                tune_cfg[select_op_info] = OpTuningConfig(select_op_info[0], 
                                                            select_op_info[1], 
                                                            'fp32', 
                                                            self.tuning_space)
                # Record the fallback history
                if not fallback_records: 
                    fallback_records = [[select_op_info]]
                else:
                    fallback_records.append(fallback_records[-1] + [select_op_info])
                logger.debug(f"*** The fallback ops record: \n{tuning_record_msg(fallback_records)}")
                yield tune_cfg

            logger.info(f"*** The accuracy meeting the accuracy requirements, stop fallback ops.")
            while self.objectives.compare(self.last_tune_result, self.baseline):
                if len(fallback_records) == 0 or len(fallback_records[-1]) <= 1:
                    logger.info(f"*** Stop re-quant due to no int8 op or only 1 int8 op left.")
                    break
                logger.info(f"*** Start to re-quant the fallback op in the previous stage.")
                # Track the current fallback ops
                tmp_fallback_ops = fallback_records[-1] if fallback_records else [] 
                start = time()
                ops_lst = self.adaptor.calculate_op_sensitivity(self.model, 
                                                                self.calib_dataloader, 
                                                                deepcopy(self._tune_cfg_converter(tune_cfg)),
                                                                self.output_op_names, 
                                                                self.confidence_batches,
                                                                fallback=False,
                                                                requantize_cfgs=requantize_cfg['op'])
                logger.debug(f"*** The op sensitivity analysis took {time() - start:.2f}s.")
                if not ops_lst: 
                    logger.warning("No op to be requantized")
                    break
                for select_op_info in ops_lst:
                    #assert select_op_info in tmp_fallback_ops, f"{select_op_info} not in fallback list."
                    if select_op_info not in tmp_fallback_ops:
                        logger.debug(f"{select_op_info} not in fallback list.")
                        continue
                    
                    new_fallback_ops = deepcopy(tmp_fallback_ops)
                    new_fallback_ops.remove(select_op_info)
                    if new_fallback_ops not in fallback_records:
                        logger.info(f"*** The op {select_op_info} have the lowest sensitivity in the current state, \
                                    re-quantize it.")
                        tune_cfg[select_op_info] = op_quant_cfgs[select_op_info]
                        fallback_records.append(new_fallback_ops)
                        logger.debug(f"*** The fallback ops record: \n{tuning_record_msg(fallback_records)}")
                        yield tune_cfg
                        break
                    else:
                        logger.debug(f"*** Skip re-qaunt {select_op_info}, due the config has been evallated.")
                        continue
            self.re_quant = False
            logger.info(f"*** The accuracy not meeting the accuracy requirements, stop re-quantize ops.")