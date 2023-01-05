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
from .strategy import strategy_registry, TuneStrategy
from ..utils import logger

from .utils.tuning_sampler import OpTypeWiseTuningSampler, FallbackTuningSampler, ModelWiseTuningSampler
from .utils.tuning_structs import OpTuningConfig
from .utils.tuning_space import TUNING_ITEMS_LST

@strategy_registry
class BasicTuneStrategy(TuneStrategy):
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
            BasicTuneStrategy,
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
        """The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.

        Yields:
            tune_config (dict): It's a dict containing the tuning configuration to run.
        """
        from copy import deepcopy
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        for calib_sampling_size in calib_sampling_size_lst:
            # Initialize the tuning config for each op according to the quantization approach 
            op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg = self.initial_tuning_cfg()
            # Optype-wise tuning tuning items: the algorithm/scheme/granularity of activation(weight)
            early_stop_tuning = False
            stage1_cnt = 0
            quant_ops = quant_mode_wise_items['static'] if 'static' in quant_mode_wise_items else []
            quant_ops += quant_mode_wise_items['dynamic'] if 'dynamic' in quant_mode_wise_items else []
            quant_ops += quant_mode_wise_items['fp8_e5m2'] if 'fp8_e5m2' in quant_mode_wise_items else []
            stage1_max = 1e9  # TODO set a more appropriate value
            op_wise_tuning_sampler = OpTypeWiseTuningSampler(tuning_space, [], [], 
                                                             op_item_dtype_dict, initial_op_tuning_cfg)

            for op_tuning_cfg in op_wise_tuning_sampler:
                stage1_cnt += 1
                if early_stop_tuning and stage1_cnt > stage1_max:
                    logger.info("Early stopping the stage 1.")
                    break
                op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
                op_tuning_cfg['bn_calib_sampling_size'] = self.bn_calib_sampling_size
                yield op_tuning_cfg

            if self.objectives.compare(self.last_tune_result, self.baseline):
                logger.info("Suggested FP8 op types are: {}; Accuracy is {}".format(alowed_types[:-1], self.last_tune_result))

            self.re_quant = True
            all_op_type = set()
            for k ,v in op_tuning_cfg.items():
                if len(k) != 2:
                    continue
                op_name, op_type = k
                all_op_type.add(op_type)
            tuning_op_types = copy.deepcopy(all_op_type)
            log_op_types = []
            if 'Conv2d' in all_op_type:
                log_op_types.append('Conv2d')
            if 'Linear' in all_op_type:
                log_op_types.append('Linear')

            # only conv, linear check
            tune_cfg = copy.deepcopy(op_tuning_cfg)
            alowed_types = log_op_types
            for k ,v in op_tuning_cfg.items():
                if len(k) != 2:
                    continue
                op_name, op_type = k
                if op_type not in alowed_types:
                    fp32_config = OpTuningConfig(op_name, op_type, 'fp32', self.tuning_space)
                    tune_cfg[k] = fp32_config
            yield tune_cfg

            if self.objectives.compare(self.last_tune_result, self.baseline):
                last_accu = self.last_tune_result[0]
                # per op type check
                accu_pass_op_types = []
                for op in tuning_op_types:
                    if op in ['Conv2d', 'Linear']:
                        continue
                    tune_cfg = copy.deepcopy(op_tuning_cfg)
                    alowed_types = log_op_types + [op]
                    for k ,v in op_tuning_cfg.items():
                        if len(k) != 2:
                            continue
                        op_name, op_type = k
                        if op_type not in alowed_types:
                            fp32_config = OpTuningConfig(op_name, op_type, 'fp32', self.tuning_space)
                            tune_cfg[k] = fp32_config
                    yield tune_cfg

                    if self.objectives.compare(self.last_tune_result, self.baseline):
                        accu_pass_op_types.append(op)
                        last_accu = self.last_tune_result[0]

                # accumulate op type check
                if accu_pass_op_types:
                    alowed_types = log_op_types + [accu_pass_op_types[0]]
                    if len(accu_pass_op_types) == 1:
                        logger.info("Suggested FP8 op types are: {}; Accuracy is {}".format(alowed_types, last_accu))
                        return

                    for op in accu_pass_op_types[1:]:
                        tune_cfg = copy.deepcopy(op_tuning_cfg)
                        alowed_types.append(op)
                        for k ,v in op_tuning_cfg.items():
                            if len(k) != 2:
                                continue
                            op_name, op_type = k
                            if op_type not in alowed_types:
                                fp32_config = OpTuningConfig(op_name, op_type, 'fp32', self.tuning_space)
                                tune_cfg[k] = fp32_config
                        yield tune_cfg

                        if not self.objectives.compare(self.last_tune_result, self.baseline):
                            logger.info("Suggested FP8 op types are: {}; Accuracy is {}".format(alowed_types[:-1], last_accu))
                            return
                        else:
                            last_accu = self.last_tune_result[0]
                    logger.info("Suggested FP8 op types are: {}; Accuracy is {}".format(alowed_types, last_accu))
                    return
                else:
                    logger.info("Suggested FP8 op types are: {}; Accuracy is {}".format(log_op_types, last_accu))
                    return
            elif 'Conv2d' in all_op_type and 'Linear' in all_op_type:
                count_linear = 0
                exempt_modules = [self.adaptor.first_conv, self.adaptor.last_linear]
                logger.info("Disable first conv and last linear:{}".format(exempt_modules))
                for k ,v in op_tuning_cfg.items():
                    if len(k) != 2:
                        continue
                    op_name, op_type = k
                    if op_type == 'Linear':
                        count_linear += 1
                    if op_name in exempt_modules:
                        fp32_config = OpTuningConfig(op_name, op_type, 'fp32', self.tuning_space)
                        tune_cfg[k] = fp32_config
                yield tune_cfg
                if count_linear == 1:
                    log_op_types.remove('Linear')
                logger.info("Suggested FP8 op types are: {}; Accuracy is {}".format(log_op_types, self.last_tune_result[0]))
                return
            else:
                logger.info("Suggested FP8 op types are: {}; Accuracy is {}".format(log_op_types, self.last_tune_result[0]))
                return

    def initial_dynamic_cfg_based_on_static_cfg(self, op_static_cfg:OpTuningConfig):
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
        
        