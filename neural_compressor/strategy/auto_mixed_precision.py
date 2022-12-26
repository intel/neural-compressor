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

"""The auto-mixed precision strategy."""

import copy
import numpy as np
from collections import OrderedDict
from .strategy import strategy_registry, TuneStrategy
from ..utils import logger

from .utils.tuning_sampler import OpTypeWiseTuningSampler, FallbackTuningSampler
from .utils.tuning_structs import OpTuningConfig


@strategy_registry
class AutoMixedPrecisionTuneStrategy(TuneStrategy):
    """Tuning strategy for auto mixed precision."""
    
    def next_tune_cfg(self):
        """Generate the next tuning config.
        
        Tuning configurations are generated according to the following rules:
        1. First, it tries to convert all ops into target date type as many as possible.
        2. If the accuracy does  not meets the requirements, it starts the stage of fallback 
            which converts ops into higher precision.
        
        Yields:
            tune_config (dict): A dict containing the tuning configuration.
        """
        from copy import deepcopy

        # filter quantization dtype
        # TODO align with the old mixed-precison
        target_dtype = self.cfg.graph_optimization.precisions if self.cfg.graph_optimization \
            else self.cfg.mixed_precision.precisions

        tuning_space = self.tuning_space
        initial_op_tuning_cfg = {}
        for item in tuning_space.root_item.options:
            if item.item_type == 'op':
                op_name, op_type = item.name
                initial_op_tuning_cfg[item.name] = OpTuningConfig(op_name, op_type, 'fp32', tuning_space)

        # step1. target_dtype AMAP, collect the ops that support target_dtype
        if not target_dtype:
            target_dtype = 'bf16'
        else:
            target_dtype = target_dtype[0]
        bf16_items = tuning_space.query_items_by_quant_mode(target_dtype)
        bf16_items_name = [item.name for item in bf16_items]
        op_tuning_cfg = deepcopy(initial_op_tuning_cfg)
        for op_name_type in bf16_items_name:
            op_tuning_cfg[op_name_type] = OpTuningConfig(op_name_type[0], op_name_type[1], target_dtype, tuning_space)
        calib_sampling_size = 1
        op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
        yield op_tuning_cfg

        # step2. fallback
        target_dtype = 'fp32'
        fallback_items_name_lst = bf16_items_name[::-1]
        if fallback_items_name_lst:
            logger.info(f"Start to fallback op to {target_dtype} one by one.")
            self._fallback_started()
        op_dtypes = OrderedDict(zip(fallback_items_name_lst, [target_dtype] * len(fallback_items_name_lst)))
        initial_op_tuning_cfg = deepcopy(op_tuning_cfg)
        fallback_sampler = FallbackTuningSampler(tuning_space, tuning_order_lst=[],
                                                initial_op_tuning_cfg=initial_op_tuning_cfg,
                                                op_dtypes=op_dtypes, accumulate=False)
        op_fallback_acc_impact = OrderedDict()
        for op_index, op_tuning_cfg in enumerate(fallback_sampler):
            op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
            yield op_tuning_cfg
            acc, _ = self.last_tune_result
            op_fallback_acc_impact[fallback_items_name_lst[op_index]] = acc

        # do accumulated fallback according to the order in the previous stage
        if len(op_fallback_acc_impact) > 0:
            ordered_ops = sorted(op_fallback_acc_impact.keys(), key=lambda key: op_fallback_acc_impact[key],
                                reverse=self.higher_is_better)
            op_dtypes = OrderedDict(zip(ordered_ops, [target_dtype] * len(fallback_items_name_lst)))
            logger.info("Start to accumulate fallback to {target_dtype}.")
            initial_op_tuning_cfg = deepcopy(op_tuning_cfg)
            fallback_sampler = FallbackTuningSampler(tuning_space, tuning_order_lst=[],
                                                    initial_op_tuning_cfg=initial_op_tuning_cfg,
                                                    op_dtypes=op_dtypes, accumulate=True)
            for op_tuning_cfg in fallback_sampler:
                op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
                yield op_tuning_cfg

    def traverse(self):
        """Traverse the tuning space according to auto-mixed precision strategy."""
        # get fp32 model baseline
        if self.baseline is None and (self.eval_dataloader or self.eval_func):
            logger.info("Get FP32 model baseline.")
            self.baseline = self._evaluate(self.model)
            # record the FP32 baseline
            self._add_tuning_history()

            if self.baseline:
                self.tune_data['baseline'] = self.baseline[0] if \
                    isinstance(self.baseline[0], list) else [self.baseline[0]]

                for name, data in zip(self.metric_name, self.tune_data['baseline']):
                    self.tune_data[name] = [data]

                if self.metric_weight:
                    self.tune_data['Weighted accuracy'] = \
                        [np.mean(np.array(self.tune_data['baseline']) * self.metric_weight)]
                    self.tune_data['baseline'] = self.tune_data['Weighted accuracy']

                baseline_msg = '[Accuracy:' + \
                    ''.join([' {:.4f}'.format(i) for i in self.tune_data['baseline']]) + \
                    ''.join([', {}: {:.4f}'.format(x,y) for x,y in zip( \
                    self.objectives.representation, self.baseline[1]) if x != 'Accuracy']) + ']'
            else: # pragma: no cover
                if self.metric_weight:
                    self.tune_data['Weighted accuracy'] = ['n/a']
                self.tune_data['baseline'] = ['n/a']

                for name, data in zip(self.metric_name, self.tune_data['baseline']):
                    self.tune_data[name] = ['n/a']
                baseline_msg = 'n/a'

            logger.info("FP32 baseline is: {}".format(baseline_msg))

        trials_count = 0
        for op_tuning_cfg in self.next_tune_cfg():
            # add tune_cfg here as quantize use tune_cfg
            tune_cfg = self._tune_cfg_converter(op_tuning_cfg)
            trials_count += 1
            tuning_history = self._find_tuning_history(tune_cfg)
            if tuning_history and trials_count < self.cfg.tuning.exit_policy.max_trials:
                self.last_tune_result = tuning_history['last_tune_result']
                self.best_tune_result = tuning_history['best_tune_result']
                logger.warn("Find evaluated tuning config, skip.")
                continue

            logger.debug("Dump current mixed precision configuration:")
            logger.debug(tune_cfg)
            self.last_qmodel = self.adaptor.quantize(
                tune_cfg, self.model, self.calib_dataloader, self.q_func)
            assert self.last_qmodel
            if self.eval_dataloader or self.eval_func:
                q_config = copy.deepcopy(self.last_qmodel.q_config)
                self.last_tune_result = self._evaluate(self.last_qmodel)
                self.cur_best_acc, self.cur_best_tuning_cfg = self.update_best_op_tuning_cfg(op_tuning_cfg)
                need_stop = self.stop(self.cfg.tuning.exit_policy.timeout, trials_count)
                # record the tuning history
                saved_tune_cfg = copy.deepcopy(tune_cfg)
                saved_last_tune_result = copy.deepcopy(self.last_tune_result)
                self._add_tuning_history(saved_tune_cfg, saved_last_tune_result, q_config=q_config)
            else:
                # If the eval_dataloader was not specified under the config yaml file,
                # We only converted the model with customized precisions.
                self.best_qmodel = self.last_qmodel
                need_stop = True

            if need_stop:
                break


