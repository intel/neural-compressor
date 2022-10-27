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
from typing import Dict, Any, List
from .strategy import strategy_registry, TuneStrategy
from ..utils import logger

from .st_utils.tuning_sampler import OpTypeWiseTuningSampler, FallbackTuningSampler
from .st_utils.tuning_structs import OpTuningConfig

@strategy_registry
class MSETuneStrategy(TuneStrategy):
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
            MSETuneStrategy,
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

    def mse_metric_gap(self, fp32_tensor, dequantize_tensor):
        """Calculate the euclidean distance between fp32 tensor and int8 dequantize tensor

        Args:
            fp32_tensor (tensor): The FP32 tensor.
            dequantize_tensor (tensor): The INT8 dequantize tensor.
        """
        fp32_max = np.max(fp32_tensor)
        fp32_min = np.min(fp32_tensor)
        dequantize_max = np.max(dequantize_tensor)
        dequantize_min = np.min(dequantize_tensor)
        fp32_tensor = (fp32_tensor - fp32_min) / (fp32_max - fp32_min)
        dequantize_tensor = (dequantize_tensor - dequantize_min) / \
            (dequantize_max - dequantize_min)
        diff_tensor = fp32_tensor - dequantize_tensor
        euclidean_dist = np.sum(diff_tensor ** 2)
        return euclidean_dist / fp32_tensor.size

    def mse_impact_lst(self, op_list: List, fp32_model,  best_qmodel):
        """_summary_

        Args:
            op_list (List): [(op_name, op_type), ...]
            fp32_model: model before quantized
            current_best_model :  model after quantized
        """
        op_name_lst = [element[0] for element in op_list ]
        op_mapping = {}
        for (op_name, op_type) in list(op_list):
            op_mapping[op_name] = (op_name, op_type)
        current_best_tune_cfg = self._tune_cfg_converter(self.cur_best_tuning_cfg)
        fp32_dump_content = self.adaptor.inspect_tensor(fp32_model, 
            self.calib_dataloader, op_name_lst, [1], inspect_type='activation', 
            save_to_disk=True, save_path="./nc_workspace/", 
            quantization_cfg=current_best_tune_cfg)
        fp32_tensor_dict = fp32_dump_content['activation'][0]
        best_qmodel = self.q_model = self.adaptor.quantize(current_best_tune_cfg, self.model, \
                                                           self.calib_dataloader, self.q_func)
        quant_dump_content = self.adaptor.inspect_tensor(best_qmodel, 
            self.calib_dataloader, op_name_lst, [1], inspect_type='activation',
            save_to_disk=True, save_path="./nc_workspace/", 
            quantization_cfg=current_best_tune_cfg)
        dequantize_tensor_dict = quant_dump_content['activation'][0]
        ops_mse = {
            op: self.mse_metric_gap(
                list(fp32_tensor_dict[op].values())[0],
                list(dequantize_tensor_dict[op].values())[0]) for op in fp32_tensor_dict}
        ordered_op_names = sorted(ops_mse.keys(), key=lambda key: ops_mse[key], reverse=self.higher_is_better)
        
        ordered_op_name_types = [op_mapping[name] for name in ordered_op_names]
        return ordered_op_name_types


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
            # step1. collect the ops that support static and dynamic
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

            # step3. optype-wise tuning tuning items: the algorithm/scheme/granularity of activation(weight)
            early_stop_tuning = False
            stage1_cnt = 0
            int8_ops = quant_mode_wise_items['dynamic'] + quant_mode_wise_items['static']
            stage1_max = min(5, len(int8_ops))  # TODO set a more appropriate value
            op_wise_tuning_sampler = OpTypeWiseTuningSampler(tuning_space, [], [], 
                                                             op_item_dtype_dict, initial_op_tuning_cfg)
            for op_tuning_cfg in op_wise_tuning_sampler:
                stage1_cnt += 1
                if early_stop_tuning and stage1_cnt > stage1_max:
                    logger.info("Early stopping the stage 1.")
                    break
                op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
                yield op_tuning_cfg

            # step4. fallback the ops supported both static and dynamic from static to dynamic
            # tuning items: None
            static_dynamic_items = [item for item in tuning_space.query_items_by_quant_mode('static') if
                                    item in tuning_space.query_items_by_quant_mode('dynamic')]
            if static_dynamic_items:
                logger.info("Fallback all ops that support both dynamic and static to dynamic.")
            else:
                logger.info("Non ops that support both dynamic")

            def dynamic_op_tuning_cfg_from_static(op_tuning_cfg: OpTuningConfig):
                new_op_tuning_cfg = deepcopy(op_tuning_cfg)
                new_op_tuning_cfg.op_quant_mode = 'dynamic'
                return new_op_tuning_cfg

            new_op_tuning_cfg = deepcopy(self.cur_best_tuning_cfg)
            for item in static_dynamic_items:
                new_op_tuning_cfg[item.name] = dynamic_op_tuning_cfg_from_static(new_op_tuning_cfg[item.name])
            new_op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
            yield new_op_tuning_cfg

            best_op_tuning_cfg_stage1 = deepcopy(self.cur_best_tuning_cfg)

            # step5. fallback
            for target_dtype in ['bf16', 'fp32']:
                fallback_items_lst = [item for item in int8_ops if
                                    item in tuning_space.query_items_by_quant_mode(target_dtype)]
                if fallback_items_lst:
                    logger.info(f"Start to fallback op to {target_dtype} one by one.")
                    self._fallback_started()
                # replace it with sorted items list
                fallback_items_name_lst = [item.name for item in fallback_items_lst]
                # TODO check the best_qmodel
                ordered_op_name_types = self.mse_impact_lst(fallback_items_name_lst, self.model, self.best_qmodel)
                self.ordered_ops = [op_name for (op_name, op_type) in ordered_op_name_types]
                op_dtypes = OrderedDict(zip(ordered_op_name_types, [target_dtype] * len(fallback_items_name_lst)))
                initial_op_tuning_cfg = deepcopy(best_op_tuning_cfg_stage1)
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
                    ordered_ops = sorted(op_fallback_acc_impact.keys(), 
                                         key=lambda key: op_fallback_acc_impact[key],
                                         reverse=self.higher_is_better)
                    op_dtypes = OrderedDict(zip(ordered_ops, [target_dtype] * len(fallback_items_name_lst)))
                    logger.info(f"Start to accumulate fallback to {target_dtype}.")
                    initial_op_tuning_cfg = deepcopy(best_op_tuning_cfg_stage1)
                    fallback_sampler = FallbackTuningSampler(tuning_space, tuning_order_lst=[],
                                                            initial_op_tuning_cfg=initial_op_tuning_cfg,
                                                            op_dtypes=op_dtypes, accumulate=True)
                    for op_tuning_cfg in fallback_sampler:
                        op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
                        yield op_tuning_cfg
