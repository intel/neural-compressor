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
from collections import OrderedDict
import numpy as np
from .strategy import strategy_registry, TuneStrategy


@strategy_registry
class MSETuneStrategy(TuneStrategy):
    """The tuning strategy using MSE policy in tuning space.

       This MSE policy runs fp32 model and int8 model seperately to get all activation tensors,
       and then compares those tensors by MSE algorithm to order all ops with MSE distance for
       deciding the impact of each op to final accuracy.
       It will be used to define opwise tuningspace by priority.

    Args:
        model (object):                        The FP32 model specified for low precision tuning.
        conf (Conf):                           The Conf class instance initialized from user yaml
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

    def __init__(self, model, conf, dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None, q_hooks=None):
        self.ordered_ops = None
        super().__init__(
            model,
            conf,
            dataloader,
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

    def next_tune_cfg(self):
        """The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.

        """
        # Model wise tuning
        op_cfgs = {}
        best_cfg = None
        best_acc = 0

        for iterations in self.calib_iter:
            op_cfgs['calib_iteration'] = int(iterations)
            for combined_cfg in self.combined_model_wise_quant_cfgs:
                op_cfgs['op'] = OrderedDict()
                for op, op_cfg in self.opwise_quant_cfgs.items():
                    if op[1] in combined_cfg.keys() and len(op_cfg) > 0:
                        op_cfgs['op'][op] = copy.deepcopy(
                            self._get_common_cfg(combined_cfg[op[1]], op_cfg))
                    else:
                        op_cfgs['op'][op] = copy.deepcopy(
                            self.opwise_tune_cfgs[op][0])

                yield op_cfgs
                acc, _ = self.last_tune_result
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = copy.deepcopy(op_cfgs)

        if best_cfg is not None:
            # Inspect FP32 and dequantized tensor
            if self.ordered_ops is None:
                op_lists = self.opwise_quant_cfgs.keys()
                fp32_dump_content = self.adaptor.inspect_tensor(
                    self.model, self.calib_dataloader, op_lists, [1])
                fp32_tensor_dict = fp32_dump_content['activation'][0]
                best_qmodel = self.adaptor.quantize(best_cfg, self.model, self.calib_dataloader)
                quant_dump_content = self.adaptor.inspect_tensor(
                    best_qmodel, self.calib_dataloader, op_lists, [1])
                dequantize_tensor_dict = quant_dump_content['activation'][0]
                ops_mse = {
                    op: self.mse_metric_gap(
                        list(fp32_tensor_dict[op].values())[0],
                        list(dequantize_tensor_dict[op].values())[0]) for op in fp32_tensor_dict}
                self.ordered_ops = sorted(ops_mse.keys(), key=lambda key: ops_mse[key],
                                          reverse=True)

            if ops_mse is not None:
                ordered_ops = sorted(ops_mse.keys(), key=lambda key: ops_mse[key], reverse=True)
                op_cfgs = copy.deepcopy(best_cfg)
                for op in ordered_ops:
                    if not isinstance(op, tuple):
                        cfg_key = [item[0] for item in list(op_cfgs['op'].keys())]
                        op = list(op_cfgs['op'].keys())[cfg_key.index(op)]
                    old_cfg = copy.deepcopy(op_cfgs['op'][op])
                    op_cfgs['op'][op]['activation'].clear()
                    op_cfgs['op'][op]['activation']['dtype'] = 'fp32'
                    if 'weight' in op_cfgs['op'][op]:
                        op_cfgs['op'][op]['weight'].clear()
                        op_cfgs['op'][op]['weight']['dtype'] = 'fp32'
                    yield op_cfgs
                    acc, _ = self.last_tune_result
                    if acc <= best_acc:
                        op_cfgs['op'][op] = copy.deepcopy(old_cfg)
                    else:
                        best_acc = acc

                op_cfgs = copy.deepcopy(best_cfg)
                for op in ordered_ops:
                    op_cfgs['op'][op]['activation'].clear()
                    op_cfgs['op'][op]['activation']['dtype'] = 'fp32'
                    if 'weight' in op_cfgs['op'][op]:
                        op_cfgs['op'][op]['weight'].clear()
                        op_cfgs['op'][op]['weight']['dtype'] = 'fp32'
                    yield op_cfgs
        else:
            op_cfgs['op'] = OrderedDict()
            for op in self.opwise_tune_cfgs.keys():
                op_cfgs['op'][op] = copy.deepcopy(self.opwise_tune_cfgs[op][0])
            yield op_cfgs

        return
