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
import torchvision


class HessianTrace:
    """
    please refer to
    Yao, Zhewei, et al. "Pyhessian: Neural networks through the lens of the hessian." 2020 IEEE international conference on big data (Big data). IEEE, 2020.
    Dong, Zhen, et al. "Hawq-v2: Hessian aware trace-weighted quantization of neural networks." Advances in neural information processing systems 33 (2020): 18518-18529.
    https://github.com/openvinotoolkit/nncf/blob/develop/nncf/torch/quantization/hessian_trace.py
    """

    def __init__(self, model, dataloader, criterion=None):
        self.model = model  ##TODO need to check fused or not
        self.dataloader = dataloader
        self.max_iter = 500
        self.tolerance = 1e-5
        self.eps = 1e-6
        self.index = 0
        self.device = self.get_device(self.model)
        self.criterion = criterion
        if self.criterion == None:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)  ##TODO need to set in config
        self.criterion = self.criterion.to(self.device)
        self.weight_to_op, self.op_list = self.get_fused_mapping()

    def is_fused_module(self, module):
        """This is a helper function for `_propagate_qconfig_helper` to detecte
           if this module is fused.
        Args:
            module (object): input module
        Returns:
            (bool): is fused or not
        """
        op_type = str(type(module))
        if 'fused' in op_type:
            return True
        else:
            return False

    def get_fused_mapping(self):
        model = self.model
        weights_info = dict(model.named_parameters())
        weight_to_op = {}
        for op_name, child in model.named_modules():
            if self.is_fused_module(child):
                for name, _ in child.named_children():
                    if op_name + "." + name + ".weight" in weights_info:  ##TODO check if this is right
                        weight_to_op[op_name + "." + name + ".weight"] = op_name
                        break
            else:
                if op_name + ".weight" in weights_info:
                    weight_to_op[op_name + ".weight"] = op_name
        op_list = []
        for key in weight_to_op.keys():
            op_list.append(weight_to_op[key])
        return weight_to_op, op_list

    def get_device(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            return p.data.device

    def get_gradients(self, model, data, criterion, create_graph=False, enable_act=False):
        model.zero_grad()
        input = data[0].to(self.device)
        target = data[1].to(self.device)
        if enable_act:
            input.requires_grad = True
        output = model(input)
        loss = criterion(output, target)
        loss.backward(create_graph=create_graph)
        gradients = []
        for n, p in model.named_parameters():
            if p.grad != None:
                gradient = p.grad
                gradients.append(gradient + 0.0) ## add 0 to create a copy
        model.zero_grad()
        return gradients

    def get_params(self, model):
        parameters = [p for p in model.parameters() if p.requires_grad]
        return parameters

    def sample_rademacher(self, params):
        samples = []
        for param in params:
            r = torch.randint_like(param, high=2, device=self.device)
            r.masked_fill_(r == 0, -1)
            samples.append(r)
        return samples

    def hutchinson_one_step(self, params, enable_act, num_batches):
        v = self.sample_rademacher(params)
        H_v = [0] * len(v)
        cnt = 0
        for step, data in enumerate(self.dataloader):
            batch_size = data[0].shape[0]
            cnt += batch_size
            gradients = self.get_gradients(self.model, data, self.criterion, create_graph=True, enable_act=enable_act)
            H_v_one = torch.autograd.grad(gradients, params, v, only_inputs=True, retain_graph=False)
            H_v = [pre + cur * float(batch_size) for cur, pre in zip(H_v_one, H_v)]
            if step == num_batches - 1:
                break
        if cnt > 0:
            H_v = [item / cnt for item in H_v]
        v_t_H_v = torch.stack([torch.mean(h_v * v_t) for (h_v, v_t) in zip(H_v, v)])  ##maybe sum is better
        return v_t_H_v

    def backward_hook(self, name):
        def grad_hook(model, grad_input, grad_output):
            self.layer_acts_grads[name] = [grad_input, grad_output]

        return grad_hook

    def forward_hook(self, name):
        def enable_input_grad_hook(model, inputs, outputs):
            try:
                input = inputs[0]  ##TODO check whether this is right
            except:
                input = inputs

            if input.is_leaf == False:
                if input.requires_grad is False:
                    input.requires_grad = True
                    self.layer_acts[name] = input

        return enable_input_grad_hook

    def register_hook(self):
        for name, module in self.model.named_modules():
            if name in self.op_list:
                forward_handle = module.register_forward_hook(self.forward_hook(name))
                backward_handle = module.register_backward_hook(self.backward_hook(name))
                self.hook_handlers.append(forward_handle)
                self.hook_handlers.append(backward_handle)

    def unregister_hook(self):
        for handel in self.hook_handlers:
            handel.remove()

    def get_avg_traces(self, enable_act=False, num_batches=2):
        """
        Estimates average hessian trace for each parameter
        """
        assert num_batches > 0
        if enable_act:
            self.hook_handlers = []
            self.layer_acts = {}
            self.layer_acts_grads = {}
            self.register_hook()
        ##num_data_iter = self.op_cfgs_list[0]['calib_iteration']
        ##num_all_data = num_data_iter * self.dataloader.batch_size
        ##op_list = self.op_list
        ##TODO setting this in config
        params = [p for p in self.model.parameters() if p.requires_grad]

        layer_traces_per_iter = []
        prev_avg_model_trace = 0
        for i in range(self.max_iter):
            layer_traces = self.hutchinson_one_step(params, enable_act, num_batches)
            layer_traces_per_iter.append(layer_traces)
            layer_traces_estimate = torch.mean(torch.stack(layer_traces_per_iter), dim=0)
            model_trace = torch.sum(layer_traces_estimate)
            diff_ratio = abs(model_trace - prev_avg_model_trace) / (prev_avg_model_trace + self.eps)
            if diff_ratio < self.tolerance and i > 10:  ##TODO magic number
                break
            prev_avg_model_trace = model_trace

        layer_traces = layer_traces_estimate
        self.unregister_hook()
        return layer_traces


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

        quant_ops = quant_mode_wise_items['static'] if 'static' in quant_mode_wise_items else []
        quant_ops += quant_mode_wise_items['dynamic'] if 'dynamic' in quant_mode_wise_items else []

        target_dtype = "fp32"  ##TODO support bf16
        target_type_lst = set(tuning_space.query_items_by_quant_mode(target_dtype))
        fp_op_list = [item.name for item in quant_ops if item in target_type_lst]
        # for n, p in self._fp32_model.named_modules():
        #     print(n)
        # for n, p in self._fp32_model.named_parameters():
        #     print(n)

        orig_eval = True
        if self._fp32_model.training:
            orig_eval = False
        self._fp32_model.train()
        ht = HessianTrace(self._fp32_model, self.calib_dataloader)
        ht.get_avg_traces()
        if orig_eval:
            self._fp32_model.eval()

        # tmp = 1
        # fallback_items_name_lst = [item.name for item in fallback_items_lst][::-1] # from bottom to up
        # ops_sensitivity = self.adaptor.get_hessian_trace(self._fp32_model,
        #                                                         self.calib_dataloader,
        #                                                         self.
        #                                                         method_args={'name': 'hessian_trace'})
        # tmp = 1

    def next_tune_cfg_bk(self):
        """The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.

        Yields:
            tune_config (dict): It's a dict containing the tuning configuration to run.
        """
        from copy import deepcopy
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options

        calib_sampling_size = calib_sampling_size_lst[0]
        # Initialize the tuning config for each op according to the quantization approach
        op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg = self.initial_tuning_cfg()
        # Optype-wise tuning tuning items: the algorithm/scheme/granularity of activation(weight)
        early_stop_tuning = False
        stage1_cnt = 0
        quant_ops = quant_mode_wise_items['static'] if 'static' in quant_mode_wise_items else []
        quant_ops += quant_mode_wise_items['dynamic'] if 'dynamic' in quant_mode_wise_items else []
        stage1_max = 1e9  # TODO set a more appropriate value
        op_wise_tuning_sampler = OpTypeWiseTuningSampler(tuning_space, [], [],
                                                         op_item_dtype_dict, initial_op_tuning_cfg)
        # for op_tuning_cfg in op_wise_tuning_sampler:
        #     stage1_cnt += 1
        #     if early_stop_tuning and stage1_cnt > stage1_max:
        #         logger.info("Early stopping the stage 1.")
        #         break
        #     op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
        #     yield op_tuning_cfg
        # Fallback the ops supported both static and dynamic from static to dynamic
        # Tuning items: None
        # if self.cfg.quantization.approach == 'post_training_auto_quant':
        #     static_dynamic_items = [item for item in tuning_space.query_items_by_quant_mode('static') if
        #                             item in tuning_space.query_items_by_quant_mode('dynamic')]
        #     if static_dynamic_items:
        #         logger.info("Fallback all ops that support both dynamic and static to dynamic.")
        #     else:
        #         logger.info("Non ops that support both dynamic")
        #
        #     new_op_tuning_cfg = deepcopy(self.cur_best_tuning_cfg)
        #     for item in static_dynamic_items:
        #         new_op_tuning_cfg[item.name] = self.initial_dynamic_cfg_based_on_static_cfg(
        #                                        new_op_tuning_cfg[item.name])
        #     new_op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
        #     yield new_op_tuning_cfg
        best_op_tuning_cfg_stage1 = deepcopy(self.cur_best_tuning_cfg)

        # Fallback
        for target_dtype in ['bf16', 'fp32']:
            target_type_lst = set(tuning_space.query_items_by_quant_mode(target_dtype))
            fallback_items_lst = [item for item in quant_ops if item in target_type_lst]
            if fallback_items_lst:
                logger.info(f"Start to fallback op to {target_dtype} one by one.")
                self._fallback_started()
            # fallback_items_name_lst = [item.name for item in fallback_items_lst][::-1] # from bottom to up
            ops_sensitivity = self.adaptor.calculate_op_sensitivity(self._fp32_model,
                                                                    self.calib_dataloader,
                                                                    method_args={'name': 'hessian_trace'})

            fallback_items_name_lst = sorted(ops_sensitivity, key=lambda items: items[1], reverse=True)

            op_dtypes = OrderedDict(zip(fallback_items_name_lst, [target_dtype] * len(fallback_items_name_lst)))
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
