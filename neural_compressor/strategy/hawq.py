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
from typing import Dict, List, Optional, Any, Union, Callable, Set


class HessianTrace:
    """
    please refer to
    Yao, Zhewei, et al. "Pyhessian: Neural networks through the lens of the hessian." 2020 IEEE international conference on big data (Big data). IEEE, 2020.
    Dong, Zhen, et al. "Hawq-v2: Hessian aware trace-weighted quantization of neural networks." Advances in neural information processing systems 33 (2020): 18518-18529.
    https://github.com/openvinotoolkit/nncf/blob/develop/nncf/torch/quantization/hessian_trace.py
    """

    def __init__(self, model, dataloader, criterion=None):
        from torch.quantization.quantize_fx import fuse_fx
        self.model = fuse_fx(model.model)

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

    def get_qnt_weight_loss(self, weights_name):

        fp32_model = self.fp32model

        qnt_model = self.q_model

        # print(self.model.state_dict())
        for n, p in self.model.named_parameters():
            print(n)

        print("*" * 20)

        for n, p in self.q_model._model.named_parameters():
            print(n)
        pass

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

    def mapping_module_to_op(self, name):
        # length = len("_model.")
        # if len(name) < length:
        #     return name
        # else:
        return name

    def get_fused_mapping(self):
        model = self.model
        weights_info = dict(model.named_parameters())
        weight_to_op = {}
        for op_name, child in model.named_modules():
            if self.is_fused_module(child):
                for name, _ in child.named_children():
                    if op_name + "." + name + ".weight" in weights_info:  ##TODO check if this is right

                        weight_to_op[op_name + "." + name + ".weight"] = self.mapping_module_to_op(op_name)
                        break
            else:
                name = op_name + ".weight"
                if name in weights_info and name not in weight_to_op.keys():
                    weight_to_op[op_name + ".weight"] = op_name
        op_list = []
        for key in weight_to_op.keys():
            op_list.append(weight_to_op[key])
        return weight_to_op, op_list

    def get_device(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            return p.data.device

    def get_gradients(self, model, data, criterion, create_graph=False):
        model.zero_grad()
        input = data[0].to(self.device)
        ##self._input_shape = input.shape  ## for resetting input activation
        target = data[1].to(self.device)
        # if enable_act:
        #     input.requires_grad = True
        output = model(input)
        loss = criterion(output, target)
        # torch.autograd.backward(loss, create_graph=create_graph)
        loss.backward(create_graph=create_graph)
        gradients = []
        for n, p in model.named_parameters():
            if p.grad != None and n in self.weight_names:
                gradient = p.grad
                gradients.append(gradient + 0.0)  ## add 0 to create a copy
        model.zero_grad()
        return gradients

    # def get_params(self, model):
    #     parameters = [p for p in model.parameters() if p.requires_grad]
    #     return parameters

    def sample_rademacher(self, params):
        samples = []
        for param in params:
            r = torch.randint_like(param, high=2, device=self.device)
            r.masked_fill_(r == 0, -1)
            samples.append(r)
        return samples

    def get_vtHv_weight(self, params, num_samples):
        num_batches = (num_samples + self.dataloader.batchsize - 1) // self.dataloader
        v = self.sample_rademacher(params)
        H_v = [0] * len(v)
        cnt = 0
        for step, data in enumerate(self.dataloader):
            batch_size = data[0].shape[0]
            cnt += batch_size
            gradients = self.get_gradients(self.model, data, self.criterion, create_graph=True)
            H_v_one = torch.autograd.grad(gradients, params, v, only_inputs=True, retain_graph=False)
            H_v = [pre + cur * float(batch_size) for cur, pre in zip(H_v_one, H_v)]
            if step == num_batches - 1:
                break
        if cnt > 0:
            H_v = [item / cnt for item in H_v]
        v_t_H_v = torch.stack([torch.mean(h_v * v_t) for (h_v, v_t) in zip(H_v, v)])  ##maybe sum is better
        return v_t_H_v

    def get_vtHv_act(self, params, num_samples):
        v = self.sample_rademacher(params)
        H_v = [0] * len(v)
        cnt = 0
        for step, data in enumerate(self.dataloader):
            if cnt >= num_samples:
                break
            for i in range(self.dataloader.batchsize):  ##force to batchsize to be 1
                input = data[0][i:i + 1]
                target = data[1][i:i + 1]

                self.get_gradients(self.model, (input, target), self.criterion, create_graph=True)
                layer_acts = [self.layer_acts[key] for key in self.layer_acts.keys()]
                layer_act_gradients = [self.layer_acts_grads[key] for key in self.layer_acts.keys()]
                hv_one = torch.autograd.grad(layer_act_gradients, layer_acts, v, only_inputs=True, retain_graph=False)
                cnt += 1
                if cnt >= num_samples:
                    break

    def _get_act_grad_hook(self, name):
        def act_grad_hook(model, grad_input, grad_output):
            self.layer_acts_grads[name] = [grad_input, grad_output]

        return act_grad_hook

    def _get_enable_act_grad_hook(self, name):
        def enable_act_grad_hook(model, inputs, outputs):
            try:
                input = inputs[0]  ##TODO check whether this is right
            except:
                input = inputs

            if input.requires_grad is False:
                input.requires_grad = True
            self.layer_acts[name] = input

        return enable_act_grad_hook

    # def _get_disable_input_grad_hook(self, name):
    #     def disable_input_grad_hook(model, inputs, outputs):
    #         try:
    #             input = inputs[0]  ##TODO check whether this is right
    #         except:
    #             input = inputs
    #         if input.is_leaf == False:## you can only change requires_grad flags of leaf variables
    #             if input.requires_grad is True:
    #                 input.requires_grad = False
    #
    #
    #     return disable_input_grad_hook

    def _unregister_hook(self):
        for handel in self.hook_handles:
            handel.remove()

    def register_act_grad_hooks(self):
        for name, module in self.model.named_modules():
            if self.mapping_module_to_op(name) in self.op_list:
                hook_handle = module.register_forward_hook(self._get_enable_act_grad_hook(name))
                self.hook_handles.append(hook_handle)
                hook_handle = module.register_backward_hook(self._get_act_grad_hook(name))
                self.hook_handles.append(hook_handle)

    def reset_act_gradient_and_hooks(self):
        # tmp_input = torch.zeros(self._input_shape, device=self.device)
        # for name, module in self.model.named_modules():
        #     if name in self.op_list:
        #         hook_handle = module.register_forward_hook(self._get_disable_input_grad_hook(name))
        #         self.hook_handles.append(hook_handle)
        # self.model(tmp_input)
        self._unregister_hook()

    def get_params(self):
        weight_names = [n for n, p in self.model.named_parameters() if
                        p.requires_grad and "bias" not in n]  ##remove bias
        params = [p for n, p in self.model.named_parameters() if p.requires_grad and "bias" not in n]  ##remove bias
        self.weight_names = weight_names
        self.params = params

    def get_weight_traces(self, num_samples):

        layer_traces_per_iter = []
        prev_avg_model_trace = 0
        for i in range(self.max_iter):
            layer_traces = self.get_vtHv_weight(self.params, num_samples)
            layer_traces_per_iter.append(layer_traces)
            layer_traces_estimate = torch.mean(torch.stack(layer_traces_per_iter), dim=0)
            model_trace = torch.sum(layer_traces_estimate)
            diff_ratio = abs(model_trace - prev_avg_model_trace) / (prev_avg_model_trace + self.eps)
            if diff_ratio < self.tolerance and i > 10:  ##TODO magic number
                break
            if i == 50:  ##TODO for debug
                break
            prev_avg_model_trace = model_trace
        weight_name_to_traces = {}

        for weight_name, trace in zip(self.weight_names, layer_traces):
            weight_name_to_traces[weight_name] = trace
        op_name_to_trace = {}
        for weight_name in self.weight_names:
            op_name = self.weight_to_op[weight_name]
            op_name_to_trace[op_name] = weight_name_to_traces[weight_name]
        return op_name_to_trace
        return layer_traces_estimate

    def get_act_traces(self, num_samples):
        self.hook_handles = []
        self.layer_acts = {}
        self.layer_acts_grads = {}
        self.register_act_grad_hooks()
        for i in range(self.max_iter):
            pass

    def get_avg_traces(self, enable_act=True, num_samples=100):
        """
        Estimates average hessian trace for each parameter
        """

        assert num_samples > 0

        ##num_data_iter = self.op_cfgs_list[0]['calib_iteration']
        ##num_all_data = num_data_iter * self.dataloader.batch_size
        ##op_list = self.op_list
        ##TODO setting this in config
        self.get_params()
        # names = [n for n, p in self.model.named_parameters() if p.requires_grad and "bias" not in n]##remove bias
        # params = [p for n, p in self.model.named_parameters() if p.requires_grad and "bias" not in n]##remove bias

        ## handle activation
        if enable_act:
            self.get_act_traces(num_samples)
            ##change batchsize to 1

        #
        # layer_traces = layer_traces_estimate
        # if enable_act:
        #     self.reset_act_gradient_and_hooks()


##copy from torch.quantization._numeric_suite
def _find_match(
        str_list: Union[Dict[str, Any], List[str]], key_str: str,
        postfix: str,
) -> Optional[str]:
    split_str = key_str.split(".")
    if split_str[-1] == postfix:
        match_string = "".join(key_str.split(".")[0:-1])
        for s2 in str_list:
            pattern1 = "".join(s2.split(".")[0:-1])
            pattern2 = "".join(s2.split(".")[0:-2])
            if match_string == pattern1:
                return s2
            if match_string == pattern2:
                return s2

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        if postfix == "_packed_params":
            match_string = "".join(key_str.split(".")[0:-2])
            if len(match_string) == 0:
                return None
            for s2 in str_list:
                pattern1 = "".join(s2.split(".")[0:-1])
                pattern2 = "".join(s2.split(".")[0:-2])
                if match_string == pattern1:
                    return s2
                if match_string == pattern2:
                    return s2
        return None
    else:
        return None


##copy form torch.quantization._numeric_suite
def compare_weights(
        float_dict: Dict[str, Any], quantized_dict: Dict[str, Any]
) -> Dict[str, Dict[str, torch.Tensor]]:
    r"""Compare the weights of the float module with its corresponding quantized
    module. Return a dict with key corresponding to module names and each entry being
    a dictionary with two keys 'float' and 'quantized', containing the float and
    quantized weights. This dict can be used to compare and compute the quantization
    error of the weights of float and quantized models.

    Example usage::

        wt_compare_dict = compare_weights(
            float_model.state_dict(), qmodel.state_dict())
        for key in wt_compare_dict:
            print(
                key,
                compute_error(
                    wt_compare_dict[key]['float'],
                    wt_compare_dict[key]['quantized'].dequantize()
                )
            )

    Args:
        float_dict: state dict of the float model
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict with key corresponding to module names and each entry being
        a dictionary with two keys 'float' and 'quantized', containing the float and
        quantized weights
    """

    weight_dict: Dict[str, Dict] = {}
    for key in quantized_dict:
        match_key = _find_match(float_dict, key, "weight")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key]
            continue

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        match_key = _find_match(float_dict, key, "_packed_params")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key][0]

        # For LSTM
        split_str = key.split(".")
        if split_str[-1] == "param" and split_str[-3] == "_all_weight_values":
            layer = split_str[-2]
            module_name = ".".join(split_str[:-3])
            float_weight_ih_key = module_name + ".weight_ih_l" + layer
            float_weight_hh_key = module_name + ".weight_hh_l" + layer
            if float_weight_ih_key in float_dict and float_weight_hh_key in float_dict:
                weight_dict[key] = {}
                weight_dict[key]["float"] = float_dict[float_weight_ih_key]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][0].__getstate__()[0][0]
                )
                weight_dict[key]["float"] = float_dict[float_weight_hh_key]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][1].__getstate__()[0][0]
                )

    return weight_dict


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

        # import torch.quantization._numeric_suite as ns
        # self.model.eval()
        # fused_model = fuse_fx(self.model.model)
        # res = compare_weights(fused_model.state_dict(), self.q_model.state_dict())

        # Fallback the ops supported both static and dynamic from static to dynamic
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
        self._fp32_model.eval()
        ht = HessianTrace(self._fp32_model, self.calib_dataloader)

        q_model_state_dict = {
        }
        for key in self.q_model.state_dict().keys():
            length = len("_model.")
            new_key = key[length:]
            q_model_state_dict[new_key] = self.q_model.state_dict()[key]

        weight_quant_loss = compare_weights(ht.model.state_dict(), q_model_state_dict)

        op_to_traces = ht.get_avg_traces()
        if orig_eval == False:
            self._fp32_model.train()

        ordered_ops = sorted(op_to_traces.keys(),
                             key=lambda key: op_to_traces[key],
                             reverse=self.higher_is_better)
        # WA for add op type
        op_info_map = {}
        for op_info in list(initial_op_tuning_cfg.keys()):
            op_info_map[op_info[0]] = op_info  # op_name: (op_name, op_type)
        tmp_ordered_ops = [op_info_map[op_name] for op_name in ordered_ops]
        op_dtypes = OrderedDict(zip(tmp_ordered_ops, [target_dtype] * len(ordered_ops)))
        logger.info(f"Start to accumulate fallback to {target_dtype}.")

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
