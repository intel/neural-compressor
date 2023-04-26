#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json

try:
    from neural_compressor.utils.utility import LazyImport

    torch = LazyImport('torch')
    from ...utils import logger
except:
    import torch
    import logging

    logger = logging.getLogger()
from collections import UserDict


def forward_wrapper(model, input, device='cpu'):
    if isinstance(input, dict) or isinstance(input, UserDict):
        if device == 'cpu':
            output = model(**input)
        else:  # pragma: no cover
            for inp in input.keys():
                input[inp] = input[inp].to(device) \
                    if isinstance(input[inp], torch.Tensor) else input[inp]
            output = model(**input)
    elif isinstance(input, list) or isinstance(input, tuple):
        if device == 'cpu':
            output = model(*input)
        else:  # pragma: no cover
            input = [inp.to(device) \
                         if isinstance(inp, torch.Tensor) else inp
                     for inp in input]  # pylint: disable=E1133
            output = model(*input)
    else:
        if device == 'cpu' or not isinstance(input, torch.Tensor):
            output = model(input)
        else:  # pragma: no cover
            input = input.to(device)  # pylint: disable=no-member
            output = model(input)
    return output


def model_forward(model, dataloader, iters, device):
    try:
        cnt = 0
        for idx, (input, label) in enumerate(dataloader):
            output = forward_wrapper(model, input, device)
            cnt += 1
            if cnt >= iters:
                break
    except Exception as e:
        cnt = 0
        for idx, input in enumerate(dataloader):
            output = forward_wrapper(model, input, device)
            cnt += 1
            if cnt >= iters:
                break


def quant_dequant_w(m, num_bits=8, scheme='sym'):  ##TODO take sym as default
    eps = torch.finfo(torch.float32).eps
    if isinstance(m, torch.nn.Linear):
        x = m.weight
        if scheme == 'sym':
            q_min, q_max = -2. ** (num_bits - 1), 2. ** (num_bits - 1) - 1.
            scale = torch.max(torch.abs(x), dim=1).values / (float(q_max - q_min) / 2)
        else:
            q_min, q_max = 0, 2. ** num_bits - 1.
            scale = (torch.max(x, dim=1).values - torch.min(x, dim=1).values) / (2 ** num_bits - 1)

        scale = torch.clip(scale, min=eps)

        if scheme == 'sym':
            bias = 0
        else:
            bias = torch.round(0 - (torch.min(x, dim=1).values) / scale)
            bias = bias.unsqueeze(dim=-1)
        scale = scale.unsqueeze(dim=-1)
        q_x = x / scale + bias
        q_x.clamp_(q_min, q_max).round_()
        return (q_x - bias) * scale
    elif isinstance(m, torch.nn.Conv2d):
        x = m.weight
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(-1, x.shape[-1])
        if scheme == 'sym':
            q_min, q_max = -2. ** (num_bits - 1), 2. ** (num_bits - 1) - 1.
            scale = torch.max(torch.abs(x), dim=0).values / (2 ** (num_bits - 1) - 1)
        else:
            q_min, q_max = 0, 2. ** num_bits - 1.
            scale = (torch.max(x, dim=0).values - torch.min(x, dim=0).values) / (2 ** num_bits - 1)
        scale = torch.clip(scale, min=eps)
        if scheme == 'sym':
            bias = 0
        else:
            bias = torch.round(0 - (torch.min(x, dim=0).values) / scale)
            bias = bias.unsqueeze(dim=0)
        scale = scale.unsqueeze(dim=0)

        q_x = x / scale + bias
        q_x.clamp_(q_min, q_max).round_()
        q_dq_x = (q_x - bias) * scale
        q_dq_x = q_dq_x.view(m.weight.shape[0], m.weight.shape[2], m.weight.shape[3], m.weight.shape[1])
        q_dq_x = torch.permute(q_dq_x, (0, 3, 1, 2))
        return q_dq_x
    else:
        logger.warning("unsupported layer type, please have a check")


def quant_dequant_x(x, min_x=None, max_x=None, num_bits=8):
    eps = torch.finfo(torch.float32).eps
    q_min, q_max = 0, 2. ** num_bits - 1.
    if max_x == None or min_x == None:
        max_x = torch.max(x)
        min_x = torch.min(x)
    else:
        max_x = torch.max(max_x)
        min_x = torch.min(min_x)
    scale = (max_x - min_x) / (2 ** num_bits - 1)
    scale = torch.clip(scale, min=eps)
    bias = torch.round(0 - min_x) / scale
    q_x = x / scale + bias
    q_x.clamp_(q_min, q_max).round_()
    return scale * (q_x - bias)


def get_module(model, key):
    """Get module from model by key name

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    attrs = key.split('.')
    module = model
    for attr in attrs:
        try:
            attr = int(attr)
            module = module[attr]
        except:
            module = getattr(module, attr)
    return module


def set_module(model, key, new_module):
    """Set new module into model by key name

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    attrs = key.split('.')
    module = model
    for attr in attrs[:-1]:
        try:
            attr = int(attr)
            module = module[attr]
        except:
            module = getattr(module, attr)
    setattr(module, attrs[-1], new_module)


class TorchSmoothQuant:
    """
    Fake input channel quantization, for more details please refer to
    [1] SmoothQuant: Accurate and Efficient
    Post-Training Quantization for Large Language Models
    [2] SPIQ: Data-Free Per-Channel Static Input Quantization
    Currently, we only handle the layers whose smooth scale could be absorbed, we will support other layers later.
    We only support inplace mode which means the model weights will be changed, you can call recover function
    to recover the weights if needed
    """

    def __init__(self, model, dataloader, q_func=None, traced_model=None):
        """
        :param model: Torch model :param dataloader: Calibration dataloader :param traced_model: A specific model
        shares the same architecture as the model and could be traced by torch.jit. If not supplied, we use model
        instead.
        """
        self.model = model
        device, dtype = self._get_device()
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.dataloader = dataloader
        self.q_func = q_func
        self.input_values = {}
        self.output_values = {}
        self.input_maxes = {}
        self.input_mins = {}
        self.input_maxes_abs = {}
        self.traced_model = traced_model
        if self.traced_model == None:
            self.traced_model = self.model
        self.weight_scale_info = {}
        self.absorb_scales_info = {}
        self.insert_mul = True
        self.allow_absorb = False
        self.self_absorb_layers = {}
        self.absorb_to_layer = {}

    def _get_device(self):
        """
        Get the model device
        :return:Model device
        """
        for _, p in self.model.named_parameters():
            return p.data.device, p.data.dtype

    def _save_input_pc_hook(self, name, percentile=100):
        """
        A forward hook to save input max of a module
        :param name: the module name
        :return: A hook function
        """

        def save_input_hook(module, inputs, outputs):
            if name not in self.input_maxes.keys():
                self.input_maxes[name] = []
                self.input_mins[name] = []
                self.input_maxes_abs[name] = []
            input = inputs[0]
            ##TODO check input channel is correct
            if len(module.weight.shape) == 4:  ##conv3d or conv1d not supported now, need better way
                input = input.permute(0, 2, 3, 1)
            input = input.reshape(-1, input.shape[-1])
            max_tensor = torch.max(input, dim=0)[0]
            min_tensor = torch.min(input, dim=0)[0]
            k_index = int(input.shape[0] * percentile / 100)
            res, _ = torch.kthvalue(torch.abs(input), k_index, dim=0)
            self.input_maxes_abs[name].append(res)
            self.input_maxes[name].append(max_tensor)
            self.input_mins[name].append(min_tensor)
            # self.input_values[name] = input
            # self.output_values[name] = outputs

        return save_input_hook

    def _save_input_output_hook(self, name):
        """
        A forward hook to save input and output values of a module
            param name: the module name
            return: A hook function
        """

        def save_input_output_hook(module, inputs, outputs):
            input = inputs[0]
            # if name in self.input_values:
            #     self.input_values[name].append(input)
            #     self.output_values[name].append(outputs)
            # else:
            cnt = 32
            if name in self.input_values.keys() and len(self.input_values[name]) < cnt:
                self.input_values[name].append(input)
                self.output_values[name].append(outputs)
            if name not in self.input_values.keys():
                self.input_values[name] = [input]  ##TODO save more,like 8
                self.output_values[name] = [outputs]  ##TODO do not save output

        return save_input_output_hook


    def _add_input_output_observer(self, input_output_modules):
        for key in input_output_modules.keys():
            hook_func = self._save_input_output_hook(key)
            hook_handle = input_output_modules[key].register_forward_hook(hook_func)
            self.hook_handles.append(hook_handle)

    def _add_min_max_observer(self, modules, percentile=100):
        """
        :param modules: the modules which the observer will insert to
        :return:
        """
        self.hook_handles = []
        for key in modules.keys():
            hook_func = self._save_input_pc_hook(key, percentile)
            hook_handle = modules[key].register_forward_hook(hook_func)
            self.hook_handles.append(hook_handle)
        # if input_output_modules:
        #     self._add_input_output_observer(input_output_modules)


    def _remove_observer(self):
        """
        remove the observer from the model
        :return:
        """
        for hook_handle in self.hook_handles:
            hook_handle.remove()

    def _calibrate(self, absorb_to_layer, calib_iter, percentile, save_input_output=False):
        """
        :param absorb_to_layer: A dict,key is the absorb layer, val is a list of the to be smoothed layer
        :param calib_iter: Data size for calibration
        :return: A dict that saved the layer name and the channe-wised max value info
        """
        layer_to_absorb = {}
        for key in absorb_to_layer:
            for layer_name in absorb_to_layer[key]:
                layer_to_absorb[layer_name] = key
        hook_module_names = [absorb_to_layer[key][0] for key in absorb_to_layer.keys()]
        hook_modules = {}

        for index, name in enumerate(hook_module_names):
            module = get_module(self.model, name)
            if module.__class__.__name__.split(".")[-1] in self.op_types:
                hook_modules[name] = module
        if len(hook_modules) == 0:
            return {}
        self._add_min_max_observer(hook_modules, percentile)
        if save_input_output:
            hook_modules_input_output = {}
            hook_layer_names=[]
            for key in self.absorb_to_layer:
                hook_layer_names += self.absorb_to_layer[key]
            for name in hook_layer_names:
                hook_modules_input_output[name] = get_module(self.model, name)
            self._add_input_output_observer(hook_modules_input_output)

        self._dump_min_max(calib_iter=calib_iter)
        self._remove_observer()
        return self.input_maxes_abs

    def _dump_min_max(self, calib_iter=100):
        """
        Dump min max per channel information, the min max value will be saved in input_maxes attribute
        :param calibration_method: only support min_max currently
        :param calib_iter: Sample size for calibration
        :return:
        """
        if self.q_func:
            self.q_func(self.model)
        else:
            assert self.dataloader, "Please set dataloader for calibration."
            model_forward(self.model, self.dataloader, calib_iter, self.device)
        ##stack
        for key in self.input_maxes.keys():
            max_val = self.input_maxes[key]
            max_val = torch.stack(max_val, dim=0)
            min_val = self.input_mins[key]
            min_val = torch.stack(min_val, dim=0)
            self.input_maxes[key] = torch.max(max_val, dim=0)[0]
            self.input_mins[key] = torch.min(min_val, dim=0)[0]
            ##abs_max_val = torch.abs(self.input_maxes[key])
            ##self.input_maxes_abs[key] = abs_max_val
            # self.input_maxes_abs[key] = torch.max(torch.stack(self.input_maxes_abs[key], dim=0), dim=0)[0]
            abs_max_val = torch.max(torch.abs(self.input_mins[key]), torch.abs(self.input_maxes[key]))
            ##abs_max_val = self.input_maxes[key] - self.input_mins[key]
            self.input_maxes_abs[key] = abs_max_val
        # for key in self.input_values.keys():
        #     self.input_values[key] = torch.cat(self.input_values[key], dim=0)  ##this may introduce memory issue
        #     self.output_values[key] = torch.cat(self.output_values[key], dim=0)

    def _reshape_in_channel_to_last(self, layer_name):
        """
        Move the input channel to the last dim
        :param layer_name: Layer name
        :return: The reshaped weight
        """
        weight = get_module(self.model, layer_name).weight  ##TODO oc*ic, support transposed conv
        if len(weight.shape) == 4:
            weight = weight.permute(0, 2, 3, 1)
            weight = weight.reshape(-1, weight.shape[-1])
        return weight

    def _reshape_scale_for_weight(self, layer, scale):
        """
        reshape the scale for weight input channel, depthwise output channel
        :param layer:  torch module
        :param scale: orig scale
        :return: reshaped scale
        """
        if isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:  ##only depthwise conv could hit here
            scale = scale.view(scale.shape[0], 1, 1, 1)  ##mount on output channel

        elif isinstance(layer, torch.nn.Conv2d):
            scale = scale.view(1, scale.shape[0], 1, 1)

        elif isinstance(layer, torch.nn.Linear):
            scale = scale.view(1, scale.shape[0])

        return scale

    def _reshape_scale_for_input(self, layer, scale):
        """
        reshape the scale for input feature in channel
        :param layer:
        :param scale:
        :return:
        """
        if isinstance(layer, torch.nn.Conv2d):
            scale = scale.view(1, scale.shape[0], 1, 1)

        elif isinstance(layer, torch.nn.Linear):
            scale = scale.view(1, scale.shape[0])

        return scale

    def _scale_layer_weight(self, layer_name, scale):  ##input channel
        """
        Scale the layer weights at input channel, depthwise conv output channel
        :param layer_name: The layer name
        :param scale: The scale to be multiplied
        :return:
        """
        layer = get_module(self.model, layer_name)
        if layer.__class__.__name__ == "SQLinearWrapper":
            layer = layer.sq_linear
        scale = self._reshape_scale_for_weight(layer, scale)
        layer.weight = torch.nn.Parameter(layer.weight * scale)
        return scale

    def _absorb_scales(self, layer_name, scale, alpha=0.5):  ##output channel
        """
        Absorb the scale to the layer at output channel
        :param layer_name: The module name
        :param scale: The scale to be absorbed
        :param alpha_key: The alpha passed to SQLinearWrapper
        :return:
        """
        layer = get_module(self.model, layer_name)
        if self.insert_mul:
            if layer.__class__.__name__ == "SQLinearWrapper":
                set_module(self.model, layer_name, layer.sq_linear)  ##recover
            else:
                from .model_wrapper import SQLinearWrapper
                input_minmax = [self.input_mins[layer_name], self.input_maxes[layer_name]]
                new_module = SQLinearWrapper(layer, scale, input_minmax, alpha)
                set_module(self.model, layer_name, new_module)

        elif self.allow_absorb:
            if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.GroupNorm) or \
                    isinstance(layer, torch.nn.InstanceNorm2d):
                if layer.affine:
                    layer.weight *= scale
                    layer.bias *= scale
                else:
                    layer.affine = True
                    weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                    layer.weight = torch.nn.Parameter(
                        weight, requires_grad=False)
                    bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                    layer.bias = torch.nn.Parameter(bias, requires_grad=False
                                                    )
            elif isinstance(layer, torch.nn.LayerNorm):
                if layer.elementwise_affine:
                    layer.weight *= scale
                    layer.bias *= scale
                else:
                    layer.elementwise_affine = True
                    weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                    layer.weight = torch.nn.Parameter(
                        torch.ones(weight, requires_grad=False))
                    bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                    layer.bias = torch.nn.Parameter(
                        bias, requires_grad=False)

            elif isinstance(layer, torch.nn.Conv2d):
                ##the order could not be changed
                if hasattr(layer, "bias") and (layer.bias != None):
                    layer.bias *= scale
                scale = scale.view(scale.shape[0], 1, 1, 1)
                layer.weight *= scale

            elif isinstance(layer, torch.nn.Linear):
                if hasattr(layer, "bias") and (layer.bias != None):
                    layer.bias *= scale
                scale = scale.view(scale.shape[0], 1)
                layer.weight *= scale

            elif layer.__class__.__name__ == "LlamaRMSNorm" \
                    or layer.__class__.__name__ == "T5LayerNorm":  ##quite tricky
                layer.weight *= scale

            else:
                logger.warning(f"found unsupported layer {type(layer)}, try to multiply scale to "
                               f"weight and bias directly, this may introduce accuracy issue, please have a check ")
                if hasattr(layer, "weight") and layer.weight != None:
                    layer.weight *= scale
                if hasattr(layer, "bias") and layer.bias != None:
                    layer.bias *= scale

    def _cal_scale(self, input_max, weights, alpha, scale_type="orig"):

        if scale_type == "orig":  # same as the paper
            weights = torch.cat(weights, dim=0)
            weight_max = torch.max(torch.abs(weights), dim=0)[0]
            input_power = torch.pow(input_max, alpha)
            logger.debug(f"{max(input_max)}, {min(input_max)}")
            weight_power = torch.pow(weight_max, 1 - alpha)
            scale = torch.clip(input_power / weight_power, min=1e-5)
            scale[input_power == 0] = 1.0
            return scale
        if scale_type == "code_2":
            weights = torch.cat(weights, dim=0)
            weight_oc_max = torch.max(torch.abs(weights), dim=1)[0]
            weight_oc_max[weight_oc_max == 0] = 1e-5
            weight_ratio = torch.abs(weights) / (weight_oc_max.unsqueeze(-1))
            max_weight_ratio = torch.max(weight_ratio, dim=0)[0]
            input_power = torch.pow(input_max, alpha)
            weight_power = torch.pow(max_weight_ratio, 1 - alpha)
            scale = torch.clip(input_power / weight_power, min=1e-5)
            scale[input_power == 0] = 1.0
            return scale
        if scale_type == "code_3":
            ##weights = torch.cat(weights, dim=0)
            mask_weights = []
            for weight in weights:
                weight_oc_max = torch.max(torch.abs(weight), dim=1)[0]
                weight_oc_max[weight_oc_max == 0] = 1e-5
                weight_ratio = torch.abs(weight) / (weight_oc_max.unsqueeze(-1))
                mask = weight_ratio == 1
                mask_weight = torch.abs(weight) * mask
                mask_1_cnt = torch.sum(mask, dim=0)
                mask_1_cnt = mask_1_cnt.to(torch.float32)
                mask_1_cnt[mask_1_cnt == 0] = 1e-5
                mask_weight = torch.sum(mask_weight, dim=0)
                mask_weight = mask_weight / mask_1_cnt
                zero_index = mask_weight == 0
                max_value, index = torch.max(weight_ratio[:, zero_index], dim=0)
                other_weight_value = (torch.abs(weight))[:, zero_index][index, torch.arange(len(index))]
                mask_weight[zero_index] = other_weight_value
                mask_weight = mask_weight.unsqueeze(dim=0)
                mask_weights.append(mask_weight)
            mask_weights = torch.cat(mask_weights, dim=0)
            mask_weights = torch.mean(mask_weights, dim=0)
            input_power = torch.pow(input_max, alpha)
            weight_power = torch.pow(mask_weights, 1 - alpha)
            scale = torch.clip(input_power / weight_power, min=1e-5)
            scale[input_power == 0] = 1.0
            return scale

    def _adjust_parameters(self, absorb_to_layer, input_maxes, alpha=0.5):
        """
        adjust the weights and biases
        :param absorb_to_layer: A dict mapping absorb layer to smooth quantized layer
        :param input_maxes: The channel-wise input max info for layers
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, a float of a dict
        :return:
        """
        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        weight_scales_info = {}
        absorb_scales_info = {}
        for index, key in enumerate(absorb_to_layer.keys()):
            if isinstance(alpha, float):
                alpha_key = alpha
            elif isinstance(alpha, dict):
                alpha_key = alpha[key]
            input_max = absorb_to_input_maxes[key]
            layers = absorb_to_layer[key]
            weights = []
            for layer in layers:
                weight = self._reshape_in_channel_to_last(layer)
                weights.append(weight)

            # weight_max_per_channel = torch.max(torch.abs(weights), dim=0)[0]
            # input_power = torch.pow(input_max, alpha_key)
            # logger.debug(f"{max(input_max)}, {min(input_max)}")
            # weight_power = torch.pow(weight_max_per_channel, 1 - alpha_key)
            # # logger.info(f"{absorb_to_layer[key][0]} layer sparsity is
            # # {1.0-torch.count_nonzero(input_power)/input_power.numel()}")
            #
            # scale = torch.clip(input_power / weight_power, min=1e-5)
            # scale[input_power == 0] = 1.0
            scale = self._cal_scale(input_max, weights, alpha_key)

            self._absorb_scales(key, 1.0 / scale, alpha_key)
            absorb_scales_info[key] = 1.0 / scale
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                self._scale_layer_weight(layer_name, scale)
                weight_scales_info[layer_name] = scale
        return weight_scales_info, absorb_scales_info

    def _check_need_calibration(self, alpha, percentile, op_types,
                                scales_per_op, calib_iter):
        """
        check need calibration or not
        :param alpha: current alpha
        :param percentile: current percentile
        :param op_types: current op_types
        :param scales_per_op: current scales_per_op
        :param calib_iter:: current scales_per_op
        :return:
        """
        need_calib = True
        if len(self.input_maxes) == 0:  ## the first time
            need_calib = True
            self.alpha = alpha
            self.percentile = percentile
            self.op_types = op_types
            self.scales_per_op = scales_per_op
            self.calib_iter = calib_iter
            return need_calib

        if self.percentile == percentile and self.op_types == op_types \
                and self.scales_per_op == scales_per_op and self.calib_iter == calib_iter:
            if isinstance(alpha, float):
                need_calib = False
            elif self.alpha == "auto":
                need_calib = False

        self.alpha = alpha
        self.percentile = percentile
        self.op_types = op_types
        self.scales_per_op = scales_per_op
        self.calib_iter = calib_iter
        return need_calib

    def _get_auto_loss(self, output, output_q, loss_type="mse", loss_alpha=1.0):
        if loss_type == "mse":
            return torch.mean((output - output_q) ** 2)
        elif loss_type == "mean_scale_error":
            output[output == 0] = 1e-5
            loss = torch.mean(torch.pow(torch.abs(output - output_q) / torch.abs(output), loss_alpha))
            return loss


    # def _auto_tune_alpha_save_memory(self, input_maxes, alpha_space, attn_method):
    #     pass

    def _auto_tune_alpha(self, input_maxes, alpha_min=0.3, alpha_max=0.7, alpha_step=0.05, attn_method='min'):
        """
        Perform alpha-tuning to obtain layer-wise optimal alpha values and adjust parameters accordingly.
        input_maxes:
        alpha_min: min value of alpha search space.
        alpha_max: max value of alpha search space.
        alpha_step: step size of alpha search space.
        attn_method: criterion method used on attention ops; currently min, max and mean are supported.
        """
        logger.info("auto tuning alpha")
        import copy
        alpha_scale = 100
        alpha_space = list(range(round(alpha_min * alpha_scale), round((alpha_max + alpha_step) * alpha_scale),
                                 round(alpha_step * alpha_scale)))
        alpha_space = [alpha / alpha_scale for alpha in alpha_space]

        ans_layer2absorb, self.layer_to_absorb, ans = {}, {}, {}
        ## Searching optimal alphas
        for idx, absorb_key in enumerate(self.absorb_to_layer):

            loss_all_layers = {}
            for layer_key in self.absorb_to_layer[absorb_key]:
                loss_alpha = {}
                for alpha in alpha_space:
                    self.weight_scale_info, self.absorb_scales_info = self._adjust_parameters(
                        {absorb_key: self.absorb_to_layer[absorb_key]},
                        {self.absorb_to_layer[absorb_key][0]: input_maxes[self.absorb_to_layer[absorb_key][0]]}, alpha)
                    input_of_ops, output_of_ops = self.input_values[layer_key], self.output_values[layer_key]
                    loss = 0
                    input_scale = self._reshape_scale_for_input(get_module(self.model, layer_key),
                                                                self.absorb_scales_info[absorb_key])
                    layer = get_module(self.model, layer_key)
                    if layer.__class__.__name__ == "SQLinearWrapper":
                        layer = layer.sq_linear
                    weight_qdq = quant_dequant_w(layer)
                    layer_cp = copy.deepcopy(layer)
                    layer_cp.weight.data = weight_qdq
                    for input_of_op, output_of_op in zip(input_of_ops, output_of_ops):
                        input_of_op_q = quant_dequant_x(input_of_op * input_scale, self.input_mins[
                            self.absorb_to_layer[absorb_key][0]] * input_scale, self.input_maxes[
                                                            self.absorb_to_layer[absorb_key][0]] * input_scale,
                                                        )
                        output_of_op_q = layer_cp(input_of_op_q)
                        loss += self._get_auto_loss(output_of_op, output_of_op_q)
                    self.recover()
                    loss_alpha[alpha] = loss
                    if layer_key not in ans:  # Update alpha results
                        ans[layer_key] = alpha
                    else:
                        ans[layer_key] = alpha if loss < loss_alpha[ans[layer_key]] else ans[layer_key]
                loss_all_layers[layer_key] = loss_alpha
                if absorb_key not in ans_layer2absorb:
                    ans_layer2absorb[absorb_key] = ans[layer_key]
                else:
                    if attn_method == 'max':
                        ans_layer2absorb[absorb_key] = max(ans_layer2absorb[absorb_key], ans[layer_key])
                    elif attn_method == 'min':
                        ans_layer2absorb[absorb_key] = min(ans_layer2absorb[absorb_key], ans[layer_key])
                    elif attn_method == 'mean':
                        pass
            if attn_method == 'mean':
                mean_loss = {}
                for alpha in alpha_space:
                    mean_loss[alpha] = 0
                    for key in loss_all_layers.keys():
                        mean_loss[alpha] += loss_all_layers[key][alpha]
                min_alpha = min(mean_loss, key=mean_loss.get)
                if len(loss_all_layers) > 1:
                    ans_layer2absorb[absorb_key] = min_alpha
        logger.info("auto tuning alpha done")
        for key in ans_layer2absorb.keys():
            logger.info(f"{key}:{ans_layer2absorb[key]}")

        return ans_layer2absorb

    def transform(self, alpha=0.5, folding=False, percentile=100, op_types=['Linear', 'Conv2d'],
                  scales_per_op=False, calib_iter=100,
                  auto_alpha_args={'alpha_min': 0.3, 'alpha_max': 1.0, 'alpha_step': 0.05, 'attn_method': 'min'}):
        """
        The main entry of smooth quant
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, please refer
        to the paper for more details
        :param folding: whether insert mul(False) or just allow foldable layers(True) for SmoothQuant
        :param percentile: remove the activation outlier when calculating the scale
        :param op_types: The op typed to be smooth quantized
        :param scales_per_op: Not supported now
        :param calib_iter: Data size for calibration
        :return: A FP32 model with the same architecture as the orig model but with different weight which will be
        benefit to quantization
        """
        logger.info("call new sq")
        if folding:
            self.insert_mul, self.allow_absorb = False, True
        else:
            self.insert_mul, self.allow_absorb = True, False
        if isinstance(alpha, float) and (alpha < 0 or alpha > 1):
            logger.warning("alpha should be a float value in [0, 1] or 'auto' ")
            if alpha < 0:
                alpha = 0
                logger.warning("reset alpha to 0 ")
            if alpha > 1.0:
                alpha = 1.0
                logger.warning("reset alpha to 1.0 ")

        if not isinstance(self.model, torch.nn.Module):
            logger.warning("smooth quant is ignored since the model is not a torch module")
            return self.model
        self.recover()
        need_calibration = self._check_need_calibration(alpha, percentile, op_types, scales_per_op, calib_iter)
        with torch.no_grad():
            input_maxes_abs = self.input_maxes_abs
            if need_calibration:  ##avoid multiple calibaration during tuning if the only difference is alpha
                if self.insert_mul:
                    self.self_absorb_layers = self._get_all_layer_names()  # TODO: only support linear now.
                if self.allow_absorb:
                    self.absorb_to_layer, no_absorb_layers = self._trace(
                        op_types)  ##TODO we need to insert mul layer for no_absorb_layers later
                    if self.absorb_to_layer == None and no_absorb_layers == None:
                        logger.warning("sorry, could not trace the model, smooth quant is ignored")
                        logger.warning("if you are using huggingface model,"
                                       "you could set torchscript to True "
                                       "when loading the model")
                        return self.model

                # remove self.self_absorb_layers if it exists in self.absorb_to_layer
                for k, v in self.absorb_to_layer.items():
                    for i in v:
                        if i in self.self_absorb_layers:
                            self.self_absorb_layers.pop(i)
                self.absorb_to_layer.update(self.self_absorb_layers)


                if self.absorb_to_layer == None and no_absorb_layers == None:
                    logger.warning("sorry, could not trace the model, smooth quant is ignored")
                    logger.warning("if you are using huggingface model,"
                                   "you could set torchscript to True ")
                    return self.model
                save_input_output = False
                if alpha == "auto":
                    save_input_output = True

                input_maxes_abs = self._calibrate(self.absorb_to_layer, calib_iter, percentile, save_input_output)
                if alpha == 'auto':
                    self.alpha_per_layer = self._auto_tune_alpha(input_maxes_abs, **auto_alpha_args)  ##save the alpha

            if alpha == 'auto':
                alpha = self.alpha_per_layer

            self.weight_scale_info, self.absorb_scales_info = self._adjust_parameters(self.absorb_to_layer,
                                                                                      input_maxes_abs, alpha)
            self.input_values, self.output_values = {}, {}
            return self.model

    def recover(self):
        """
        recover the model weights
        :return:
        """
        with torch.no_grad():
            for key in self.weight_scale_info:
                self._scale_layer_weight(key, 1.0 / self.weight_scale_info[key])
            for key in self.absorb_scales_info:
                self._absorb_scales(key, 1.0 / self.absorb_scales_info[key])
            self.weight_scale_info = {}  ##clear the data
            self.absorb_scales_info = {}

    def _get_all_layer_names(self, op_types=['Linear']):
        """
        Try the model to find the layers which can be smooth quantized.
        :param op_types: The op types to be smooth quantized
        :return:
        self_absorb_layer: A dict, absorb layer name (itself): layers to be smooth quantized
        """
        self_absorb_layer = {}
        for name, module in self.model.named_modules():
            for op_type in op_types:
                if op_type == str(module.__class__.__name__):
                    self_absorb_layer[name] = [name]
        return self_absorb_layer

    def _trace(self, op_types):
        """
        Try the model to find the layers which can be smooth quantized.
        :param op_types: The op types to be smooth quantized
        :return:
        absorb_to_layer: A dict, absorb layer name:layers to be smooth quantized
        no_absorb_layers: A list saving the layers which could not find the absorb layer
        """
        tg = GraphTrace()
        for idx, input in enumerate(self.dataloader):
            example_inputs = input
            break
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(self.traced_model, example_inputs, op_types)
        return absorb_to_layer, no_absorb_layers


def get_parent(node):
    if node.inputs() == None:
        return None
    return list(node.inputs())[0].node()


class GraphTrace:
    """
    """

    def __init__(self):
        self.supported_torch_module_to_aten = {
            "Linear": "aten::linear",
            "Conv2d": "aten::_convolution",
            "ConvTranspose2d": "aten::_convolution",
            "LayerNorm": "aten::layer_norm",
            "BatchNorm2d": "aten::batch_norm",
            "GroupNorm": "aten::group_norm",
            "InstanceNorm2d": "aten::instance_norm",
            "LlamaRMSNorm": "aten::mul",
            "T5LayerNorm": "aten::mul",
        }

        ##TODO potential bug, need to check only have one bug
        ##TODO, must statisfy af(x)=f(ax),current skip layer may be incomplete
        self.skip_ops_to_find_absorb = ["aten::to",
                                        "aten::relu",
                                        "aten::leaky_relu",
                                        "aten::hardtanh"
                                        ]

        self.could_absorb_layers = ["aten::layer_norm", "aten::batch_norm", "aten::linear", "aten::_convolution",
                                    "aten::group_norm",
                                    "aten::instance_norm",
                                    "aten::mul"]  ##TODO,suppport more norm

    def trace(self, model, dummy_input):
        traced_model = None
        optimize_numerics = False
        if isinstance(dummy_input, dict):
            try:
                traced_model = torch.jit.trace(model, dummy_input["input_ids"], strict=False)
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except:
                pass
        else:
            try:
                traced_model = torch.jit.trace(model, dummy_input, strict=False)
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except:
                try:
                    traced_model = torch.jit.trace(model, dummy_input[0], strict=False)
                    traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
                except:
                    pass
        return traced_model

    def get_nodes(self, traced_model, op_types=['Linear']):
        if isinstance(op_types, str):
            op_types = [op_types]
        nodes = []
        for node in traced_model.graph.nodes():
            node_type = node.kind()
            for op_type in op_types:
                if node_type == op_type:
                    nodes.append((node, op_type))
                    break
        return nodes

    def get_prev_absorb_layer(self, nodes):
        prev_absorb_layer = []
        for node in nodes:
            parent = get_parent(node)
            while 1:
                if parent.kind() in self.skip_ops_to_find_absorb:
                    parent = get_parent(parent)
                    continue
                if parent.kind() in self.could_absorb_layers:
                    prev_absorb_layer.append(parent)
                else:
                    prev_absorb_layer.append(None)
                break
        return prev_absorb_layer

    def mapping_torch_module_to_aten(self, op_types):
        res = []
        for op in op_types:
            if op not in self.supported_torch_module_to_aten.keys():
                logger.warning(f"{op} is not supported in smooth quant, ignoring...")
                continue
            res.append(self.supported_torch_module_to_aten[op])
        res = list(set(res))
        return res

    def _check_valid_conv(self, module):
        """
        remove group conv except depthwise conv
        :param module:
        :return:
        """
        if not isinstance(module, torch.nn.Conv2d):
            return True
        if module.groups > 1:
            if module.in_channels == module.out_channels and \
                    module.groups == module.in_channels:
                return True
            else:
                return False
        return True

    def get_absorb_to_layer(self, model, example_input, op_types):
        traced_model = self.trace(model, example_input)
        if traced_model == None:
            return None, None
        aten_op_types = self.mapping_torch_module_to_aten(op_types)
        nodes_types = self.get_nodes(traced_model, aten_op_types)
        nodes = [node_type[0] for node_type in nodes_types]
        nodes_prev_absorb = self.get_prev_absorb_layer(nodes)
        absorb_to_layer = {}
        no_absorb_layers = []
        for index, absorb in enumerate(nodes_prev_absorb):
            if absorb == None:
                no_absorb_layers.append(nodes[index])
                continue
            node = nodes[index]
            layer_name = '.'.join(node.scopeName().split('/')[-1].split('.')[1:])
            absorb_name = '.'.join(absorb.scopeName().split('/')[-1].split('.')[1:])
            if absorb_name in absorb_to_layer.keys():
                absorb_to_layer[absorb_name].append(layer_name)
            else:
                absorb_to_layer[absorb_name] = [layer_name]
        absorb_to_layer = self.remove_unsupported_layers(model, absorb_to_layer)
        return absorb_to_layer, no_absorb_layers

    def remove_unsupported_layers(self, model, absorb_to_layer):
        res = {}

        for key in absorb_to_layer.keys():

            absorb_layer = get_module(model, key)
            layer_type = absorb_layer.__class__.__name__
            if layer_type not in self.supported_torch_module_to_aten.keys():
                continue
            supported = True
            for layer_name in absorb_to_layer[key]:
                layer = get_module(model, layer_name)
                layer_type = layer.__class__.__name__
                if (layer_type not in self.supported_torch_module_to_aten.keys()) or not self._check_valid_conv(layer):
                    supported = False
                    break
            if supported:
                res[key] = absorb_to_layer[key]
        return res


def update_sq_scale(ipex_config_path, smoothquant_scale_info):
    """update ipex_config.json with smoothquant scale info generated by our algorithm.

    Args:
        ipex_config_path (str): a path to temporary ipex_config.json file.
        smoothquant_scale_info (dict): a dict contains smoothquant scale info.
    """
    with open(ipex_config_path, 'r') as f:
        ipex_config = json.load(f)
        for module_name, v in ipex_config.items():
            if 'q_op_infos' in v and v['q_op_infos']:
                for op_num, v1 in v['q_op_infos'].items():
                    # update alpha data instead of updating weight scale
                    op_name = v1['fqn']  # fqn always exists even it's empty.
                    if op_name in smoothquant_scale_info:
                        # observers were overridden by the fallback step, setting it back.
                        v1['activation_observer'] = {'name': 'SmoothQuantActivationObserver',
                                                     'smooth_quant_enabled': False, 'dtype': 'torch.quint8',
                                                     'qscheme': 'torch.per_tensor_affine', 'reduce_range': False,
                                                     'quant_min': 0, 'quant_max': 255,
                                                     'alpha': smoothquant_scale_info[op_name]['alpha']
                                                     }
                        v1['weight_observer'] = {'name': 'SmoothQuantWeightObserver',
                                                 'smooth_quant_enabled': False, 'dtype': 'torch.qint8',
                                                 'qscheme': 'torch.per_channel_symmetric', 'reduce_range': False,
                                                 'quant_min': -128, 'quant_max': 127,
                                                 'alpha': smoothquant_scale_info[op_name]['alpha']  # only update alpha
                                                 }
        f.close()
    # overwrite ipex_config_path
    with open(ipex_config_path, 'w') as f1:
        json.dump(ipex_config, f1, indent=4)
        f1.close()
