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

try:
    from neural_compressor.utils.utility import LazyImport

    torch = LazyImport('torch')
    from ...utils import logger
except:
    import torch
    import logging

    logger = logging.getLogger()
from collections import UserDict, defaultdict


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
            if iters != -1 and cnt >= iters:
                break
    except Exception as e:
        cnt = 0
        for idx, input in enumerate(dataloader):
            output = forward_wrapper(model, input, device)
            cnt += 1
            if iters != -1 and cnt >= iters:
                break


def model_forward_per_sample(model, sample, device):
    try:
        output = forward_wrapper(model, sample, device)
        return output

    except Exception as e:
        output = forward_wrapper(model, sample[0], device)
        return output


def quant_dequant_w(m, num_bits=8, scheme='asym'):  ##TODO take sym as default
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


def quant_dequant_x(x, num_bits=8):
    eps = torch.finfo(torch.float32).eps
    q_min, q_max = 0, 2. ** num_bits - 1.
    scale = (torch.max(x) - torch.min(x)) / (2 ** num_bits - 1)
    scale = torch.clip(scale, min=eps)
    bias = torch.round(0 - (torch.min(x)) / scale)
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

    def __init__(self, model, dataloader, example_inputs=None, q_func=None, traced_model=None):
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
        self.example_inputs = example_inputs
        self.q_func = q_func
        self.input_values = {}
        self.output_values = {}
        self.input_maxes = {}
        self.input_mins = {}
        self.input_abs_maxes = {}
        self.hook_layer_names = []
        self.hook_values_handles = []
        self.traced_model = traced_model
        if self.traced_model == None:
            self.traced_model = self.model
        self.weight_scale_info = {}
        self.absorb_scales_info = {}
        self.insert_mul = False
        self.allow_absorb = True
        self.record_max_info = False
        self.max_value_info = {} # to record max values for alpha tune
        self.self_absorb_layers = {}
        self.absorb_to_layer = {}

    def _get_device(self):
        """
        Get the model device
        :return:Model device
        """
        for _, p in self.model.named_parameters():
            return p.data.device, p.data.dtype

    def _save_input_pc_hook(self, name):
        """
        A forward hook to save input max of a module
        :param name: the module name
        :return: A hook function
        """

        def save_input_hook(module, inputs, outputs):
            input = inputs[0]
            ##TODO check input channel is correct
            if len(module.weight.shape) == 4:  ##conv3d or conv1d not supported now, need better way
                input = input.permute(0, 2, 3, 1)
            input = input.reshape(-1, input.shape[-1])
            max_tensor = torch.max(input, dim=0)[0]
            min_tensor = torch.min(input, dim=0)[0]
            if name not in self.input_maxes.keys():
                self.input_maxes[name] = max_tensor
                self.input_mins[name] = min_tensor
            else:
                self.input_maxes[name] = torch.max(max_tensor, self.input_maxes[name])
                self.input_mins[name] = torch.min(min_tensor, self.input_mins[name])

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
            self.input_values[name] = [input]  ##TODO save more,like 8
            self.output_values[name] = [outputs]  ##TODO do not save output

        return save_input_output_hook

    def _add_observer(self, modules, input_output_modules=None):
        """
        :param modules: the modules which the observer will insert to
        :return:
        """
        self.hook_handles = []
        for key in modules.keys():
            hook_func = self._save_input_pc_hook(key)
            hook_handle = modules[key].register_forward_hook(hook_func)
            self.hook_handles.append(hook_handle)
        if self.alpha == 'auto' and input_output_modules:
            logger.warning("Auto alpha for Smoothquant records input & output"
                           + ", please avoid out of memory.")
            for key in input_output_modules.keys():
                hook_func = self._save_input_output_hook(key)
                hook_handle = input_output_modules[key].register_forward_hook(hook_func)
                self.hook_values_handles.append(hook_handle)

    def _remove_observer(self):
        """
        remove the observer from the model
        :return:
        """
        for hook_handle in self.hook_handles:
            hook_handle.remove()
        if self.hook_values_handles:
            for hook_handle in self.hook_values_handles:
                hook_handle.remove()

    def _calibrate(self, absorb_to_layer, calib_iter, save_input_output=False):
        """
        :param absorb_to_layer: A dict,key is the absorb layer, val is a list of the to be smoothed layer
        :param calib_iter: Data size for calibration
        :return: A dict that saved the layer name and the channe-wised max value info
        """
        layer_to_absorb = {}
        for key in absorb_to_layer:
            for layer_name in absorb_to_layer[key]:
                layer_to_absorb[layer_name] = key
        hook_module_names_tmp = [absorb_to_layer[key][0] for key in absorb_to_layer.keys()]
        hook_modules = {}

        for index, name in enumerate(hook_module_names_tmp):
            module = get_module(self.model, name)
            if isinstance(module, torch.nn.Linear) or isinstance(module,
                                                                 torch.nn.Conv2d):
                if isinstance(module, torch.nn.Conv2d):
                    if self._check_dw_conv(module):
                        pass
                    elif module.groups > 1:
                        continue

                hook_modules[name] = module
        if len(hook_modules) == 0:
            return {}
        hook_modules_input_output = {}
        for name in self.hook_layer_names:
            hook_modules_input_output[name] = get_module(self.model, name)
        self._add_observer(hook_modules, hook_modules_input_output)
        self._dump_min_max(calib_iter=calib_iter)
        self._remove_observer()
        return self.input_abs_maxes

    def _dump_min_max(self, calibration_method="min_max", calib_iter=100):
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
            abs_max_val = torch.max(torch.abs(self.input_mins[key]), torch.abs(self.input_maxes[key]))
            self.input_abs_maxes[key] = abs_max_val
        for key in self.input_values.keys():
            self.input_values[key] = torch.cat(self.input_values[key], dim=0)  ##this may introduce memory issue
            self.output_values[key] = torch.cat(self.output_values[key], dim=0)

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
            return scale # weigth update is done in SQLinearWrapper initialization
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

        if self.insert_mul:
            from .model_wrapper import SQLinearWrapper
            layer = get_module(self.model, layer_name)
            if isinstance(layer, SQLinearWrapper):
                layer._recover_sq_linear()
                set_module(self.model, layer_name, layer.sq_linear)  ##recover
            else:
                input_minmax = [self.input_mins[layer_name], self.input_maxes[layer_name]]
                new_module = SQLinearWrapper(layer, scale, input_minmax, alpha)
                set_module(self.model, layer_name, new_module)

        elif self.allow_absorb:
            layer = get_module(self.model, layer_name)
            if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.GroupNorm) or \
                    isinstance(layer, torch.nn.InstanceNorm2d):
                if layer.affine:
                    layer.weight *= scale
                    if layer.bias != None:
                        layer.bias *= scale
                else:
                    layer.affine = True
                    weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                    layer.weight = torch.nn.Parameter(
                        weight, requires_grad=False)
                    bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                    layer.bias = torch.nn.Parameter(bias, requires_grad=False
                                                    )
            elif isinstance(layer, torch.nn.LayerNorm) or layer.__class__.__name__ == "LPLayerNorm":
                if layer.elementwise_affine:
                    layer.weight *= scale
                    if layer.bias != None:
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

    def _adjust_parameters(self, absorb_to_layer, input_maxes, alpha=0.5, tuning=False):
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

            weights = torch.cat(weights, dim=0)
            weight_max_per_channel = torch.max(torch.abs(weights), dim=0)[0]

            if self.record_max_info and not tuning:
                # the input of layers with same absorb layer is the same.
                input_minmax = [self.input_mins[layers[0]], self.input_maxes[layers[0]]]
                self.max_value_info[key] = {}
                self.max_value_info[key]['alpha'] = alpha_key
                self.max_value_info[key]['input_minmax'] = input_minmax
                self.max_value_info[key]['weight_max'] = weight_max_per_channel
                self.max_value_info[key]['absorbed_layer'] = layers
                continue

            input_power = torch.pow(input_max, alpha_key)
            weight_power = torch.pow(weight_max_per_channel, 1 - alpha_key)

            scale = torch.clip(input_power / weight_power, min=1e-5)
            scale[input_power == 0] = 1.0

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
                and self.scales_per_op == scales_per_op and self.calib_iter != calib_iter:
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

    def _check_dw_conv(self, module):
        if not isinstance(module, torch.nn.Conv2d):
            return False

        return module.groups > 1 and module.in_channels == module.out_channels and \
               module.groups == module.in_channels

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
        for idx, key in enumerate(self.absorb_to_layer):
            absorb_to_layer_sample, input_max_op = {}, {}
            absorb_key = key
            absorb_to_layer_sample[absorb_key] = self.absorb_to_layer[absorb_key]
            loss_all_layers = {}
            for layer_key in self.absorb_to_layer[absorb_key]:
                # if self._check_dw_conv(get_module(self.model,layer_key)):

                if layer_key not in self.layer_to_absorb.values():
                    if layer_key in input_maxes:
                        self.layer_to_absorb[absorb_key] = layer_key
                layer_key_ = self.layer_to_absorb[absorb_key]
                input_max_op[layer_key] = input_maxes[layer_key_]
                loss_alpha = {}
                for alpha in alpha_space:
                    self.weight_scale_info, self.absorb_scales_info = self._adjust_parameters(
                        absorb_to_layer_sample, input_max_op, alpha, tuning=True
                    )
                    input_of_op, output_of_op = self.input_values[layer_key], self.output_values[layer_key]
                    input_scale = self._reshape_scale_for_input(get_module(self.model, layer_key),
                                                                self.absorb_scales_info[absorb_key])
                    input_of_op_q = quant_dequant_x(input_of_op * input_scale)
                    layer = get_module(self.model, layer_key)

                    if layer.__class__.__name__ == "SQLinearWrapper":
                        layer = layer.sq_linear
                    layer_cp = copy.deepcopy(layer)
                    layer_cp.weight.data = quant_dequant_w(layer_cp)
                    output_of_op_q = layer_cp(input_of_op_q)
                    self.recover()
                    loss = torch.sum(torch.abs(output_of_op - output_of_op_q) ** 2)
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
        logger.debug(ans_layer2absorb)
        return ans_layer2absorb

    def transform(self, alpha=0.5, folding=False, percentile=99.999, op_types=['Linear', 'Conv2d'],
                  scales_per_op=False, calib_iter=100,
                  auto_alpha_args={'alpha_min': 0.3, 'alpha_max': 0.7, 'alpha_step': 0.05, 'attn_method': 'min'}):
        """
        The main entry of smooth quant
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, please refer
        to the paper for more details
        :param folding: whether insert mul(False) or just allow foldable layers(True) for SmoothQuant
        :param percentile: Not supported now
        :param op_types: The op typed to be smooth quantized
        :param scales_per_op: Not supported now
        :param calib_iter: Data size for calibration
        :return: A FP32 model with the same architecture as the orig model but with different weight which will be
        benefit to quantization
        """
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

        self.alpha = alpha
        if not isinstance(self.model, torch.nn.Module):
            logger.warning("smooth quant is ignored since the model is not a torch module")
            return self.model
        self.recover()
        need_calibration = self._check_need_calibration(alpha, percentile, op_types, scales_per_op, calib_iter)
        with torch.no_grad():
            input_maxes = self.input_maxes
            if need_calibration:  ##avoid multiple calibaration during tuning if the only difference is alpha
                if self.insert_mul:
                    self.self_absorb_layers = self._get_all_layer_names()  # TODO: only support linear now.
                    # fetch modules with the same input
                    group_modules = self._trace(op_types, skip_unsupported_layers=False)
                    for k, v in group_modules.items():
                        # use one input for qkv
                        for i in v:
                            if i in self.self_absorb_layers:
                                self.self_absorb_layers.pop(i)
                        self.self_absorb_layers[v[0]] = v
                    logger.debug(f"self_absorb_layers:{self.self_absorb_layers}")
                if self.allow_absorb:
                    self.absorb_to_layer, no_absorb_layers = self._trace(
                        op_types)  ##TODO we need to insert mul layer for no_absorb_layers later
                    if self.absorb_to_layer == None and no_absorb_layers == None:
                        return self.model

                # remove self.self_absorb_layers if it exists in self.absorb_to_layer
                for k, v in self.absorb_to_layer.items():
                    for i in v:
                        if i in self.self_absorb_layers:
                            self.self_absorb_layers.pop(i)
                self.absorb_to_layer.update(self.self_absorb_layers)
                for key in self.absorb_to_layer:
                    self.hook_layer_names += self.absorb_to_layer[key]

                if self.absorb_to_layer == None and no_absorb_layers == None:
                    logger.warning("sorry, could not trace the model, smooth quant is ignored")
                    logger.warning("if you are using huggingface model,"
                                   "you could set torchscript to True ")
                    return self.model
                save_input_output = False
                if alpha == "auto":
                    save_input_output = True

                input_maxes = self._calibrate(self.absorb_to_layer, calib_iter, save_input_output)

                # Check if input_maxes match self.absorb_to_layer 
                # (due to self._get_all_layer_names use layer tree instead of forward_path)
                if not folding:
                    diff_modules = set(self.absorb_to_layer.keys()).difference(input_maxes.keys())
                    for d in diff_modules:
                        del self.absorb_to_layer[d]
                        
                if alpha == 'auto':
                    self.alpha_per_layer = self._auto_tune_alpha(input_maxes, **auto_alpha_args)  ##save the alpha

            if alpha == 'auto':
                alpha = self.alpha_per_layer
            example_inputs = self._get_example_input()
            if example_inputs != None:
                out_pre_sq = model_forward_per_sample(self.model, example_inputs, self.device)

            if self.record_max_info:
                # max_info is recorded in self.max_value_info
                self._adjust_parameters(self.absorb_to_layer, input_maxes, alpha)
                self.model._smoothquant_optimized = False
                return self.model

            self.weight_scale_info, self.absorb_scales_info = self._adjust_parameters(self.absorb_to_layer,
                                                                                      input_maxes, alpha)

            self.model._smoothquant_optimized = True
            if example_inputs != None:
                # Check mathematical equivelancy
                out_post_sq = model_forward_per_sample(self.model, example_inputs, self.device)

                if not self.output_is_equal(out_post_sq, out_pre_sq):
                    logger.warning(
                        "Mathematical equivelancy of Smoothquant is not preserved. "
                        "Please kindly report this issue to https://github.com/intel/neural-compressor.")
                    # self.recover()
                    # self.model._smoothquant_optimized = False
            else:
                logger.warning(" Could not get example input, equivelancy check is skipped")

            self.input_values, self.output_values = {}, {}
            return self.model

    def output_is_equal(self, out1, out2, atol=1e-04):
        try:
            if isinstance(out1, tuple):
                return all(torch.all(torch.isclose(out1[i], out2[i], atol=atol)) for i in range(len(out1)))
            elif isinstance(out1, dict):
                return all(torch.all(torch.isclose(out1[k], out2[k], atol=atol)) for k in out1.keys())
            elif isinstance(out1, torch.Tensor):
                return torch.all(torch.isclose(out1, out2, atol=atol))
            return False
        except:
            logger.warning("Automatically check failed, Please check equivelancy manually "
                           "between out_pre_sq and out_post_sq if necessary.")
            return True

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
        # remove duplicate Linear if Linear is wrapped by Linear
        key_list = list(self_absorb_layer.keys())
        key_list.sort()
        duplicate_list = []
        for i, k1 in enumerate(key_list):
            for k2 in key_list[i+1:]:
                if k1 in k2:
                    duplicate_list.append(k1)
        for i in duplicate_list:
            self_absorb_layer.pop(i)
        return self_absorb_layer

    def _get_example_input(self):
        if self.dataloader == None and self.example_inputs == None:
            return None
        if self.example_inputs is None:
            ##assert self.dataloader, "Please provide dataloader or example_inputs"
            for idx, input in enumerate(self.dataloader):
                self.example_inputs = input
                break

        return self.example_inputs

    def _trace(self, op_types, skip_unsupported_layers=True):
        """
        Try the model to find the layers which can be smooth quantized.
        :param op_types: The op types to be smooth quantized
        :return:
        absorb_to_layer: A dict, absorb layer name:layers to be smooth quantized
        no_absorb_layers: A list saving the layers which could not find the absorb layer
        """
        tg = GraphTrace()
        self._get_example_input()
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(
            self.traced_model, self.example_inputs, op_types, 
            skip_unsupported_layers=skip_unsupported_layers
        )
        if not skip_unsupported_layers:
            return absorb_to_layer
        if absorb_to_layer == None and no_absorb_layers == None:
            logger.warning("sorry, could not trace the model, smooth quant is skipped")
            logger.warning("if you are using huggingface model,"
                            "you could set torchscript to True "
                            "when loading the model or set the return_dict to False")
        elif absorb_to_layer == {}:
            logger.warning("could not find any layer to be absorbed")
        else:
            to_absorb_cnt = 0
            for key, item in absorb_to_layer.items():
                to_absorb_cnt += len(item)
            logger.info(
                f" {to_absorb_cnt} out of {to_absorb_cnt + len(no_absorb_layers)} "
                f"layers could be absorbed in smooth quant")
        return absorb_to_layer, no_absorb_layers


def get_parent(node, all_parents=False):
    if node.inputs() == None:
        return None
    elif len(list(node.inputs())) == 0:
        return None
    if not all_parents:
        return list(node.inputs())[0].node()
    else:
        return list(node.inputs())


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
            "LPLayerNorm": "aten::layer_norm"  ##mpt_chat
        }
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
        if isinstance(dummy_input, dict) or isinstance(dummy_input, UserDict):
            try:
                traced_model = torch.jit.trace(model, example_kwarg_inputs=dict(dummy_input), strict=False)
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except Exception as e:
                logger.warning(e)
                logger.warning("Jit trace in GraphTrace failed, absorb layer detection is skipped")
        else:
            try:
                traced_model = torch.jit.trace(model, dummy_input, strict=False)
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except:
                try:
                    traced_model = torch.jit.trace(model, dummy_input[0], strict=False)
                    traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
                except Exception as e:
                    logger.warning(e)
                    logger.warning("Jit trace in GraphTrace failed, absorb layer detection is skipped")
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

                    parent_out_kinds = []
                    for val_user in list(parent.outputs())[0].uses():
                        next_node = val_user.user
                        parent_out_kinds.append(next_node.kind())
                    parent_out_kinds = set(parent_out_kinds)
                    parent_out_kinds.discard('aten::size')

                    if parent_out_kinds == parent_out_kinds.intersection(self.could_absorb_layers):
                        prev_absorb_layer.append(parent)
                    elif parent_out_kinds.intersection(self.skip_ops_to_find_absorb):
                        res = self.skip_op_absorb_helper(parent)
                        prev_absorb_layer.append(parent) if res else prev_absorb_layer.append(None)
                    else: # When parent to multiple ops, sq transformation could be wrong.
                        prev_absorb_layer.append(None)
                else:
                    prev_absorb_layer.append(None)
                break
        return prev_absorb_layer


    def skip_op_absorb_helper(self, parent_node):
        for val_user in list(parent_node.outputs())[0].uses():
            next_node = val_user.user
            if next_node.kind() == 'aten::size':
                continue
            elif next_node.kind() in self.could_absorb_layers:
                continue
            elif next_node.kind() in self.skip_ops_to_find_absorb:
                node_res = self.skip_op_absorb_helper(next_node)
                if not node_res:
                    return False
            else:
                return False
        return True

    def mapping_torch_module_to_aten(self, op_types):
        res = []
        for op in op_types:
            if op not in self.supported_torch_module_to_aten.keys():
                logger.warning(f"{op} is not supported in smooth quant, ignoring...")
                continue
            res.append(self.supported_torch_module_to_aten[op])
        res = list(set(res))
        return res

    def get_absorb_to_layer(self, model, example_input, op_types, skip_unsupported_layers=True):
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
                no_absorb_layers.append(
                    '.'.join(nodes[index].scopeName().split('/')[-1].split('.')[1:]))
                continue
            node = nodes[index]
            layer_name = '.'.join(node.scopeName().split('/')[-1].split('.')[1:])
            absorb_name = '.'.join(absorb.scopeName().split('/')[-1].split('.')[1:])
            if layer_name == "" or absorb_name == "":
                continue
            if absorb_name in absorb_to_layer.keys():
                absorb_to_layer[absorb_name].append(layer_name)
            else:
                absorb_to_layer[absorb_name] = [layer_name]
        if skip_unsupported_layers:
            absorb_to_layer = self.remove_unsupported_layers(model, absorb_to_layer, no_absorb_layers)
        return absorb_to_layer, no_absorb_layers

    def remove_unsupported_layers(self, model, absorb_to_layer, no_absorb_layers):
        res = {}

        for key in absorb_to_layer.keys():
            absorb_layer = get_module(model, key)
            layer_type = absorb_layer.__class__.__name__
            if layer_type not in self.supported_torch_module_to_aten.keys():
                no_absorb_layers.extend(absorb_to_layer[key])
                continue
            supported = True
            for layer_name in absorb_to_layer[key]:
                layer = get_module(model, layer_name)
                layer_type = layer.__class__.__name__
                if layer_type not in self.supported_torch_module_to_aten.keys():
                    supported = False
                    no_absorb_layers.extend(absorb_to_layer[key])
                    break
            if supported:
                res[key] = absorb_to_layer[key]
        return res
