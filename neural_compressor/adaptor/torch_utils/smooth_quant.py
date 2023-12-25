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

import copy
import json

try:
    from neural_compressor.utils.utility import LazyImport

    torch = LazyImport("torch")
    from ...utils import logger
except:
    import logging

    import torch

    logger = logging.getLogger()
from collections import UserDict, defaultdict

import numpy
from tqdm import tqdm


def enough_memo_store_scale(device, need_space):
    if device == "cuda":  # pragma: no cover
        current_gpu_index = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory
        used_memory = torch.cuda.memory_allocated(current_gpu_index)
        free_space = total_memory - used_memory
    else:
        import psutil

        free_space = psutil.virtual_memory().free
    return free_space >= need_space


def move_input_to_device(input, device=torch.device("cpu")):
    if isinstance(input, dict) or isinstance(input, UserDict):
        tmp_input = {}
        for k, inp in input.items():
            tmp_input[k] = move_input_to_device(inp, device)
        input = tmp_input
    elif isinstance(input, list) or isinstance(input, tuple):
        is_tuple = isinstance(input, tuple)
        tmp_input = []
        for inp in input:
            tmp_input.append(move_input_to_device(inp, device))
        input = tuple(tmp_input) if is_tuple else tmp_input
    elif isinstance(input, torch.Tensor):
        input = input.to(device)  # pylint: disable=no-member
    return input


##TODO potential bug, data typeR
def forward_wrapper(model, input, device=torch.device("cpu")):
    try:
        model = model.to(device)
        input = move_input_to_device(input, device)
    except Exception as e:
        logger.warning(e)
        logger.warning("Please check the input device if the error raised.")
    if isinstance(input, dict) or isinstance(input, UserDict):
        output = model(**input)
    elif isinstance(input, list) or isinstance(input, tuple):
        try:
            output = model(*input)
        except:
            output = model(input)
    else:
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


def quant_dequant_w(m, num_bits=8, scheme="sym"):
    eps = torch.finfo(torch.float32).eps
    if isinstance(m, torch.nn.Linear):
        x = m.weight
        tmp = torch.zeros(torch.max(x, dim=1).values.size())
        if scheme == "sym":
            q_min, q_max = -(2.0 ** (num_bits - 1)), 2.0 ** (num_bits - 1) - 1.0
            x_max = torch.max(torch.abs(x), dim=1).values
            scale = x_max / (float(q_max - q_min) / 2)
        else:
            q_min, q_max = 0, 2.0**num_bits - 1.0
            x_max = torch.maximum(torch.max(x, dim=1).values, tmp)
            x_min = torch.minimum(torch.min(x, dim=1).values, tmp)
            scale = (x_max - x_min) / (2**num_bits - 1)

        scale = torch.clip(scale, min=eps)

        if scheme == "sym":
            bias = 0
        else:
            bias = torch.round(0 - (torch.min(x, dim=1).values) / scale)
            bias = bias.unsqueeze(dim=-1)
        scale = scale.unsqueeze(dim=-1)
        q_x = torch.round(x / scale + bias)
        q_x.clamp_(q_min, q_max)
        return (q_x - bias) * scale
    elif isinstance(m, torch.nn.Conv2d):
        x = m.weight
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(-1, x.shape[-1])
        tmp = torch.zeros(torch.max(x, dim=0).values.size())
        if scheme == "sym":
            q_min, q_max = -(2.0 ** (num_bits - 1)), 2.0 ** (num_bits - 1) - 1.0
            x_max = torch.max(torch.abs(x), dim=0).values
            scale = x_max / (2 ** (num_bits - 1) - 1)
        else:
            q_min, q_max = 0, 2.0**num_bits - 1.0
            x_max = torch.maximum(torch.max(x, dim=0).values, tmp)
            x_min = torch.minimum(torch.min(x, dim=0).values, tmp)
            scale = (x_max - x_min) / (2**num_bits - 1)
        scale = torch.clip(scale, min=eps)
        if scheme == "sym":
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
    q_min, q_max = 0, 2.0**num_bits - 1.0
    if max_x is None or min_x is None:
        max_x, min_x = torch.max(x), torch.min(x)
    else:
        max_x = torch.max(max_x)
        min_x = torch.min(min_x)
    scale = (max_x - min_x) / (2**num_bits - 1)
    scale = torch.clip(scale, min=eps)
    bias = torch.round((0 - min_x) / scale)
    q_x = torch.round(x / scale + bias)
    q_x.clamp_(q_min, q_max)
    return scale * (q_x - bias)


def get_module(model, key):
    """Get module from model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    module = model
    name_list = key.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
        elif hasattr(module, "sq_linear"):  # for peft models
            module = getattr(module, "sq_linear")
            module = getattr(module, name)
        elif hasattr(module, "orig_layer"):  # for peft models and auto alpha
            module = getattr(module, "orig_layer")
            module = getattr(module, name)
        else:
            module = module
    return module


def set_module(model, key, new_module):
    """Set new module into model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        elif hasattr(module, ("sq_linear")):  # for peft models that Linears are contained in Linear
            module = getattr(module, "sq_linear")
            module = getattr(module, name)
        elif hasattr(module, ("orig_layer")):  # for peft models and auto alpha
            module = getattr(module, "orig_layer")
            module = getattr(module, name)
        else:
            module = module

    if hasattr(module, "sq_linear") and name_list[-1] != "sq_linear":  # for peft models
        module = getattr(module, "sq_linear")
    if hasattr(module, "orig_layer") and name_list[-1] != "orig_layer":  # for peft models and auto alpha
        module = getattr(module, "orig_layer")
    setattr(module, name_list[-1], new_module)


def cal_scale(input_max, weights, alpha, scale_type="orig"):
    if scale_type == "orig":  # same as the paper
        weights = torch.cat(weights, dim=0)
        weight_max = torch.max(torch.abs(weights), dim=0)[0]
        input_power = torch.pow(input_max, alpha)
        logger.debug(f"{max(input_max)}, {min(input_max)}")
        weight_power = torch.pow(weight_max, 1 - alpha)
        scale = torch.clip(input_power / weight_power, min=1e-5)
        scale[input_power == 0] = 1.0
        if input_power.size() == weight_power.size():
            scale[weight_power == 0] = 0.0  ##FIXME
        return scale


class WrapperLayer(torch.nn.Module):
    def __init__(self, layer, input_min, input_max, save_q_input=False):
        super(WrapperLayer, self).__init__()
        self.add_module("orig_layer", layer)  # set orig_layer in get/set_module
        self.quant = False
        self.q_input = None
        self.fp32_output = None
        self.input_max = input_max
        self.input_min = input_min
        self.weight_scale = None
        self.input_scale = None
        self.save_q_input = save_q_input
        self.do_blockwise = False

    def enable_quant(self):
        self.quant = True

    def disable_quant(self):
        self.quant = False

    def update_scale(self, input_scale, weight_scale):
        self.input_scale = input_scale
        self.weight_scale = weight_scale

    ##TODO better tradeoff performance and memory, currently it's too slow
    def q_dq_forward(self, x, input_scale, weight_scale):
        layer_copy = copy.deepcopy(self.orig_layer)
        if weight_scale is not None:
            layer_copy.weight *= weight_scale
        q_dq_weight = quant_dequant_w(layer_copy)
        layer_copy.weight.data.copy_(q_dq_weight)
        if input_scale is None:
            x = quant_dequant_x(x, self.input_min, self.input_max)
        else:
            x = input_scale * x
            x = quant_dequant_x(x, self.input_min * input_scale, self.input_max * input_scale)  ##FIXME
        output = layer_copy(x)
        return output

    def q_dq_forward_blockwise(self, x, input_scale):
        layer_copy = copy.deepcopy(self.orig_layer)
        if input_scale is None:
            x = quant_dequant_x(x, self.input_min, self.input_max)
        else:
            x = input_scale * x
            x = quant_dequant_x(x, self.input_min * input_scale, self.input_max * input_scale)  ##FIXME
        output = layer_copy(x)
        return output

    def forward(self, x):
        if self.quant:
            # self.q_input = x * scale ##save the q_input
            if self.save_q_input:
                self.q_input = x
            if not self.do_blockwise:
                output = self.q_dq_forward(x, self.input_scale, self.weight_scale)
            else:
                output = self.q_dq_forward_blockwise(x, self.input_scale)

        else:
            output = self.orig_layer(x)
        self.output = output
        return output


class TorchSmoothQuant:
    """Fake input channel quantization, for more details please refer to
    [1] SmoothQuant: Accurate and Efficient
    Post-Training Quantization for Large Language Models
    [2] SPIQ: Data-Free Per-Channel Static Input Quantization
    Currently, we only handle the layers whose smooth scale could be absorbed, we will support other layers later.

    We only support inplace mode which means the model weights will be changed, you can call recover function
    to recover the weights if needed
    """

    def __init__(self, model, dataloader=None, example_inputs=None, q_func=None, traced_model=None):
        """
        :param model: Torch model :param dataloader: Calibration dataloader :param traced_model: A specific model
        shares the same architecture as the model and could be traced by torch.jit. If not supplied, we use model
        instead.
        """
        self.model = model
        if not isinstance(self.model, torch.nn.Module):
            return
        device, dtype = self._get_device()
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.dataloader = dataloader
        self.example_inputs = example_inputs
        self.q_func = q_func
        self.input_maxes = {}
        self.input_mins = {}
        self.input_maxes_abs = {}
        self.traced_model = traced_model
        if self.traced_model is None:
            self.traced_model = self.model
        self.weight_scale_info = {}
        self.absorb_scales_info = {}
        self.insert_mul = False
        self.allow_absorb = True
        self.record_max_info = False
        self.max_value_info = {}  # to record max values for alpha tune
        self.self_absorb_layers = {}
        self.absorb_to_layer = {}
        self.adjust_alpha_space = False
        self.weight_clip = True
        self.default_alpha = 0.5

        self._save_scale = False
        self.weight_scale_dict = {}

        self.do_blockwise = False
        self.block_inputs = {}
        self.block_outputs = {}

    def _get_device(self):
        """Get the model device
        :return:Model device."""
        for _, p in self.model.named_parameters():
            return p.data.device, p.data.dtype

    def _save_input_pc_hook(self, name, percentile=100):
        """A forward hook to save input max of a module
        :param name: the module name
        :return: A hook function."""

        def save_input_hook(module, inputs, outputs):
            input = inputs[0]
            ##TODO check input channel is correct
            if len(module.weight.shape) == 4:  ##conv3d or conv1d not supported now, need better way
                input = input.permute(0, 2, 3, 1)
            input = input.reshape(-1, input.shape[-1])
            max_tensor = torch.max(input, dim=0)[0]
            min_tensor = torch.min(input, dim=0)[0]
            k_index = int(input.shape[0] * percentile / 100)
            res, _ = torch.kthvalue(torch.abs(input), k_index, dim=0)
            ##res = torch.max(torch.abs(input),dim=0)[0]
            if name not in self.input_maxes.keys():
                self.input_mins[name], self.input_maxes[name] = min_tensor, max_tensor
                self.input_maxes_abs[name] = res
            else:
                self.input_mins[name] = torch.min(self.input_mins[name], min_tensor)
                self.input_maxes[name] = torch.max(self.input_maxes[name], max_tensor)
                self.input_maxes_abs[name] = torch.max(self.input_maxes_abs[name], res)

        return save_input_hook

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

    def _remove_observer(self):
        """Remove the observer from the model
        :return:"""
        for hook_handle in self.hook_handles:
            hook_handle.remove()

    def _calibrate(self, absorb_to_layer, calib_iter, percentile):
        """
        :param absorb_to_layer: A dict,key is the absorb layer, val is a list of the to be smoothed layer
        :param calib_iter: Data size for calibration
        :return: A dict that saved the layer name and the channel-wise max value info
        """
        ##hook all the module
        hook_modules = {}
        for n, module in self.model.named_modules():
            if isinstance(module, tuple(self.op_types)):
                hook_modules[n] = module

        self._add_min_max_observer(hook_modules, percentile)

        self._dump_min_max(calib_iter=calib_iter)
        self._remove_observer()
        return self.input_maxes_abs

    def _dump_min_max(self, calib_iter=100):
        """Dump min max per channel information, the min max value will be saved in input_maxes attribute
        :param calibration_method: only support min_max currently
        :param calib_iter: Sample size for calibration
        :return:"""
        logger.info("Calibrating...")
        if self.q_func:
            self.q_func(self.model)
        else:
            assert self.dataloader, "Please set dataloader for calibration."
            model_forward(self.model, self.dataloader, calib_iter, self.device)

    def _reshape_in_channel_to_last(self, layer_name):
        """Move the input channel to the last dim
        :param layer_name: Layer name
        :return: The reshaped weight."""
        layer = get_module(self.model, layer_name)
        if layer.__class__.__name__ == "WrapperLayer":
            layer = layer.orig_layer

        weight = layer.weight  ##TODO oc*ic, support transposed conv
        if len(weight.shape) == 4:
            weight = weight.permute(0, 2, 3, 1)
            weight = weight.reshape(-1, weight.shape[-1])
        return weight

    def _reshape_scale_for_weight(self, layer, scale):
        """Reshape the scale for weight input channel, depthwise output channel
        :param layer:  torch module
        :param scale: orig scale
        :return: reshaped scale."""
        if hasattr(layer, "orig_layer"):
            layer = layer.orig_layer
        if isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:  ##only depthwise conv could hit here
            scale = scale.view(scale.shape[0], 1, 1, 1)  ##mount on output channel

        elif isinstance(layer, torch.nn.Conv2d):
            scale = scale.view(1, scale.shape[0], 1, 1)

        elif isinstance(layer, torch.nn.Linear):
            scale = scale.view(1, scale.shape[0])

        return scale

    def get_blocks(self):
        block_names = []
        for n, m in self.model.named_modules():
            if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
                for nn, mm in m.named_children():
                    block_name = n + "." + nn
                    block_names.append(block_name)
        return block_names

    def _reshape_scale_for_input(self, layer, scale):
        """Reshape the scale for input feature in channel
        :param layer:

        :param scale:
        :return:
        """
        if hasattr(layer, "orig_layer"):
            layer = layer.orig_layer
        if isinstance(layer, torch.nn.Conv2d):
            scale = scale.view(1, scale.shape[0], 1, 1)

        elif isinstance(layer, torch.nn.Linear):
            scale = scale.view(1, scale.shape[0])

        return scale

    def _scale_layer_weight(self, layer_name, scale, alpha=0.5, input_minmax=None):  ##input channel
        """Scale the layer weights at input channel, depthwise conv output channel
        :param layer_name: The layer name
        :param scale: The scale to be multiplied
        :param alpha: alpha for SQLinearWrapper
        :param input_minmax: input_minmax for SQLinearWrapper
        :return:"""
        layer = get_module(self.model, layer_name)
        if self.insert_mul:
            from .model_wrapper import SQLinearWrapper

            layer = get_module(self.model, layer_name)
            if isinstance(layer, SQLinearWrapper):
                layer._recover_sq_linear()
                set_module(self.model, layer_name, layer.sq_linear)  ##recover
            else:
                new_module = SQLinearWrapper(layer, 1.0 / scale, input_minmax, alpha)
                set_module(self.model, layer_name, new_module)
        elif self.allow_absorb:
            scale = self._reshape_scale_for_weight(layer, scale)
            layer.weight = torch.nn.Parameter(layer.weight * scale)
        return scale

    def _absorb_scales(self, layer_name, scale):  ##output channel
        """Absorb the scale to the layer at output channel
        :param layer_name: The module name
        :param scale: The scale to be absorbed
        :param alpha_key: The alpha passed to SQLinearWrapper
        :return:"""
        if self.insert_mul or not self.allow_absorb:
            return  # absorb is updated in SQLinearWrapper in def _scale_layer_weight

        ##if self.allow absorb
        layer = get_module(self.model, layer_name)
        if layer.__class__.__name__ == "WrapperLayer":
            layer = layer.orig_layer
        if (
            isinstance(layer, torch.nn.BatchNorm2d)
            or isinstance(layer, torch.nn.GroupNorm)
            or isinstance(layer, torch.nn.InstanceNorm2d)
        ):
            if layer.affine:
                layer.weight *= scale
                layer.bias *= scale
            else:
                layer.affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        elif isinstance(layer, torch.nn.LayerNorm):
            if layer.elementwise_affine:
                layer.weight *= scale
                layer.bias *= scale
            else:
                layer.elementwise_affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(torch.ones(weight, requires_grad=False))
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        elif isinstance(layer, torch.nn.Conv2d):
            ##the order could not be changed
            if hasattr(layer, "bias") and (layer.bias is not None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1, 1, 1)
            layer.weight *= scale

        elif isinstance(layer, torch.nn.Linear):
            if hasattr(layer, "bias") and (layer.bias is not None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1)
            layer.weight *= scale

        elif layer.__class__.__name__ == "LlamaRMSNorm" or layer.__class__.__name__ == "T5LayerNorm":  ##quite tricky
            layer.weight *= scale

        else:
            logger.warning(
                f"found unsupported layer {type(layer)}, try to multiply scale to "
                f"weight and bias directly, this may introduce accuracy issue, please have a check "
            )
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight *= scale
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias *= scale

    def _cal_scales(self, absorb_to_layer, input_maxes, alpha=0.5, tuning=False):
        """Cal the adjust scales
        :param absorb_to_layer: A dict mapping absorb layer to smooth quantized layer
        :param input_maxes: The channel-wise input max info for layers
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, a float of a dict
        :return:"""
        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        weight_scales_info = {}
        absorb_scales_info = {}
        for index, key in enumerate(absorb_to_layer.keys()):
            alpha_tmp = alpha[key] if isinstance(alpha, dict) else alpha
            if alpha_tmp < 0:
                scale = torch.ones((1), device=self.device)
            else:
                input_max = absorb_to_input_maxes[key]
                layer_names = absorb_to_layer[key]
                weights = []
                for layer_name in layer_names:
                    weight = self._reshape_in_channel_to_last(layer_name)
                    weights.append(weight)

                weight_max_per_channel = torch.max(torch.abs(torch.cat(weights, dim=0)), dim=0)[0]
                if self.weight_clip:
                    weight_max_per_channel = weight_max_per_channel.clamp(min=1e-5)
                if self.record_max_info and not tuning:
                    # the input of layers with same absorb layer is the same.
                    input_minmax = [self.input_mins[layer_names[0]], self.input_maxes[layer_names[0]]]
                    self.max_value_info[key] = {}
                    self.max_value_info[key]["alpha"] = alpha_tmp
                    self.max_value_info[key]["input_minmax"] = input_minmax
                    self.max_value_info[key]["weight_max"] = weight_max_per_channel
                    self.max_value_info[key]["absorbed_layer"] = layer_names
                    continue

                if self._save_scale:
                    if key in self.weight_scale_dict and alpha_tmp in self.weight_scale_dict[key]:
                        scale = self.weight_scale_dict[key][alpha_tmp]
                    else:
                        scale = cal_scale(input_max, weights, alpha_tmp)
                else:
                    scale = cal_scale(input_max, weights, alpha_tmp)

            absorb_scales_info[key] = 1.0 / scale
            absorb_scales_info[key][scale == 0] = 0
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                ##self._scale_layer_weight(layer_name, scale)
                weight_scales_info[layer_name] = scale
                if self._save_scale:
                    if layer_name not in self.weight_scale_dict:
                        self.weight_scale_dict[layer_name] = {}
                    self.weight_scale_dict[layer_name][alpha_tmp] = scale
        return absorb_scales_info, weight_scales_info

    def _adjust_parameters(self, absorb_to_layer, input_maxes, alpha=0.5, tuning=False):
        """Adjust the weights and biases
        :param absorb_to_layer: A dict mapping absorb layer to smooth quantized layer
        :param input_maxes: The channel-wise input max info for layers
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, a float of a dict
        :return:"""
        absorb_scales_info, weight_scales_info = self._cal_scales(absorb_to_layer, input_maxes, alpha, tuning)
        if not absorb_scales_info or not weight_scales_info:
            return weight_scales_info, absorb_scales_info
        for index, key in enumerate(absorb_to_layer.keys()):
            if isinstance(alpha, float):
                alpha_tmp = alpha
            elif isinstance(alpha, dict):
                alpha_tmp = alpha[key]
            absorb_scale = absorb_scales_info[key]
            self._absorb_scales(key, absorb_scale)
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                input_minmax = [self.input_mins[layer_names[0]], self.input_maxes[layer_names[0]]]
                self._scale_layer_weight(layer_name, weight_scales_info[layer_name], alpha_tmp, input_minmax)
        return weight_scales_info, absorb_scales_info

    def _check_need_calibration(self, alpha, percentile, op_types, scales_per_op, calib_iter):
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

        if (
            self.percentile == percentile
            and self.op_types == op_types
            and self.scales_per_op == scales_per_op
            and self.calib_iter == calib_iter
        ):
            if isinstance(alpha, float) or self.alpha == "auto":
                need_calib = False

        self.alpha, self.percentile = alpha, percentile
        self.op_types, self.scales_per_op = op_types, scales_per_op
        self.calib_iter = calib_iter
        return need_calib

    def _get_auto_loss(self, output, output_q, loss_type="abs", loss_alpha=1.0):
        """Get the loss for auto tuning
        :param output: Fp32 output for one layer
        :param output_q: Quant output for one layer
        :param loss_type: The type of loss
        :param loss_alpha: Loss alpha i for mean scale error
        :return: A tensor of the loss."""
        if len(output.shape) <= 2:
            max_value = torch.max(torch.abs(output))
        else:
            output = output.reshape(output.shape[0], -1)
            output_q = output_q.reshape(output_q.shape[0], -1)
            max_value = torch.max(torch.abs(output), dim=-1).values.unsqueeze(-1)
            max_value = torch.clip(max_value, 1e-5)
        output = output / max_value  ##FIXME need copy not replace
        output_q = output_q / max_value
        # if loss_type == "nsr":  # nsr is unused at this point.
        #     output[output == 0] = 1e-5
        #     loss = torch.sum(torch.log(1.0 + torch.abs(output - output_q) / torch.abs(output)))
        #     return loss
        if loss_type == "abs":
            return torch.sum(torch.pow(torch.abs(output - output_q), 0.5))
        else:
            return torch.sum((output - output_q) ** 2)

    def _get_sq_layer_names(self):
        """Get the all the hook sq layer
        :return: All the sq layer names."""
        ##TODO this may not fit for folding=False
        module_names = []
        for key in self.absorb_to_layer:
            module_names += self.absorb_to_layer[key]
        return module_names

    def _get_all_hook_module_names(self):
        module_names = []
        for n, module in self.model.named_modules():
            if isinstance(module, tuple(self.op_types)):
                module_names.append(n)
        return module_names

    def _qdq_model_wrapper_for_auto(self, save_q_input=False):
        """Wrapper all the module with qdq
        :return:"""
        module_names = self._get_all_hook_module_names()
        self.to_unwrap_module_names = module_names
        for name in module_names:
            if name not in self.input_mins:  # skip module if it's not used in calibration
                continue
            module = get_module(self.model, name)
            new_module = WrapperLayer(module, self.input_mins[name], self.input_maxes[name], save_q_input=save_q_input)
            set_module(self.model, name, new_module)

    def _qdq_model_unwrapper_for_auto(self):
        module_names = self.to_unwrap_module_names
        for name in module_names:
            module = get_module(self.model, name)
            if not hasattr(module, "orig_layer"):  # skip module if it's not used in calibration
                continue
            set_module(self.model, name, module.orig_layer)

    def _change_qdq_for_auto(self, enable=True):
        module_names = self._get_all_hook_module_names()
        for name in module_names:
            name = name.split(".orig_layer")[0]
            module = get_module(self.model, name)
            if not hasattr(module, "orig_layer"):  # skip module if it's not used in calibration
                continue
            if enable:
                module.enable_quant()
            else:
                module.disable_quant()

    def _update_scales_for_auto(self, absorb_scales, weight_scales):
        for key in self.absorb_to_layer.keys():
            layer_names = self.absorb_to_layer[key]
            for layer_name in layer_names:
                layer = get_module(self.model, layer_name)
                input_scale = absorb_scales[key]
                weight_scale = weight_scales[layer_name]
                input_scale = self._reshape_scale_for_input(layer, input_scale)
                weight_scale = self._reshape_scale_for_weight(layer, weight_scale)
                layer.update_scale(input_scale, weight_scale)  ##FIXME

    def _add_blockwise_observer(self, block_modules):
        """
        :param block_modules: the block modules which the observer will insert to
        :return:
        """
        self.blockwise_hook_handles = []
        for key in block_modules.keys():
            hook_func = self._save_blockwise_hook(key)
            hook_handle = block_modules[key].register_forward_hook(hook_func)
            self.blockwise_hook_handles.append(hook_handle)

    def _save_blockwise_hook(self, name):
        """A forward hook to save inputs/outputs of a block
        :param name: the block name
        :return: A hook function."""

        def save_blockwise_hook(module, inputs, outputs):
            self.block_inputs[name] = inputs[0]
            self.block_outputs[name] = outputs[0]

        return save_blockwise_hook

    def _get_one_batch_auto_loss(self, input, alpha_space, orig_best_alpha, input_maxes):
        self._change_qdq_for_auto(enable=False)
        module_names = self._get_sq_layer_names()

        if self.do_blockwise:
            block_modules = {}
            for key in self.block_names:
                block_modules[key] = get_module(self.model, key)
            self._add_blockwise_observer(block_modules)

        forward_wrapper(self.model, input, self.device)  ##disable quant and get fp32 output

        fp32_output = {}
        if not self.do_blockwise:
            for name in module_names:
                module = get_module(self.model, name)
                fp32_output[name] = module.output
                module.output = None
        else:
            for block_name in self.block_names:
                fp32_output[block_name] = self.block_outputs[block_name]
        self._change_qdq_for_auto(enable=True)
        absorb_input_scales, weight_scales = self._cal_scales(
            self.absorb_to_layer, input_maxes, orig_best_alpha, tuning=True
        )
        self._update_scales_for_auto(absorb_input_scales, weight_scales)
        forward_wrapper(self.model, input, self.device)  ##save quant_input
        for mod_name in module_names:  # save fp32 values
            mod = get_module(self.model, mod_name)
            if mod_name in self.fp32_output_val:
                self.fp32_output_val[mod_name].append(torch.norm(mod.output))
            else:
                self.fp32_output_val[mod_name] = [torch.norm(mod.output)]
            del mod

        loss_alphas = {}
        if not self.do_blockwise:
            for name in module_names:
                module = get_module(self.model, name)
                loss = self._get_auto_loss(fp32_output[name], module.output)
                cur_alpha = orig_best_alpha
                if isinstance(orig_best_alpha, dict):
                    cur_alpha = orig_best_alpha[name]
                key_name = str(cur_alpha)
                loss_alphas[name] = {key_name: loss}
        else:
            for block_name in self.block_names:
                block = get_module(self.model, block_name)
                loss = self._get_auto_loss(fp32_output[block_name], self.block_outputs[block_name])
                cur_alpha = orig_best_alpha
                if isinstance(orig_best_alpha, dict):
                    cur_alpha = orig_best_alpha[self.block_to_module[block_name][0]]
                key_name = str(cur_alpha)
                loss_alphas[block_name] = {key_name: loss}
        # for name in module_names:
        #     loss_alphas[name]={}
        for alpha in alpha_space:
            absorb_input_scales, weight_scales = self._cal_scales(self.absorb_to_layer, input_maxes, alpha, tuning=True)
            self._update_scales_for_auto(absorb_input_scales, weight_scales)
            if not self.do_blockwise:
                for name in module_names:
                    losses = loss_alphas[name]
                    if str(alpha) in losses.keys():
                        continue
                    module = get_module(self.model, name)
                    output = module.q_dq_forward(module.q_input, module.input_scale, module.weight_scale)
                    loss = self._get_auto_loss(fp32_output[name], output)
                    loss_alphas[name][str(alpha)] = loss
            else:
                for block_name in self.block_names:
                    losses = loss_alphas[block_name]
                    if str(alpha) in losses.keys():
                        continue
                    block = get_module(self.model, block_name)
                    block_copy = copy.deepcopy(block)
                    for name in self.block_to_module[block_name]:
                        if name == block_name and len(self.block_to_module[block_name]) == 1:
                            module, module_copy = block, block_copy
                        else:
                            module = get_module(block, name)
                            module_copy = copy.deepcopy(module)
                        if module.weight_scale is not None:
                            module_copy.orig_layer.weight *= module.weight_scale
                        q_dq_weight = quant_dequant_w(module_copy.orig_layer)
                        module_copy.orig_layer.weight.data.copy_(q_dq_weight)
                        module_copy.do_blockwise = True
                        if not (name == block_name and len(self.block_to_module[block_name]) == 1):
                            set_module(block_copy, name, module_copy)
                    try:
                        output = block_copy(self.block_inputs[block_name])[0]
                    except:  # Llama model decoder_layer forward requires position_id
                        position_ids = torch.arange(self.block_inputs[block_name].size()[1])
                        position_ids = position_ids.view(self.block_inputs[block_name].size()[0], -1)
                        output = block_copy(self.block_inputs[block_name], position_ids=position_ids)[0]
                    loss = self._get_auto_loss(fp32_output[block_name], output)
                    loss_alphas[block_name][str(alpha)] = loss
                    del block_copy  # release memory
        return loss_alphas

    def _get_best_alpha(self, absorb_to_layer, loss_alphas, shared_criterion):
        def dict_to_list(dic):
            res = []
            for key in dic.keys():
                res.append((key, dic[key]))
            return res

        best_alpha = {}
        for ln_name in absorb_to_layer.keys():
            layer_names = absorb_to_layer[ln_name]
            cur_shared_criterion = shared_criterion
            if len(layer_names) == 1:
                cur_shared_criterion = "min"
            if cur_shared_criterion == "mean":
                loss_tmp = {}
                for alpha in loss_alphas[layer_names[0]].keys():
                    if alpha not in loss_tmp.keys():
                        loss_tmp[alpha] = 0
                    for layer_name in layer_names:
                        loss_tmp[alpha] += loss_alphas[layer_name][alpha]
                res = dict_to_list(loss_tmp)
                res.sort(key=lambda x: x[1])

                best_alpha[ln_name] = float(res[0][0])

            elif cur_shared_criterion == "min" or cur_shared_criterion == "max":
                tmp_best_alpha = []
                for layer_name in layer_names:
                    res = dict_to_list(loss_alphas[layer_name])
                    res.sort(key=lambda x: x[1])
                    tmp_best_alpha.append(float(res[0][0]))
                if cur_shared_criterion == "min":
                    best_alpha[ln_name] = min(tmp_best_alpha)
                else:
                    best_alpha[ln_name] = max(tmp_best_alpha)

            else:
                raise NotImplementedError
        return best_alpha

    def _auto_tune_alpha(
        self,
        input_maxes,
        calib_sample_num=32,
        alpha_min=0.3,
        alpha_max=0.7,
        alpha_step=0.05,
        shared_criterion="min",
        do_blockwise=False,
    ):
        """Perform alpha-tuning to obtain layer-wise optimal alpha values and adjust parameters accordingly.

        This function takes quantization of the former layers into consideration when qdq one layer
        Also, it reduces the memory usage at the cost of increasingtuning time
        TODO may have compatibility issue when setting folding=True, check whether having issues when bs!=1
        :param input_maxes: calibration data, input max
        :param calib_sample_num: sample count used to auto tuning alpha
        :param alpha_min: the min value of alpha
        :param alpha_max: the max value of alpha
        :param alpha_step:  the alpha step in search space
        :param shared_criterion: the criterion to choose alpha when multiple layers must share one same alpha
        :return:
        """
        logger.info("start sq auto tuning")
        round_num = max(
            len(str(alpha_min).split(".")[1]), len(str(alpha_max).split(".")[1]), len(str(alpha_step).split(".")[1])
        )
        alpha_space = numpy.round(numpy.arange(alpha_min, alpha_max + alpha_step, alpha_step), round_num).tolist()
        ##wrapper new module
        self._qdq_model_wrapper_for_auto(save_q_input=True)
        ##set alpha to 0.5 as default
        default_alpha = alpha_space[len(alpha_space) // 2]
        if 0.5 in alpha_space:
            default_alpha = 0.5
        default_alpha = self.default_alpha
        absorb_input_scales, weight_scales = self._cal_scales(
            self.absorb_to_layer, input_maxes, default_alpha, tuning=True
        )
        self._update_scales_for_auto(absorb_input_scales, weight_scales)
        total_cnt = 0
        tmp_cnt = 0
        alpha_update_iter = 0
        # multiply_factor is used to combine samples to calib_sample_num // 4 before summarizing the best alpha
        tune_cnt = 4
        multiply_factor = calib_sample_num // tune_cnt if calib_sample_num >= tune_cnt else calib_sample_num
        self.fp32_output_val = {}

        best_alphas = default_alpha
        if not self.dataloader:
            logger.info(f"Auto-tuning failed due to no dataloader, using {best_alphas} instead.")
            self._qdq_model_unwrapper_for_auto()
            return best_alphas
        bar = tqdm(self.dataloader, total=calib_sample_num, desc="auto tune alpha")
        try:
            for input, label in bar:
                loss_alphas = {}
                best_alphas_per_module = best_alphas
                if isinstance(best_alphas, dict):
                    for key in self.absorb_to_layer.keys():
                        layer_names = self.absorb_to_layer[key]
                        for layer_name in layer_names:
                            best_alphas_per_module[layer_name] = best_alphas_per_module[key]

                loss_tmp = self._get_one_batch_auto_loss(input, alpha_space, best_alphas_per_module, input_maxes)
                if self.do_blockwise:
                    if loss_alphas == {}:
                        for block_name in self.block_names:
                            for key in self.block_to_module[block_name]:
                                loss_alphas[key] = loss_tmp[block_name]
                    else:
                        for block_name in self.block_names:
                            for key in self.block_to_module[block_name]:
                                cur_loss = loss_alphas[key]
                                for alpha_key in cur_loss.keys():
                                    cur_loss[alpha_key] += loss_tmp[block_name][alpha_key]
                else:
                    if loss_alphas == {}:
                        loss_alphas = loss_tmp
                    else:
                        for key in loss_alphas.keys():
                            cur_loss = loss_alphas[key]
                            for alpha_key in cur_loss.keys():
                                cur_loss[alpha_key] += loss_tmp[key][alpha_key]
                total_cnt += self.dataloader.batch_size
                tmp_cnt += self.dataloader.batch_size
                if tmp_cnt // multiply_factor >= 1:
                    alpha_update_iter += 1
                    tmp_cnt = 0
                    best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, shared_criterion)
                    for key in best_alphas.keys():
                        logger.info(f"Auto alpha update iter: {alpha_update_iter}, {key}: {best_alphas[key]}")
                    absorb_input_scales, weight_scales = self._cal_scales(
                        self.absorb_to_layer, input_maxes, best_alphas, tuning=True
                    )
                    self._update_scales_for_auto(absorb_input_scales, weight_scales)
                    # does not need to reset the weight_scale_dict, because use the weight of ori_layer, no change
                    # self.weight_scale_dict = {}
                if total_cnt >= calib_sample_num:
                    break
        except:
            for input in bar:
                loss_alphas = {}
                best_alphas_per_module = best_alphas
                if isinstance(best_alphas, dict):
                    for key in self.absorb_to_layer.keys():
                        layer_names = self.absorb_to_layer[key]
                        for layer_name in layer_names:
                            best_alphas_per_module[layer_name] = best_alphas_per_module[key]

                loss_tmp = self._get_one_batch_auto_loss(input, alpha_space, best_alphas_per_module, input_maxes)
                if self.do_blockwise:
                    if loss_alphas == {}:
                        for block_name in self.block_names:
                            for key in self.block_to_module[block_name]:
                                loss_alphas[key] = loss_tmp[block_name]
                    else:
                        for block_name in self.block_names:
                            for key in self.block_to_module[block_name]:
                                cur_loss = loss_alphas[key]
                                for alpha_key in cur_loss.keys():
                                    cur_loss[alpha_key] += loss_tmp[block_name][alpha_key]
                else:
                    if loss_alphas == {}:
                        loss_alphas = loss_tmp
                    else:
                        for key in loss_alphas.keys():
                            cur_loss = loss_alphas[key]
                            for alpha_key in cur_loss.keys():
                                cur_loss[alpha_key] += loss_tmp[key][alpha_key]
                total_cnt += self.dataloader.batch_size
                tmp_cnt += self.dataloader.batch_size
                if tmp_cnt // multiply_factor >= 1:
                    alpha_update_iter += 1
                    tmp_cnt = 0

                    best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, shared_criterion)
                    for key in best_alphas.keys():
                        logger.info(f"Auto alpha update iter: {alpha_update_iter}, {key}: {best_alphas[key]}")
                    absorb_input_scales, weight_scales = self._cal_scales(
                        self.absorb_to_layer, input_maxes, best_alphas, tuning=True
                    )
                    self._update_scales_for_auto(absorb_input_scales, weight_scales)
                    # self.weight_scale_dict = {}
                if total_cnt >= calib_sample_num:
                    break

        best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, shared_criterion)
        for key in best_alphas.keys():
            logger.info(f"Final alpha {key}:{best_alphas[key]}")
        max_op, max_ratio, max_key = "", 0, ""
        ratio_info = {}
        for key in self.absorb_to_layer:
            for op_name in self.absorb_to_layer[key]:
                fp32_norm, loss_ = (
                    torch.sum(torch.stack(self.fp32_output_val[op_name])),
                    loss_alphas[op_name][str(best_alphas[key])],
                )
                ratio = loss_ / fp32_norm
                max_op = op_name if ratio > max_ratio else max_op
                max_key = key if ratio > max_ratio else max_key
                max_ratio = max(ratio, max_ratio)
                ratio_info[op_name] = ratio
                logger.debug(
                    f"final loss: {op_name}: {loss_}; @alpha {best_alphas[key]}; \
                    fp32_output norm: {fp32_norm}; ratio: {ratio}"
                )
        import operator

        ratio_info = dict(sorted(ratio_info.items(), key=operator.itemgetter(1), reverse=True))
        for key in list(ratio_info.keys()):
            logger.debug(f"sorted opname-ratio: {key}:  {ratio_info[key]}")
        if max_op != "":
            logger.debug(
                f"max loss: {max_op}: {loss_alphas[max_op][str(best_alphas[max_key])]} @alpha {best_alphas[max_key]}\
                fp32_output norm: {torch.sum(torch.stack(self.fp32_output_val[max_op]))}; ratio: {max_ratio}"
            )
        self._qdq_model_unwrapper_for_auto()
        logger.info("auto tuning done")
        return best_alphas

    def transform(
        self,
        alpha=0.5,
        folding=False,
        percentile=100,
        op_types=[torch.nn.Linear, torch.nn.Conv2d],
        scales_per_op=False,
        calib_iter=100,
        auto_alpha_args={
            "alpha_min": 0.0,
            "alpha_max": 1.0,
            "alpha_step": 0.1,
            "shared_criterion": "mean",
            "do_blockwise": False,
        },
        weight_clip=True,
        default_alpha=0.5,
    ):
        """The main entry of smooth quant
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, please refer
        to the paper for more details
        :param folding: whether insert mul(False) or just allow foldable layers(True) for SmoothQuant
        :param percentile: remove the activation outlier when calculating the scale
        :param op_types: The op typed to be smooth quantized
        :param scales_per_op: Not supported now
        :param calib_iter: Data size for calibration
        :param weight_clip: Whether to clip weight_max when calculating scales.

        :param auto_alpha_args: Hyperparameters used to set the alpha search space in SQ auto-tuning.
            By default the search space is 0.0-1.0 with step_size 0.1.
            do_blockwise: Whether to do blockwise auto-tuning.
        :param default_alpha: A hyperparameter that is used in SQ auto-tuning; by default it is 0.5.
        :return: A FP32 model with the same architecture as the orig model but with different weight which will be
        benefit to quantization.
        """
        if isinstance(auto_alpha_args, dict):
            self.do_blockwise = auto_alpha_args.get("do_blockwise", False)
        else:
            self.do_blockwise = False
        if self.do_blockwise:
            self.block_names = self.get_blocks()
            logger.info("Blockwise auto-tuning will be performed")
        if not isinstance(self.model, torch.nn.Module):
            logger.warning("smooth quant is ignored since the model is not a torch module")
            return self.model

        if folding:
            self.insert_mul, self.allow_absorb = False, True
        else:
            self.insert_mul, self.allow_absorb = True, False
        if isinstance(alpha, float) and (alpha < 0 or alpha > 1):
            logger.warning("reset alpha to in range [0.0, 1.0]")

            alpha = numpy.clip(alpha, 0.0, 1.0)

        self.weight_clip = weight_clip
        self.default_alpha = default_alpha
        self.auto_alpha_args = auto_alpha_args
        self.recover()
        need_calibration = self._check_need_calibration(alpha, percentile, op_types, scales_per_op, calib_iter)
        with torch.no_grad():
            str_op_types = [i.__name__ for i in op_types]
            input_maxes_abs = self.input_maxes_abs
            if need_calibration:  ##avoid multiple calibaration during tuning if the only difference is alpha
                if self.insert_mul:
                    self.self_absorb_layers = self._get_all_layer_names(op_types)  # TODO: only support linear now.
                    # fetch modules with the same input
                    group_modules = self._trace(str_op_types, skip_unsupported_layers=False)
                    if group_modules is not None:
                        # use one input for qkv
                        for k, v in group_modules.items():
                            for i in v:
                                if i in self.self_absorb_layers:
                                    self.self_absorb_layers.pop(i)
                            self.self_absorb_layers[v[0]] = v
                        logger.debug(f"self_absorb_layers:{self.self_absorb_layers}")
                if self.allow_absorb:
                    self.absorb_to_layer, no_absorb_layers = self._trace(
                        str_op_types
                    )  ##TODO we need to insert mul layer for no_absorb_layers later
                    if self.absorb_to_layer is None and no_absorb_layers is None:
                        return self.model

                # remove self.self_absorb_layers if it exists in self.absorb_to_layer
                for k, v in self.absorb_to_layer.items():
                    for i in v:
                        if i in self.self_absorb_layers:
                            self.self_absorb_layers.pop(i)
                self.absorb_to_layer.update(self.self_absorb_layers)

                if self.absorb_to_layer is None and no_absorb_layers is None:
                    logger.warning(
                        "sorry, could not trace the model, smooth quant is ignored."
                        "If you are using huggingface model,"
                        "you could set torchscript to True "
                    )
                    return self.model

                if self.do_blockwise:
                    module_names = self._get_sq_layer_names()
                    block_names, self.block_to_module = self.block_names, {}
                    for block in block_names:
                        self.block_to_module[block] = []
                    for module in module_names:
                        checked = False
                        for block in block_names:
                            if block + "." in module:
                                self.block_to_module[block].append(module)
                                checked = True
                        if not checked:
                            self.block_to_module[module] = [module]
                    self.block_names = list(self.block_to_module.keys())
                    logger.info(f"Blockwise auto-tuning: {len(self.block_names)} blocks found")
                    logger.debug(f"Blockwise auto-tuning blocks info: {self.block_to_module}")

                input_maxes_abs = self._calibrate(self.absorb_to_layer, calib_iter, percentile)

                # Check if input_maxes match self.absorb_to_layer
                # (due to self._get_all_layer_names use layer tree instead of forward_path)
                if not folding:
                    diff_modules = set(self.absorb_to_layer.keys()).difference(input_maxes_abs.keys())
                    for d in diff_modules:
                        del self.absorb_to_layer[d]

                scale_memo_use = 0
                for key in self.absorb_to_layer:
                    layer_name = self.absorb_to_layer[key][0]
                    input_max = input_maxes_abs[layer_name]
                    scale_memo_use += 4 * input_max.shape[0] * len(self.absorb_to_layer[key])
                if alpha == "auto":
                    alpha_space = (auto_alpha_args["alpha_max"] - auto_alpha_args["alpha_min"]) / auto_alpha_args[
                        "alpha_step"
                    ] + 1
                    scale_memo_use *= alpha_space
                self._save_scale = enough_memo_store_scale(self.device, scale_memo_use)

                if alpha == "auto":
                    self.alpha_per_layer = self._auto_tune_alpha(
                        input_maxes_abs, calib_sample_num=32, **auto_alpha_args
                    )  ##save the alpha

            if alpha == "auto":
                alpha = self.alpha_per_layer
            example_inputs = self._get_example_input()
            if example_inputs is not None:
                out_pre_sq = model_forward_per_sample(self.model, example_inputs, self.device)

            if folding:
                self._save_scale = False
            if self.record_max_info:
                # max_info is recorded in self.max_value_info
                self._adjust_parameters(self.absorb_to_layer, input_maxes_abs, alpha)
                self.model._smoothquant_optimized = False
                return self.model

            self.weight_scale_info, self.absorb_scales_info = self._adjust_parameters(
                self.absorb_to_layer, input_maxes_abs, alpha
            )

            self.model._smoothquant_optimized = True
            if example_inputs is not None:
                # Check mathematical equivelancy
                out_post_sq = model_forward_per_sample(self.model, example_inputs, self.device)

                if not self.output_is_equal(out_post_sq, out_pre_sq):
                    logger.warning(
                        "Mathematical equivelancy of Smoothquant is not preserved. "
                        "Please kindly report this issue to https://github.com/intel/neural-compressor."
                    )
            else:
                logger.warning(" Could not get example input, equivelancy check is skipped")

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
            logger.warning(
                "Automatically check failed, Please check equivelancy manually "
                "between out_pre_sq and out_post_sq if necessary."
            )
            return True

    def recover(self):
        """Recover the model weights
        :return:"""
        with torch.no_grad():
            for key in self.weight_scale_info:
                self._scale_layer_weight(key, 1.0 / self.weight_scale_info[key])
            for key in self.absorb_scales_info:
                self._absorb_scales(key, 1.0 / self.absorb_scales_info[key])
            self.weight_scale_info = {}  ##clear the data
            self.absorb_scales_info = {}

    def _get_all_layer_names(self, op_types=[torch.nn.Linear]):
        """Try the model to find the layers which can be smooth quantized.

        :param op_types: The op types to be smooth quantized
        :return:
        self_absorb_layer: A dict, absorb layer name (itself): layers to be smooth quantized
        """
        self_absorb_layer = {}
        op_types = [torch.nn.Linear]  # TODO： only support SQLinearWrapper
        for name, module in self.model.named_modules():
            if isinstance(module, tuple(op_types)):
                self_absorb_layer[name] = [name]
        return self_absorb_layer

    def _get_example_input(self):
        if self.dataloader is None and self.example_inputs is None:
            return None
        if self.example_inputs is None:
            try:
                for idx, (input, label) in enumerate(self.dataloader):
                    self.example_inputs = input
                    break
            except:
                for idx, input in enumerate(self.dataloader):
                    self.example_inputs = input
                    break

        return self.example_inputs

    def _trace(self, op_types, skip_unsupported_layers=True):
        """Try the model to find the layers which can be smooth quantized.

        :param op_types: The op types to be smooth quantized
        :return:
        absorb_to_layer: A dict, absorb layer name:layers to be smooth quantized
        no_absorb_layers: A list saving the layers which could not find the absorb layer
        """
        tg = GraphTrace()
        self._get_example_input()
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(
            self.traced_model,
            self.example_inputs,
            op_types,
            skip_unsupported_layers=skip_unsupported_layers,
        )
        if not skip_unsupported_layers:
            return absorb_to_layer
        if absorb_to_layer is None and no_absorb_layers is None:
            logger.warning(
                "sorry, could not trace the model, smooth quant is skipped."
                "If you are using huggingface model,"
                "you could set torchscript to True "
                "when loading the model or set the return_dict to False"
            )
        elif absorb_to_layer == {}:
            logger.warning("could not find any layer to be absorbed")
        else:
            to_absorb_cnt = 0
            for key, item in absorb_to_layer.items():
                to_absorb_cnt += len(item)
            logger.info(
                f" {to_absorb_cnt} out of {to_absorb_cnt + len(no_absorb_layers)} "
                f"layers could be absorbed in smooth quant"
            )
        return absorb_to_layer, no_absorb_layers


def get_parent(node, all_parents=False):
    if node.inputs() is None:
        return None
    elif len(list(node.inputs())) == 0:
        return None
    if not all_parents:
        return list(node.inputs())[0].node()
    else:
        return list(node.inputs())


class GraphTrace:
    """"""

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
            "LPLayerNorm": "aten::layer_norm",  ##mpt_chat
        }

        ##TODO potential bug, need to check only have one bug
        ##TODO, must satisfy af(x)=f(ax),current skip layer may be incomplete
        self.skip_ops_to_find_absorb = ["aten::to", "aten::relu", "aten::leaky_relu", "aten::hardtanh"]

        self.could_absorb_layers = [
            "aten::layer_norm",
            "aten::batch_norm",
            "aten::linear",
            "aten::_convolution",
            "aten::group_norm",
            "aten::instance_norm",
            "aten::mul",
        ]  ##TODO,support more norm

    def trace(self, model, dummy_input):
        traced_model = None
        optimize_numerics = False
        orig_device = next(model.parameters()).device.type
        if orig_device != "cpu" and orig_device != "meta":  # pragma: no cover
            model = model.to("cpu")
            dummy_input = move_input_to_device(dummy_input, "cpu")
        if isinstance(dummy_input, dict) or isinstance(dummy_input, UserDict):
            try:
                traced_model = torch.jit.trace(
                    model, example_kwarg_inputs=dict(dummy_input), strict=False, check_trace=False
                )
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
        model = model.to(orig_device)
        return traced_model

    def get_nodes(self, traced_model, op_types=["Linear"]):
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
                    parent_out_kinds.discard("aten::size")

                    if parent_out_kinds == parent_out_kinds.intersection(self.could_absorb_layers):
                        prev_absorb_layer.append(parent)
                    elif parent_out_kinds.intersection(self.skip_ops_to_find_absorb):
                        res = self.skip_op_absorb_helper(parent)
                        prev_absorb_layer.append(parent) if res else prev_absorb_layer.append(None)
                    else:  # When parent to multiple ops, sq transformation could be wrong.
                        prev_absorb_layer.append(None)
                else:
                    prev_absorb_layer.append(None)
                break
        return prev_absorb_layer

    def skip_op_absorb_helper(self, parent_node):
        for val_user in list(parent_node.outputs())[0].uses():
            next_node = val_user.user
            if next_node.kind() == "aten::size":
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

    def _check_valid_conv(self, module):
        """Remove group conv except depthwise conv
        :param module:

        :return:
        """
        if not isinstance(module, torch.nn.Conv2d):
            return True
        if module.groups > 1:
            if module.in_channels == module.out_channels and module.groups == module.in_channels:
                return True
            else:
                return False
        return True

    def get_absorb_to_layer(self, model, example_input, op_types, skip_unsupported_layers=True):
        traced_model = self.trace(model, example_input)
        if traced_model is None:
            return None, None

        aten_op_types = self.mapping_torch_module_to_aten(op_types)
        nodes_types = self.get_nodes(traced_model, aten_op_types)
        nodes = [node_type[0] for node_type in nodes_types]
        nodes_prev_absorb = self.get_prev_absorb_layer(nodes)
        absorb_to_layer = {}
        no_absorb_layers = []
        for index, absorb in enumerate(nodes_prev_absorb):
            if absorb is None:
                no_absorb_layers.append(".".join(nodes[index].scopeName().split("/")[-1].split(".")[1:]))
                continue
            node = nodes[index]
            layer_name = ".".join(node.scopeName().split("/")[-1].split(".")[1:])
            absorb_name = ".".join(absorb.scopeName().split("/")[-1].split(".")[1:])
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
                if (layer_type not in self.supported_torch_module_to_aten.keys()) or not self._check_valid_conv(layer):
                    supported = False
                    no_absorb_layers.extend(absorb_to_layer[key])
                    break
            if supported:
                res[key] = absorb_to_layer[key]
        return res
