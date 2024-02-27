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
    from neural_compressor.utils import logger
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
            if input is None:
                continue
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


# def get_block_names(model):
#     """Get the block names for transformers-like networks.

#     Args:
#     model: The model.

#     Returns:
#     block_names: A list of block names.
#     """
#     block_names = []
#     target_m = None
#     for n, m in model.named_modules():
#         if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
#             target_m = (n, m)
#     for n, m in target_m[1].named_children():
#         block_names.append(target_m[0] + "." + n)
#     return block_names


def model_forward_per_sample(model, sample, device):
    try:
        output = forward_wrapper(model, sample, device)
        return output

    except Exception as e:
        output = forward_wrapper(model, sample[0], device)
        return output


def quant_dequant_w_v1(m, num_bits=8, scheme="sym"):
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


# def quant_dequant_w(x, scale, num_bits=8):  ##default sym
#     scale = scale.unsqueeze(dim=1)
#     q_min, q_max = -(2.0 ** (num_bits - 1)), 2.0 ** (num_bits - 1) - 1.0
#     q_x = torch.round(x / scale)
#     q_x.clamp_(q_min, q_max)
#     return scale * q_x


def quant_dequant_x_v1(x, min_x=None, max_x=None, num_bits=8):
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


# def quant_dequant_x(x, scale, bias, num_bits=8):  ##default asym
#     q_min, q_max = 0, 2.0**num_bits - 1.0
#     # if max_x is None or min_x is None:
#     #     max_x, min_x = torch.max(x), torch.min(x)
#     # else:
#     #     max_x = torch.max(max_x)
#     #     min_x = torch.min(min_x)
#     # scale = (max_x - min_x) / (2**num_bits - 1)
#     # scale = torch.clip(scale, min=eps)
#     # bias = torch.round((0 - min_x) / scale)
#     q_x = torch.round(x / scale + bias)
#     q_x.clamp_(q_min, q_max)
#     return scale * (q_x - bias)


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


def cal_scale(input_max_abs, weights, alpha, weight_max_lb=1e-5):
    weights = torch.cat(weights, dim=0)
    weight_max = torch.max(torch.abs(weights), dim=0)[0]
    weight_max = torch.clip(weight_max, weight_max_lb)
    input_power = torch.pow(input_max_abs, alpha)
    logger.debug(f"{max(input_max_abs)}, {min(input_max_abs)}")
    weight_power = torch.pow(weight_max, 1 - alpha)
    weight_scale = torch.clip(input_power / weight_power, min=1e-5)
    weight_scale[input_power == 0] = 1.0
    return weight_scale


def reshape_in_channel_to_last(layer_name, model):
    """Move the input channel to the last dim
    :param layer_name: Layer name
    :return: The reshaped weight."""
    layer = get_module(model, layer_name)
    if layer.__class__.__name__ == "WrapperLayer":
        layer = layer.orig_layer

    weight = layer.weight  ##TODO oc*ic, support transposed conv
    if len(weight.shape) == 4:
        weight = weight.permute(0, 2, 3, 1)
        weight = weight.reshape(-1, weight.shape[-1])
    return weight


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
        q_dq_weight = quant_dequant_w_v1(layer_copy)
        layer_copy.weight.data.copy_(q_dq_weight)
        if input_scale is None:
            x = quant_dequant_x_v1(x, self.input_min, self.input_max)
        else:
            x = input_scale * x
            x = quant_dequant_x_v1(x, self.input_min * input_scale, self.input_max * input_scale)  ##FIXME
        output = layer_copy(x)
        return output

    def q_dq_forward_blockwise(self, x, input_scale):
        layer_copy = copy.deepcopy(self.orig_layer)
        if input_scale is None:
            x = quant_dequant_x_v1(x, self.input_min, self.input_max)
        else:
            x = input_scale * x
            x = quant_dequant_x_v1(x, self.input_min * input_scale, self.input_max * input_scale)  ##FIXME
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


def reshape_scale_as_input(layer, scale):
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


def reshape_scale_as_weight(layer, scale):
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


TUNERS = {}


def register_autotune(name):
    """Class decorator to register a smoothquant auto-tune subclass.

    :return: the class of register
    """

    def register(auto_tune):
        TUNERS[name] = auto_tune
        return auto_tune

    return register
