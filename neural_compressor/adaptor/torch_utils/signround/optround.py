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

    torch = LazyImport("torch")

    from neural_compressor.utils import logger
except:  # pragma: no cover
    import logging

    import torch

    logger = logging.getLogger()
import copy
from collections import UserDict
from functools import partial
from typing import Union

from torch.amp import autocast
from torch.autograd import Function
from torch.functional import F


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
        else:
            module = module
    setattr(module, name_list[-1], new_module)


def quant_weight_asym(weight, num_bits=4, v=0, min_scale=0, max_scale=0, use_sigmoid=False):
    """
    Quantizes and dequantizes weight asymmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        use_sigmoid: Boolean indicating whether to use sigmoid; found useful in some scenarios for adam

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """

    maxq = torch.tensor(2 ** num_bits - 1)
    zeros = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    if isinstance(min_scale, torch.Tensor):
        wmin_tmp = torch.minimum(weight.min(1)[0], zeros)
        wmax_tmp = torch.maximum(weight.max(1)[0], zeros)
        if use_sigmoid:
            wmin_tmp *= torch.sigmoid(8 * (min_scale + 0.5))
            wmax_tmp *= torch.sigmoid(8 * (max_scale + 0.5))
        else:
            wmin_tmp *= torch.sigmoid(1.0 + min_scale)
            wmax_tmp *= torch.sigmoid(1.0 + max_scale)
        wmax = torch.maximum(wmax_tmp, wmin_tmp)
        wmin = torch.minimum(wmax_tmp, wmin_tmp)
    else:
        wmin = torch.minimum(weight.min(1)[0], zeros)
        wmax = torch.maximum(weight.max(1)[0], zeros)
    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = (wmax - wmin) / maxq
    zp = round_ste(-wmin / scale)
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp), scale, zp


def quant_weight_sym(
        weight, num_bits=4, v=0, min_scale=0, max_scale=0, use_sigmoid=False
):
    """
     Quantizes and dequantizes weight symmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        use_sigmoid: Boolean indicating whether to use sigmoid; found useful in some scenarios for adam

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** (num_bits - 1) - 1).to(weight.device)
    minq = torch.tensor(-(2 ** (num_bits - 1))).to(weight.device)
    if num_bits == 1:
        maxq = torch.tensor(2 ** (num_bits - 1))
        minq = torch.tensor(2 ** (num_bits - 1) - 1)

    wmax = torch.abs(weight).max(1)[0]
    if use_sigmoid:
        wmax *= torch.sigmoid(8 * (min_scale + 0.5))
    else:
        wmax *= 1.0 + max_scale
    tmp = wmax == 0
    wmax[tmp] = +1
    scale = wmax / ((maxq - minq) / 2)
    scale.unsqueeze_(dim=-1)
    q = torch.clamp(round_ste(weight / scale + v), minq, maxq)
    return scale * q, scale, None


def quant_weight_actor(weight, num_bits, scheme, v, min_scale, max_scale, use_sigmoid):
    """
     Quantizes and dequantizes weight symmetrically or asymmetrically .

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        scheme: Sym or asym
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        use_sigmoid: Boolean indicating whether to use sigmoid; found useful in some scenarios for adam

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    assert num_bits > 0, "num_bits should be larger than 0"
    if scheme == "sym":
        return quant_weight_sym(weight, num_bits, v, min_scale, max_scale, use_sigmoid)
    else:
        return quant_weight_asym(weight, num_bits, v, min_scale, max_scale, use_sigmoid)


def quant_weight(
        weight,
        num_bits=4,
        group_size=-1,
        scheme="asym",
        v=0,
        min_scale=0,
        max_scale=0,
        use_sigmoid=False,
        return_scalezp=False,
):
    """
     Quantizes and dequantizes weight, handing the group size issue .

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: The number of elements shares scale and zero point
        scheme: Sym or asym
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        use_sigmoid: Boolean indicating whether to use sigmoid; found useful in some scenarios for adam

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    if group_size == -1 or weight.shape[1] < group_size:
        weight, scale, zp = quant_weight_actor(
            weight,
            num_bits,
            scheme=scheme,
            v=v,
            min_scale=min_scale,
            max_scale=max_scale,
            use_sigmoid=use_sigmoid,
        )
        if return_scalezp:
            return weight, scale, zp

        return weight
    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        if isinstance(v, torch.Tensor):
            v = v.reshape(-1, group_size)

        weight, scale, zp = quant_weight_actor(
            weight,
            num_bits,
            scheme=scheme,
            v=v,
            min_scale=min_scale,
            max_scale=max_scale,
            use_sigmoid=use_sigmoid,
        )

        weight = weight.reshape(orig_shape)
        if return_scalezp:
            return weight, scale, zp
        return weight
    else:
        split_index = weight.shape[1] // group_size * group_size
        weight1 = weight[:, :split_index]
        weight1 = weight1.reshape(-1, group_size)
        if isinstance(v, torch.Tensor):
            v1 = v[:, :split_index]
            v1 = v1.reshape(-1, group_size)
            v2 = v[:, split_index:]
        else:
            v1 = 0
            v2 = 0
        if isinstance(min_scale, torch.Tensor):
            min_scale = min_scale.view(weight.shape[0], -1)
            max_scale = max_scale.view(weight.shape[0], -1)
            min_scale_1 = min_scale[:, : weight.shape[1] // group_size]
            min_scale_2 = min_scale[:, weight.shape[1] // group_size:]
            max_scale_1 = max_scale[:, : weight.shape[1] // group_size]
            max_scale_2 = max_scale[:, weight.shape[1] // group_size:]
            min_scale_1 = min_scale_1.reshape(-1)
            min_scale_2 = min_scale_2.reshape(-1)
            max_scale_1 = max_scale_1.reshape(-1)
            max_scale_2 = max_scale_2.reshape(-1)
        else:
            min_scale_1 = min_scale
            min_scale_2 = min_scale
            max_scale_1 = max_scale
            max_scale_2 = max_scale

        (
            weight1,
            scale1,
            zp1,
        ) = quant_weight_actor(
            weight1,
            num_bits,
            scheme=scheme,
            v=v1,
            min_scale=min_scale_1,
            max_scale=max_scale_1,
            use_sigmoid=use_sigmoid,
        )
        weight1 = weight1.reshape(orig_shape[0], split_index)
        weight2 = weight[:, split_index:]
        weight2, scale2, zp2 = quant_weight_actor(
            weight2,
            num_bits,
            scheme=scheme,
            v=v2,
            min_scale=min_scale_2,
            max_scale=max_scale_2,
            use_sigmoid=use_sigmoid,
        )
        weight = torch.cat([weight1, weight2], dim=1)  ##TODO not supported output group
        if return_scalezp:
            scale = torch.cat([scale1, scale2], dim=0)
            zp = torch.cat([zp1, zp2], dim=0)
            return weight, scale, zp
        return weight


class WrapperMultiblock(torch.nn.Module):
    """
    Wrapper multiple blocks to one module
    """
    def __init__(self, module_list):
        """
        Init function
        Args:
            module_list: a list of consecutive modules
        """
        super(WrapperMultiblock, self).__init__()
        self.layers = torch.nn.ModuleList(module_list)

    def forward(self, x, **kwargs):
        """
        forward function
        Args:
            x: Input
            **kwargs: Other inputs

        Returns:
            output

        """
        hidden_states = x
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(hidden_states, **kwargs)
            hidden_states = layer_outputs
            if isinstance(hidden_states, tuple) or isinstance(hidden_states, list):
                hidden_states = layer_outputs[0]
        return hidden_states


def get_block_names(model):
    """
    Get the block names for transformers-like networks, this may have issues
    Args:
        model: The model

    Returns:
        all the block names

    """
    block_names = []
    target_m = None
    for n, m in model.named_modules():
        if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
            target_m = (n, m)
    for n, m in target_m[1].named_children():
        block_names.append(target_m[0] + "." + n)
    return block_names


class SaveInputs:
    def __init__(self, model, dataloader, seqlen=256, block_name=None):
        self.model = model.eval()
        self.dataloader = dataloader
        self.inputs = {}
        self.block_name = block_name
        self.seqlen = seqlen

    @torch.no_grad()
    def get_forward_func(self, name):
        def forward(_, hidden_states, *positional_args, **kwargs):  ##This may have bug for other models
            dim = int(
                (hasattr(self.model, "config") and "chatglm" in self.model.config.model_type)
            )
            if name in self.inputs:
                data = torch.cat([self.inputs[name]["input_ids"], hidden_states.to("cpu")], dim=dim)
                self.inputs[name]["input_ids"] = data
            else:
                self.inputs[name] = {}
                self.inputs[name]["input_ids"] = hidden_states.to("cpu")

            if "positional_inputs" not in self.inputs[name]:
                self.inputs[name]["positional_inputs"] = []
            for idx, item in enumerate(positional_args):
                self.inputs[name]["positional_inputs"] = move_input_to_device(positional_args)

            for key in kwargs.keys():
                if isinstance(kwargs[key], torch.Tensor) or isinstance(kwargs[key], list) or (key == "alibi"):
                    if "attention_mask" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = None
                        if kwargs[key] is not None:
                            if self.inputs[name][key] is not None:
                                self.inputs[name][key] = torch.cat(
                                    [self.inputs[name][key], kwargs[key].to("cpu")], dim=0
                                )
                            else:
                                self.inputs[name][key] = kwargs[key].to("cpu")
                    elif "alibi" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = None
                        if isinstance(kwargs[key], torch.Tensor):
                            alibi = kwargs[key]
                            batch = kwargs["attention_mask"].shape[0]
                            alibi = alibi.reshape(batch, -1, alibi.shape[1], alibi.shape[2])
                            if self.inputs[name][key] is not None:
                                self.inputs[name][key] = torch.cat([self.inputs[name][key], alibi.to("cpu")], dim=0)
                            else:
                                self.inputs[name][key] = alibi.to("cpu")
                    elif key not in self.inputs[name].keys():
                        self.inputs[name][key] = move_input_to_device(kwargs[key], device=torch.device("cpu"))
            raise NotImplementedError

        return forward

    @torch.no_grad()
    def get_inputs(self, n_samples=512):
        total_cnt = 0
        self._replace_forward()
        for data in self.dataloader:
            if data is None:
                continue

            input_ids = data["input_ids"].to(self.model.device)
            if input_ids.shape[-1] < self.seqlen:
                continue
            if total_cnt + input_ids.shape[0] > n_samples:
                input_ids = input_ids[: n_samples - total_cnt, ...]
            try:
                self.model(input_ids)
            except NotImplementedError:
                pass
            except Exception as error:
                logger.error(error)
            total_cnt += input_ids.shape[0]
            if total_cnt >= n_samples:
                break
        self._recover_forward()
        return self.inputs[self.block_name]

    def _recover_forward(self):
        for n, m in self.model.named_modules():
            if n == self.block_name:
                m.forward = m.orig_forward
                delattr(m, "orig_forward")
                break

    def _replace_forward(self):
        for n, m in self.model.named_modules():
            if n == self.block_name:
                m.orig_forward = m.forward
                m.forward = partial(self.get_forward_func(n), m)
                break


def round_ste(x: torch.Tensor):
    """
    cp from https://github.com/OpenGVLab/OmniQuant/blob/main/quantize/quantizer.py
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def sampling_inputs(input_ids, input_others, indices, seqlen):
    if len(input_ids.shape) == 3:
        if int(len(input_others["positional_inputs"]) > 0):
            current_input_ids = input_ids[:, indices, :]
        else:
            current_input_ids = input_ids[indices, :, :]
    else:
        n_samples = input_ids.shape[0] // seqlen
        current_input_ids = input_ids.view(n_samples, seqlen, -1)
        current_input_ids = current_input_ids[indices, :, :]
        current_input_ids = current_input_ids.reshape(-1, input.shape[-1])

    current_input_others = {}
    current_input_others["positional_inputs"] = input_others["positional_inputs"]
    for key in input_others.keys():
        if "attention_mask" in key or "alibi" in key:
            current_input_others[key] = None
            if input_others[key] is not None:
                current_input_others[key] = input_others[key][indices, ...]
        else:
            current_input_others[key] = input_others[key]

    return current_input_ids, current_input_others


# def move_to_device(input_ids, inputs_others=None, device=torch.device("cpu")):
#     if input_ids != None:
#         input_ids = input_ids.to(device)
#     if inputs_others is not None:
#         for key in inputs_others.keys():
#             inputs_others[key] = inputs_others[key].to(device)
#         return input_ids, inputs_others
#     return input_ids


def move_input_to_device(input, device=torch.device("cpu")):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = move_input_to_device(input[inp], device)
    elif isinstance(input, list) or isinstance(input, tuple):
        input_res = []
        for inp in input:
            input_res.append(move_input_to_device(inp, device))
        input = input_res
    return input


def block_forward(block, input_ids, input_others, amp=False, device=torch.device("cpu")):
    if input_ids.device != device:
        # input_ids, input_others = move_to_device(input_ids, input_others, device)
        input_ids = move_input_to_device(input_ids, device)
        input_others = move_input_to_device(input_others, device)
    if "alibi" in input_others.keys():
        attention_mask = input_others["attention_mask"]
        alibi = input_others["alibi"]
        alibi = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
        if amp and device != torch.device("cpu"):
            with autocast(device_type="cuda"):
                output = block(
                    input_ids, attention_mask=attention_mask, alibi=alibi
                )  ##TODO is this correct for all models with alibi?
        elif amp and device == torch.device("cpu"):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
        else:
            output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
    else:
        input_tuple = input_others.pop("positional_inputs", None)
        if amp and device != torch.device("cpu"):
            with autocast(device_type="cuda"):
                output = block.forward(input_ids, *input_tuple, **input_others)
        elif amp and device == torch.device("cpu"):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = block.forward(input_ids, *input_tuple, **input_others)
        else:
            output = block.forward(input_ids, *input_tuple, **input_others)
    if isinstance(output, list) or isinstance(output, tuple):
        output = output[0]
    return output


def get_scale_shape(weight, group_size):
    if group_size == -1 or weight.shape[1] < group_size:
        shape = weight.shape[0]
    else:
        shape = weight.shape[0] * ((weight.shape[1] + group_size - 1) // group_size)

    return shape


class WrapperTransformerConv1d(torch.nn.Module):
    def __init__(self, orig_layer, num_bits, group_size, scheme, use_sigmoid, enable_minmax_tuning=True):
        super(WrapperTransformerConv1d, self).__init__()
        self.orig_layer = orig_layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.scheme = scheme
        self.use_sigmoid = use_sigmoid
        device = self.orig_layer.weight.device
        self.weight_t = self.orig_layer.weight.t()
        self.value = torch.nn.Parameter(torch.zeros(self.weight_t.shape, device=device), requires_grad=True)
        shape = get_scale_shape(self.weight_t, group_size)

        if enable_minmax_tuning:
            self.min_scale = torch.nn.Parameter(torch.zeros(shape, device=device), requires_grad=True)
            self.max_scale = torch.nn.Parameter(torch.zeros(shape, device=device), requires_grad=True)
        else:
            self.min_scale = torch.tensor(0, device=device)
            self.max_scale = torch.tensor(0, device=device)

    def unwrapper(self, grad, min_scale_grad, max_scale_grad):
        min_scale_grad.clamp_(-1, 0)
        max_scale_grad.clamp_(-1, 0)
        weight_q, scale, zp = quant_weight(
            self.weight_t,
            self.num_bits,
            self.group_size,
            self.scheme,
            grad,
            min_scale_grad,
            max_scale_grad,
            use_sigmoid=self.use_sigmoid,
            return_scalezp=True,
        )
        self.orig_layer.weight.data.copy_(weight_q.t())
        self.orig_layer.weight.grad = None
        self.orig_layer.scale = scale.to("cpu")
        self.orig_layer.zp = zp.to("cpu")
        return self.orig_layer

    def forward(self, x):
        with torch.no_grad():
            self.min_scale.clamp_(-1, 0)
            self.max_scale.clamp_(-1, 0)
        weight_q = quant_weight(
            self.weight_t,
            self.num_bits,
            self.group_size,
            self.scheme,
            self.value,
            self.min_scale,
            self.max_scale,
            use_sigmoid=self.use_sigmoid,
        )
        size_out = x.size()[:-1] + (self.orig_layer.nf,)
        x = torch.addmm(self.orig_layer.bias, x.view(-1, x.size(-1)), weight_q.t())
        x = x.view(*size_out)
        return x


class WrapperLinear(torch.nn.Module):
    def __init__(self, orig_layer, num_bits, group_size, scheme, use_sigmoid, enable_minmax_tuning=True):
        super(WrapperLinear, self).__init__()
        self.orig_layer = orig_layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.scheme = scheme
        self.use_sigmoid = use_sigmoid
        device = self.orig_layer.weight.device
        self.value = torch.nn.Parameter(torch.zeros(self.orig_layer.weight.shape, device=device), requires_grad=True)
        shape = get_scale_shape(self.orig_layer.weight, group_size)

        if enable_minmax_tuning:
            self.min_scale = torch.nn.Parameter(torch.zeros(shape, device=device), requires_grad=True)
            self.max_scale = torch.nn.Parameter(torch.zeros(shape, device=device), requires_grad=True)
        else:
            self.min_scale = torch.tensor(0, device=device)
            self.max_scale = torch.tensor(0, device=device)

    def unwrapper(self, grad, min_scale_grad, max_scale_grad):
        min_scale_grad.clamp_(-1, 0)
        max_scale_grad.clamp_(-1, 0)

        q_dq_weight, scale, zp = quant_weight(
            self.orig_layer.weight,
            self.num_bits,
            self.group_size,
            self.scheme,
            grad,
            min_scale_grad,
            max_scale_grad,
            use_sigmoid=self.use_sigmoid,
            return_scalezp=True,
        )
        self.orig_layer.weight.data.copy_(q_dq_weight)
        self.orig_layer.weight.grad = None  ##clear grad
        self.orig_layer.scale = scale.to("cpu")
        self.orig_layer.zp = zp.to("cpu")
        return self.orig_layer

    def forward(self, x):
        weight = self.orig_layer.weight
        with torch.no_grad():
            self.min_scale.clamp_(-1, 0)
            self.max_scale.clamp_(-1, 0)
        weight_q = quant_weight(
            weight,
            self.num_bits,
            self.group_size,
            self.scheme,
            self.value,
            self.min_scale,
            self.max_scale,
            use_sigmoid=self.use_sigmoid,
        )

        return F.linear(x, weight_q, self.orig_layer.bias)


def is_supported_layer(m, supported_types):
    supported = False
    for t in supported_types:
        if isinstance(m, t):
            supported = True
    return supported


def check_is_float(data_type, num_bits, group_size, scheme):
    if ("fp" in data_type or "float" in data_type) and num_bits >= 16:
        return True
    return False


def wrapper_block(block, enable_minmax_tuning, supported_types, use_sigmoid):
    for n, m in block.named_modules():
        if not is_supported_layer(m, supported_types):
            continue
        data_type = m.data_type
        num_bits = m.bits
        group_size = m.group_size
        scheme = m.scheme
        if check_is_float(data_type, num_bits, group_size, scheme):
            continue
        if isinstance(m, torch.nn.Linear):
            new_m = WrapperLinear(
                m, num_bits, group_size, scheme, enable_minmax_tuning=enable_minmax_tuning, use_sigmoid=use_sigmoid
            )
            set_module(block, n, new_m)
        elif isinstance(m, torch.nn.Conv1d):
            pass
        elif "Conv1D" in str(type(m)):
            from transformers.modeling_utils import Conv1D

            new_m = WrapperTransformerConv1d(
                m, num_bits, group_size, scheme, enable_minmax_tuning=enable_minmax_tuning, use_sigmoid=use_sigmoid
            )
            set_module(block, n, new_m)


@torch.no_grad()
def unwrapper_block(block, grads, min_scale_grads, max_scale_grads):
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear) or isinstance(m, WrapperTransformerConv1d):
            orig_layer = m.orig_layer
            grad = 0
            min_scale_grad = 0
            max_scale_grad = 0
            if isinstance(grads, dict):
                grad = grads[n]
            if isinstance(min_scale_grads, dict):
                min_scale_grad = min_scale_grads[n]
                min_scale_grad = torch.clamp(min_scale_grad, -1, 0)
            if isinstance(max_scale_grads, dict):
                max_scale_grad = max_scale_grads[n]
                max_scale_grad = torch.clamp(max_scale_grad, -1, 0)
            orig_layer = m.unwrapper(grad, min_scale_grad, max_scale_grad)
            set_module(block, n, orig_layer)


def collect_round_grad(block):
    grads = {}
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear) or isinstance(m, WrapperTransformerConv1d):
            grad = m.value.data
            grads[n] = copy.deepcopy(grad)
    return grads


def collect_minmax_grad(block):
    min_grads = {}
    max_grads = {}
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear) or isinstance(m, WrapperTransformerConv1d):
            min_grads[n] = copy.deepcopy(torch.clamp(m.min_scale.data, -1, 0))
            max_grads[n] = copy.deepcopy(torch.clamp(m.max_scale.data, -1, 0))
    return min_grads, max_grads


class OPTRoundQuantizer(object):
    def __init__(
            self,
            model,
            tokenizer=None,
            bits: int = 4,
            group_size: int = 128,
            scheme: str = "asym",
            weight_config: dict = {},
            enable_full_range: bool = False,  ##for symmetric, TODO support later
            bs: int = 8,
            amp: bool = True,
            device="cuda:0",
            optimizer=None,
            lr_scheduler=None,
            dataloader=None,  ## to support later
            default_dataset_name: str = "NeelNanda/pile-10k",
            dataset_split: str = "train",
            use_quant_input: bool = True,
            enable_minmax_tuning: bool = True,
            lr: float = 0.005,
            minmax_lr: float = 0.005,
            low_gpu_mem_usage: bool = True,
            iters: int = 200,
            seqlen: int = 2048,
            n_samples: int = 512,
            sampler: str = "rand",
            seed: int = 42,
            n_blocks: int = 1,
            gradient_accumulate_steps: int = 1,
            not_use_mse: bool = False,
            dynamic_max_gap: int = -1,
            data_type: str = "int",  ##only support data_type
            use_sigmoid=None,
            **kwargs
    ):
        """
        Args:
            model:
            data_type:
            bits:
            group_size:
            scheme:
            weight_config:
             weight_config={
                   'layer1':##layer_name
                   {
                       'data_type': 'int',
                       'bits': 4,
                       'group_size': 32,
                       'scheme': "sym", ## or asym
                   }
                   ...
               }

            optimizer:
            lr_scheduler:
            enable_full_range:
            **kwargs:

        Returns:
        """
        self.model = model
        self.model = self.model.to("cpu")
        self.amp = amp
        self.use_quant_input = use_quant_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.n_samples = n_samples
        self.n_blocks = n_blocks
        self.bits = bits
        self.group_size = group_size
        self.scheme = scheme
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.data_type = data_type
        self.supported_types = [torch.nn.Linear]
        try:
            import transformers
            self.supported_types.append(transformers.modeling_utils.Conv1D)
        except:
            pass
        self.weight_config = weight_config
        assert dataloader is not None or tokenizer is not None
        self.dataset_split = dataset_split
        self.seed = seed
        self.tokenizer = tokenizer
        self.seqlen = seqlen
        self.train_bs = bs
        self.n_blocks = n_blocks
        self.device = device
        self.amp_dtype = torch.float16
        if self.model.dtype != torch.float32:
            self.amp_dtype = self.model.dtype
        if self.device == "cpu":
            self.amp_dtype = torch.bfloat16
        if self.amp:
            logger.info(f"using {self.amp_dtype}")

        if dataloader is None:
            self.dataloader = self.get_default_dataloader(data_name=default_dataset_name)
        else:
            self.dataloader = dataloader
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.iters = iters
        self.sampler = sampler
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.not_use_mse = not_use_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.enable_full_range = enable_full_range
        assert self.enable_full_range is False, "only support enable_full_range=False currently"
        self.optimizer = optimizer
        self.scale_grad = True
        self.use_sigmoid = use_sigmoid

        if self.optimizer is None:
            self.use_sigmoid = False
            self.scale_grad = False
            from .sign_sgd import SGD

            self.optimizer = SGD

        elif isinstance(self.optimizer, str):
            self.optimizer = getattr(torch.optim, self.optimizer)
        else:
            self.optimizer = optimizer
        if self.use_sigmoid is None:
            self.use_sigmoid = False
        self.lr_scheduler = lr_scheduler
        self.set_layerwise_config(self.weight_config)

    def check_configs(self):
        assert isinstance(self.model, torch.nn.Module)
        assert self.bits > 0, "bits must be positive"
        assert self.group_size == -1 or self.group_size >= 1, "only supports positive group_size or -1(per channel)"
        assert self.bs > 0, "batch size must be positive"
        assert self.iters > 0, "iters must be positive"
        assert self.seqlen > 0, "seqlen must be positive"
        assert self.n_blocks > 0, "n_blocks must be positive"
        assert self.gradient_accumulate_steps > 0, "gradient accumulate step must be positive"

    def set_layerwise_config(self, weight_config):
        for n, m in self.model.named_modules():
            is_supported_type = False
            for supported_type in self.supported_types:
                if isinstance(m, supported_type):
                    is_supported_type = True
                    break
            if not is_supported_type:
                continue
            if n not in weight_config.keys():
                weight_config[n] = {}
                weight_config[n]["data_type"] = self.data_type
                weight_config[n]["bits"] = self.bits
                weight_config[n]["group_size"] = self.group_size
                weight_config[n]["scheme"] = self.scheme
            else:
                if "data_type" not in weight_config[n].keys():
                    weight_config[n]["data_type"] = self.data_type
                if "bits" not in weight_config[n].keys():
                    weight_config[n]["bits"] = self.bits
                if "group_size" not in weight_config[n].keys():
                    weight_config[n]["group_size"] = self.group_size
                if "scheme" not in weight_config[n].keys():
                    weight_config[n]["scheme"] = self.scheme
            m.data_type = weight_config[n]["data_type"]
            m.bits = weight_config[n]["bits"]
            m.group_size = weight_config[n]["group_size"]
            m.scheme = weight_config[n]["scheme"]

    def default_tokenize_function(self, examples):
        example = self.tokenizer(examples["text"], truncation=True, max_length=self.seqlen)
        return example

    def get_default_dataloader(self, data_name="NeelNanda/pile-10k"):
        from datasets import load_dataset
        from torch.utils.data import DataLoader

        @torch.no_grad()
        def collate_batch(batch):
            input_ids_new = []
            for text in batch:
                input_ids = text["input_ids"]
                if input_ids.shape[0] < seqlen:
                    continue
                input_ids = input_ids[:seqlen]
                input_ids_list = input_ids.tolist()
                if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
                    continue
                input_ids_new.append(input_ids)
            if len(input_ids_new) == 0:
                return None
            tmp = torch.vstack(input_ids_new)
            res = {"input_ids": tmp}
            return res

        seqlen = self.seqlen
        calib_dataset = load_dataset(data_name, split=self.dataset_split)
        calib_dataset = calib_dataset.shuffle(seed=self.seed)
        calib_dataset = calib_dataset.map(self.default_tokenize_function, batched=True)
        calib_dataset.set_format(type="torch", columns=["input_ids"])
        calib_dataloader = DataLoader(calib_dataset, batch_size=self.train_bs, shuffle=False, collate_fn=collate_batch)
        return calib_dataloader

    def get_batch_dim(self, input_others):
        dim = int(len(input_others["positional_inputs"]) > 0)
        return dim

    @torch.no_grad()
    def get_block_outputs(self, block, input_ids, input_others, bs, device, cache_device, batch_dim):
        output = []
        for i in range(0, self.n_samples, bs):
            indices = torch.arange(i, i + bs).to(torch.long)
            tmp_input_ids, tmp_input_others = sampling_inputs(input_ids, input_others, indices, self.seqlen)
            tmp_output = block_forward(block, tmp_input_ids, tmp_input_others, self.amp, device).to(cache_device)
            output.append(tmp_output)
        output = torch.cat(output, dim=batch_dim)
        torch.cuda.empty_cache()
        return output

    def loss_scale_and_backward(self, scaler, loss):
        if scaler is not None:
            scale_loss = scaler.scale(loss)
            scale_loss.backward()
            return scale_loss
        else:
            loss *= 1000
            loss.backward()
            return loss

    def step(self, scaler, optimizer, lr_schedule):  ##TODO signround does not need this
        if scaler is not None:
            scaler.step(optimizer)
            optimizer.zero_grad()
            lr_schedule.step()
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()
            lr_schedule.step()

    def is_supported_type(self, m):
        return hasattr(m, "orig_layer")

    def quant_block(self, block, input_ids, input_others, q_input=None, device=torch.device("cpu")):
        batch_dim = self.get_batch_dim(input_others)
        if not self.low_gpu_mem_usage and input_ids.device != device:
            # input_ids, input_others = move_to_device(input_ids, input_others, device)
            input_ids = move_input_to_device(input_ids, device)  ##move input data to GPU
            input_others = move_input_to_device(input_others, device)
        torch.cuda.empty_cache()

        cache_device = device
        if self.low_gpu_mem_usage:
            cache_device = "cpu"
        output = self.get_block_outputs(block, input_ids, input_others, self.train_bs, device, cache_device, batch_dim)

        if q_input is not None:
            input_ids = q_input.to(cache_device)

        wrapper_block(block, self.enable_minmax_tuning, self.supported_types, use_sigmoid=self.use_sigmoid)

        best_loss = torch.finfo(torch.float).max
        mse_loss = torch.nn.MSELoss()
        round_params = []
        minmax_params = []
        for n, m in block.named_modules():
            if self.is_supported_type(m):
                round_params.append(m.value)
                minmax_params.append(m.min_scale)
                minmax_params.append(m.max_scale)
        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": self.minmax_lr}], lr=self.lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=self.lr, weight_decay=0)

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters, verbose=False
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)
        if len(input_ids.shape) == 3:
            n_samples = input_ids.shape[0]
        else:
            n_samples = input_ids.shape[0] // self.seqlen
        if self.sampler != "rand":
            indices = torch.randperm(n_samples)[: self.train_bs]
        last_best_iter = 0
        scaler = None
        if self.amp and self.scale_grad:
            from torch.cuda.amp import GradScaler

            scaler = GradScaler(init_scale=1024, growth_interval=100000)

        for i in range(self.iters):
            if self.sampler == "rand":
                indices = torch.randperm(n_samples)[: self.train_bs]

            total_loss = 0
            for _ in range(self.gradient_accumulate_steps):
                current_input_ids, current_input_others = sampling_inputs(
                    input_ids, input_others, indices, seqlen=self.seqlen
                )
                if len(input_ids.shape) == 3:
                    if batch_dim == 0:
                        current_output = output[indices, :, :]
                    elif batch_dim == 1:
                        current_output = output[:, indices, :]
                    else:
                        current_output = output[:, :, indices]
                else:
                    current_output = output.view(n_samples, self.seqlen, -1)
                    current_output = current_output[indices, :, :]
                    current_output = current_output.reshape(-1, current_output.shape[-1])
                current_output = move_input_to_device(current_output, device)

                output_q = block_forward(block, current_input_ids, current_input_others, self.amp, device)
                if self.amp and device != torch.device("cpu"):
                    with autocast(device_type="cuda"):
                        loss = mse_loss(output_q, current_output)
                elif self.amp:
                    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                        loss = mse_loss(output_q, current_output)
                else:
                    loss = mse_loss(output_q, current_output)
                loss = loss / self.gradient_accumulate_steps
                total_loss += loss.item()
                scale_loss = self.loss_scale_and_backward(scaler, loss)

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_mse:
                    # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)
                    best_grad = collect_round_grad(block)
                    best_min_scale_grad, best_max_scale_grad = collect_minmax_grad(block)
                    last_best_iter = i
            if self.not_use_mse and i == self.iters - 1:
                best_grad = collect_round_grad(block)
                best_min_scale_grad, best_max_scale_grad = collect_minmax_grad(block)

            if not self.not_use_mse:
                if self.dynamic_max_gap > 0 and i - last_best_iter >= self.dynamic_max_gap:
                    break
            self.step(scaler, optimizer, lr_schedule)

        unwrapper_block(block, best_grad, best_min_scale_grad, best_max_scale_grad)
        if self.use_quant_input:
            q_outputs = self.get_block_outputs(
                block, input_ids, input_others, self.train_bs, device, cache_device, batch_dim
            )

            return q_outputs, output

        else:
            return None, output

    def q_dq_weight_round(
            self,
            model: torch.nn.Module,
            inputs,
            block_names,
            n_blocks=1,
            device=torch.device("cpu"),
    ):
        q_input = None
        torch.cuda.empty_cache()
        for n, m in model.named_parameters():
            m.requires_grad_(False)
        input_ids = inputs["input_ids"]
        inputs.pop("input_ids", None)
        input_others = inputs
        torch.cuda.empty_cache()
        for i in range(0, len(block_names), n_blocks):
            if n_blocks == 1:
                n = block_names[i]
                logger.info(n)
                m = get_module(model, n)
            else:
                names = block_names[i: i + n_blocks]
                logger.info(names)
                modules = [get_module(model, n) for n in names]
                m = WrapperMultiblock(modules)

            m = m.to(device)

            q_input, input_ids = self.quant_block(
                m,
                input_ids,
                input_others,
                q_input=q_input,
                device=device,
            )
            m.to("cpu")
            torch.cuda.empty_cache()

        del q_input
        del input_ids
        del input_others
        del inputs

        torch.cuda.empty_cache()

    def quantize(self):
        logger.info("cache input")
        block_names = get_block_names(self.model)
        if len(block_names) == 0:
            logger.warning("could not find blocks, exit with original model")
            return

        if self.amp:
            self.model = self.model.to(self.amp_dtype)
        if not self.low_gpu_mem_usage:  ##TODO automaticall detect
            self.model = self.model.to(self.device)

        save_input_actor = SaveInputs(self.model, self.dataloader, self.seqlen, block_names[0])
        inputs = save_input_actor.get_inputs(n_samples=self.n_samples)
        del save_input_actor
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
        self.q_dq_weight_round(
            self.model,
            inputs,
            block_names,
            n_blocks=self.n_blocks,
            device=self.device,
        )
        for n, m in self.model.named_modules():
            if n in self.weight_config.keys():
                if hasattr(m, "scale"):
                    self.weight_config[n]["scale"] = m.scale
                    self.weight_config[n]["zp"] = m.zp
                    delattr(m, "scale")
                    delattr(m, "zp")
        return self.model, self.weight_config
