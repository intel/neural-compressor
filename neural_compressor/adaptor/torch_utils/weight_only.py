#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
# Copyright (c) 2023 Intel Corporation
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

import math
from typing import OrderedDict
from .util import set_module
from ...utils import logger
from ...utils.utility import LazyImport

tqdm = LazyImport("tqdm")
torch = LazyImport("torch")


def qdq_weight_asym(weight, num_bits=4, quantile=1.0, return_int=False):
    """Quant and dequant tensor with asym schema.

    Args:
        weight:  input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.

    Returns:
        output: qdq weight
    """
    maxq = torch.tensor(2 ** num_bits - 1)
    zeros = torch.zeros(weight.shape[0], device=weight.device)
    wmin = torch.minimum(weight.min(1)[0], zeros)
    wmax = torch.maximum(weight.max(1)[0], zeros)
    wmin = wmin * quantile
    wmax = wmax * quantile
    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = (wmax - wmin) / maxq
    zp = torch.round(-wmin / scale)
    scale.unsqueeze_(dim=-1)
    zp.unsqueeze_(dim=-1)
    q = torch.clamp(torch.round(weight / scale) + zp, 0, maxq)
    if return_int:
        return q.type(torch.uint8), scale.type(torch.float), zp.type(torch.uint8)
    return scale * (q - zp)


def qdq_weight_sym(weight, num_bits=4, quantile=1.0, return_int=False, full_range=False):
    """Quant and dequant tensor with sym schema.

    Args:
        weight : input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).
                For example: 4 bit
                    scale = amax / 8 if full_range else amax / 7
                    If True, scale = -scale if abs(min)> abs(max) else scale
                    Defaults to False.

    Returns:
        output: qdq weight
    """
    # assert num_bits > 1, "symmetric scheme only supports num_bits > 1"
    maxq = torch.tensor(2 ** (num_bits - 1) - 1).to(weight.device)
    minq = torch.tensor(-2 ** (num_bits - 1)).to(weight.device)
    if num_bits == 1:
        maxq = torch.tensor(2 ** (num_bits - 1))
        minq = torch.tensor(2 ** (num_bits - 1) - 1)
    max_val = torch.max(weight, 1)[0]
    min_val = torch.min(weight, 1)[0]
    flip_flag = torch.abs(max_val) > torch.abs(min_val)
    wmax = torch.max(torch.abs(max_val), torch.abs(min_val))
    wmax = wmax * quantile
    tmp = (wmax == 0)
    wmax[tmp] = +1
    if full_range:
        # use -8, 8 to make sure amax is not changed after fake quant
        scale = wmax / (-minq)
        tmp = scale * flip_flag.int()
        scale -= 2*tmp # set negetive scale with flip_flag
    else:
        scale = wmax / maxq
    scale.unsqueeze_(dim=-1)
    q = torch.clamp(torch.round(weight / scale), minq, maxq)
    if return_int:
        return q.type(torch.int8), scale.type(torch.float), None
    return scale * q


def qdq_weight_actor(weight, num_bits, scheme, quantile=1.0, 
                     return_int=False, full_range=False):
    """Quant and dequant tensor per channel.

    Args:
        weight : input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        output: qdq weight
    """
    assert num_bits > 0, "num_bits should be larger than 0"
    if scheme == "sym":
        return qdq_weight_sym(weight, num_bits, quantile, return_int, full_range)
    else:
        return qdq_weight_asym(weight, num_bits, quantile, return_int)


def quant_weight(weight, num_bits=4, group_size=-1, scheme="asym", quantile=1.0, 
                 return_int=False, full_range=False):
    """Quant and dequant tensor with group size.

    Args:
        weight: input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to -1.
        scheme (str, optional): sym or asym. Defaults to "asym".
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        output: qdq weight.
    """
    if group_size == -1 or weight.shape[1] < group_size:
        return qdq_weight_actor(weight, num_bits, scheme=scheme, quantile=quantile, 
                                return_int=return_int, full_range=full_range)

    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        if return_int:
            weight, scale, zp = qdq_weight_actor(
                weight, num_bits, scheme=scheme, quantile=quantile, 
                return_int=True, full_range=full_range
            )
            weight = weight.reshape(orig_shape)
            scale = scale.reshape(orig_shape[0], -1)
            if zp is not None:
                zp = zp.reshape(orig_shape[0], -1)
            return weight, scale, zp
        else:
            weight = qdq_weight_actor(
                weight, num_bits, scheme=scheme, quantile=quantile, full_range=full_range
            )
            return weight.reshape(orig_shape)
    else:
        split_index = weight.shape[1] // group_size * group_size
        weight1 = weight[:, :split_index]
        weight1 = weight1.reshape(-1, group_size)
        if return_int:
            weight1, scale1, zp1 = qdq_weight_actor(
                weight1, num_bits, scheme=scheme, 
                quantile=quantile, return_int=True, full_range=full_range
            )
            scale1 = scale1.reshape(orig_shape[0], -1)
            if zp1 is not None:
                zp1 = zp1.reshape(orig_shape[0], -1)
        else:
            weight1 = qdq_weight_actor(
                weight1, num_bits, scheme=scheme, quantile=quantile, full_range=full_range
            )
        weight1 = weight1.reshape(orig_shape[0], split_index)
        weight2 = weight[:, split_index:]
        if return_int:
            weight2, scale2, zp2 = qdq_weight_actor(
                weight2, num_bits, scheme=scheme, 
                quantile=quantile, return_int=True, full_range=full_range
            )
            weight = torch.cat([weight1, weight2], dim=1)
            scale = torch.cat([scale1, scale2], dim=1)
            if zp2 is not None:
                zp = torch.cat([zp1, zp2], dim=1)
            else:
                zp = None
            return weight, scale, zp
        else:
            weight2 = qdq_weight_actor(
                weight2, num_bits, scheme=scheme, 
                quantile=quantile, full_range=full_range
            )
            weight = torch.cat([weight1, weight2], dim=1)
            return weight


def rtn_quantize(model, num_bits=4, group_size=32, scheme="asym", 
                 quantile=1.0, weight_config={}, return_int=False, 
                 sym_full_range=False, **kwargs):
    """Quant the model with round to nearst method.

    Args:
        model: torch module
        num_bits: num bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to 32.
        scheme (str, optional): sym or asym. Defaults to "asym".
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        weight_config (dict, optional): specific layer wise configirations. Defaults to {}.
            For example, 
                weight_config={
                    'fc2':
                        {
                            'bits': 4, 
                            'group_size': 32, 
                            'scheme': 'sym'
                            'gptq_perm': [1, 1, ...] # for gptq perm
                        }
                }
        return_int (bool, optional): Choose return fp32 or int32 model.
                                     Defaults to False.
        sym_full_range (bool, optional): Choose sym range whether use -2**(bits-1).
                                     Defaults to False.

    Returns:
        model: fake quantized torch module
    """
    assert isinstance(model, torch.nn.Module), "only support torch module"
    supported_layers = ['Linear']
    if return_int:
        compression_dtype = kwargs.get("compression_dtype", torch.int32)
        compression_dim = kwargs.get("compression_dim", 1)
        scale_dtype = kwargs.get("scale_dtype", torch.float32)
        device = kwargs.get("device", 'cpu')
    for name, m in model.named_modules():
        if m.__class__.__name__ not in supported_layers:
            continue
        if name in weight_config:  # pragma: no cover
            num_bits = weight_config[name]['bits']
            group_size = weight_config[name]['group_size']
            scheme = weight_config[name]['scheme']
            quantile = weight_config[name].get('quantile', 1.0)
        logger.debug(f"RTN quantized module:{name, m}")
        if scheme == 'sym':
            logger.debug(f"RTN quantization config: num_bits={num_bits}, group_size={group_size}, " + \
                        f"scheme={scheme}, quantile={quantile}, sym_full_range={sym_full_range}")
        else:
            logger.debug(f"RTN quantization config: num_bits={num_bits}, group_size={group_size}, " + \
                        f"scheme={scheme}, quantile={quantile}")
        if num_bits <= 0:
            logger.info(f"skip {name}")
            continue
        weight = m.weight
        if return_int:
            from .model_wrapper import WeightOnlyLinear
            int_weight, scale, zp = quant_weight(
                weight, num_bits, group_size, scheme, 
                quantile, return_int=True, full_range=sym_full_range
            )
            new_module = WeightOnlyLinear(
                m.in_features, m.out_features, num_bits, group_size,
                zp=zp is not None, bias=m.bias is not None, 
                compression_dtype=compression_dtype, 
                compression_dim=compression_dim, 
                scale_dtype=scale_dtype, 
                device=device,
            )
            new_module.pack(int_weight, scale, zp, m.bias)
            if name == '':
                return new_module
            else:
                set_module(model, name, new_module)
        else:
            q_weight = quant_weight(
                weight, num_bits, group_size, scheme, quantile, 
                full_range=sym_full_range
            )
            m.weight.data.copy_(q_weight)
    return model

def gptq_quantize(model, weight_config={}, dataloader=None, nsamples=128, use_max_length = True, device=None):
    """Run weight-only quantization with """
    # TODO: unify weight_config keys, add docstring, and support default config
    assert isinstance(model, torch.nn.Module), "only support torch module"
    from .gptq import GPTQuantizer
    gptq_quantizer = GPTQuantizer(model, weight_config, dataloader, nsamples, use_max_length, device)
    fp32_modified_model, gptq_config = gptq_quantizer.execute_quantization()
    logger.info("GPTQ quantizing done.")
    return fp32_modified_model, gptq_config

def get_module_input_output(model, module_hook_config={}, dataloader=None, iters=-1, 
                            calib_func=None):
    """A help function to get input and output tensor of modules in module_name_list.

    Args:
        model: torch model.
        module_hook_config (dict, optional): required module name for input/output. Defaults to {}.
            For example:
                module_hook_config = {
                    'fc1': ['output'],
                    'fc2': ['input', 'output']
                }
        dataloader: dataloader for model input.
        iters: iterations for inference.
        calib_func: a custom inference function to replace dataloader and iters.

    Returns:
        input_values, output_values: recorded input_values, output_values.
    """
    input_values, output_values = {}, {}
    def _save_input_output_hook(name):
        """
        A forward hook to save input and output values of a module
            param name: the module name
            return: A hook function
        """
        def save_input_hook(module, inputs):
            input = inputs[0]
            if name in input_values:
                try:
                    input_values[name] = torch.cat((input_values[name], input), 0)
                except Exception as e:
                    logger.error(e)
                    assert False, "Please unify the input shape for AWQ algorithm calibration."
            else:
                input_values[name] = input
        def save_output_hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if name in output_values:
                try:
                    output_values[name] = torch.cat((output_values[name], outputs), 0)
                except Exception as e:
                    logger.error(e)
                    assert False, "Please unify the input shape for AWQ algorithm calibration."
            else:
                output_values[name] = outputs
        return save_input_hook, save_output_hook

    hook_list = []
    for name, module in model.named_modules():
        if name in module_hook_config:
            save_input_hook, save_output_hook = _save_input_output_hook(name)
            require_list = module_hook_config[name]
            if 'input' in require_list:
                hook_list.append(
                    module.register_forward_pre_hook(save_input_hook))
            if 'output' in require_list:
                hook_list.append(
                    module.register_forward_hook(save_output_hook))
    if calib_func:
        calib_func(model)
    else:
        from .smooth_quant import model_forward
        model_forward(model, dataloader, iters, device='cpu')
    for h in hook_list:
        h.remove()
    return input_values, output_values


@torch.no_grad()
def _get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def _get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


def _update_input_with_scale(args, kwargs, scales):
    new_args, new_kwargs = args, kwargs
    for i, v in enumerate(args):
        if isinstance(v, torch.Tensor):
            try:
                new_args[i] = torch.div(v, scales.view(1, 1, -1))
            except:
                new_args[i] = v
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            try:
                new_kwargs[k] = torch.div(v, scales.view(1, 1, -1))
            except:
                new_kwargs[k] = v
    return new_args, new_kwargs


@torch.no_grad()
def awq_quantize(model, weight_config={}, absorb_dict={}, dataloader=None, n_samples=128, 
                 auto_scale=True, mse_range=True, calib_func=None, n_blocks=5, 
                 return_int=False, sym_full_range=False):
    """Quant the model with Activation-aware Weight quantization(AWQ) method.

    Args:
        model (torch.nn.Module): torch model.
        weight_config (dict, optional): contains all info required by AWQ. Defaults to {}.
            For example, 
                weight_config={
                    'fc2':
                        {
                            # 'absorb_layer': 'fc1', 
                            'bits': 4, 
                            'group_size': 32, 
                            'scheme': 'sym'
                        }
                }
        absorb_dict (dict, optional): contains all absorb info required by AWQ.. Defaults to {}.
            For example,
                absorb_dict = {
                    # 'absorb_layer': absorbed_layer
                    'fc1': ['fc2', 'fc3']
                } # in this case, fc2 and fc3 need to share the same scale.
        n_samples: calibration sample number.
        auto_scale (bool, optional): whether enable scale for salient weight. Defaults to True.
        mse_range (bool, optional):  whether enable clip for weight by checking mse. Defaults to True.
        calib_func: a custom inference function to replace dataloader and iters.
        n_blocks: split model into block number to avoid OOM.
        return_int (bool, optional): Choose return fp32 or int32 model.
                                     Defaults to False.
        sym_full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        model: fake quantized model
    """
    assert isinstance(model, torch.nn.Module), "only support torch module"
    # collect module names to record their input/output for loss calculation.
    module_for_loss_dict = OrderedDict()
    # get the upper module if absorbed modules have the same absorb module.
    for absorb, absorbed in absorb_dict.items():
        # used as input for absob module
        module_for_loss_dict[absorb] = set()
        if len(absorbed) > 1:
            split_dict = {}
            split_absorb = absorb.split('.')
            for ab in absorbed:
                split_dict[ab] = ab.split('.')
            for ab, sp_list in split_dict.items():
                for i in range(len(sp_list)):
                    if sp_list[i] != split_absorb[i]:
                        group_name = '.'.join(sp_list[:i+1])
                        module_for_loss_dict[absorb].add(group_name)
                        break
        else:
            module_for_loss_dict[absorb].add(absorbed[0])

    def calibration(module_name_list):
        if calib_func:
            input_values, output_values = get_module_input_output(
                model, 
                module_name_list, 
                calib_func=calib_func,
            )
        else:
            batch_size = dataloader.batch_size
            iters = int(math.ceil(n_samples / batch_size))
            if n_samples % batch_size != 0:
                logger.info("calibration samples increase from {} to {} due to batch_size is {}".format(
                    n_samples, iters*batch_size, batch_size,
                ))
            input_values, output_values = get_module_input_output(
                model, 
                module_name_list, 
                dataloader=dataloader, 
                iters=iters,
            )
        return input_values, output_values

    layer_args = {}
    layer_kwargs = {}
    # to fetch kwargs which torch hook cannot handle
    class Catcher(torch.nn.Module):
        def __init__(self, module, name):
            super().__init__()
            self.module = module
            self.module_name = name

        def forward(self, *args, **kwargs):
            if self.module_name not in layer_args:
                layer_args[self.module_name] = list(args)
                layer_kwargs[self.module_name] = kwargs
            else:
                # to concat different batches
                for i, v in enumerate(args):
                    if isinstance(v, torch.Tensor):
                        layer_args[self.module_name][i] = \
                                    torch.cat([layer_args[self.module_name][i], v], dim=0)
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        layer_kwargs[self.module_name][k] = \
                                    torch.cat([layer_kwargs[self.module_name][k], v], dim=0)
            return self.module.forward(*args, **kwargs)

    if auto_scale or mse_range:
        from .util import fetch_module, set_module
        block_num = n_blocks
        module_num = math.ceil(len(absorb_dict) / block_num)
        logger.info(f"AWQ search splits the model into {block_num} blocks to avoid OOM, " +\
                    f"each block contains {module_num} modules")
        for idx, (absorb, absorbed) in enumerate(absorb_dict.items()):
            # Split module_name_list to avoid OOM when recording output tensors.
            if idx % module_num == 0:
                layer_args = {}
                layer_kwargs = {}
                part_module_hook_config = {}
                logger.info(f"AWQ search calibration round {idx//module_num + 1}")
                for id, name in enumerate(module_for_loss_dict):
                    if idx <= id < (idx + module_num):
                        part_module_hook_config[name] = ['output'] # fetch input tensor
                        for i in module_for_loss_dict[name]:
                            # use Catcher to record input args and kwargs
                            tmp_module = fetch_module(model, i)
                            new_module = Catcher(tmp_module, i)
                            set_module(model, i, new_module)
                            part_module_hook_config[i] = ['output'] # fetch output tensor
                input_values, output_values = calibration(part_module_hook_config)
                # recover Catcher
                for id, name in enumerate(module_for_loss_dict):
                    if idx <= id < (idx + module_num):
                        for i in module_for_loss_dict[name]:
                            # use Catcher to record input args and kwargs
                            tmp_module = fetch_module(model, i)
                            set_module(model, i, tmp_module.module)

            logger.info(f"Processing module: {absorb}:{absorbed}")
            weight = torch.cat([fetch_module(model, _m).weight for _m in absorbed], dim=0)
            w_max = _get_weight_scale(
                weight, q_group_size=weight_config[absorbed[0]]['group_size'])
            del weight
            x_max = _get_act_scale(output_values[absorb])
            absorbed_modules = {_m: fetch_module(model, _m) for _m in absorbed}
            org_stat = {_m: module.state_dict() for _m, module in absorbed_modules.items()}
            org_out = {}
            for loss_module_name in module_for_loss_dict[absorb]:
                out = output_values[loss_module_name]
                if isinstance(out, tuple):
                    out = out[0]
                org_out[loss_module_name] = out
            for loss_module_name in module_for_loss_dict[absorb]:
                blockes = {_m: fetch_module(model, _m) for _m in module_for_loss_dict[absorb]}

            if auto_scale:
                logger.info("Searching best scales with AWQ algorithm")
                best_error = float('inf')
                best_scales = None
                best_scale_alpha = None
                n_grid = 20
                history = []

                for ratio in range(n_grid):
                    ratio = ratio * 1 / n_grid
                    scales = (x_max.pow(ratio) / w_max.pow(1-ratio)
                            ).clamp(min=1e-4).view(-1)
                    scales = scales / (scales.max() * scales.min()).sqrt()
                    for name, module in absorbed_modules.items():
                        module.weight.data = module.weight.data.mul(scales.view(1, -1))
                        module.weight.data = quant_weight(
                            module.weight.data,
                            num_bits=weight_config[name]['bits'], 
                            group_size=weight_config[name]['group_size'], 
                            scheme=weight_config[name]['scheme'],
                        ) / scales.view(1, -1)

                    loss = 0
                    for name, block in blockes.items():
                        out = block(*layer_args[name], **layer_kwargs[name])
                        if isinstance(out, tuple):
                            out = out[0]
                        loss += (org_out[name] - out).float().pow(2).mean().item()  # float prevents overflow
                    history.append(loss)
                    is_best = loss < best_error
                    if is_best:
                        best_error = loss
                        best_scales = scales
                        best_scale_alpha = ratio
                    for name, module in absorbed_modules.items():
                        module.load_state_dict(org_stat[name])

                logger.debug("The loss history of different scale:{}".format(history))
                logger.debug("The best alpha for scale: {}:{}".format(absorb, best_scale_alpha))
                assert best_scales is not None, "Loss is infinity! Cannot find the correct scale."
                best_scales = best_scales.view(-1)
                assert torch.isnan(best_scales).sum() == 0, best_scales
                scales = best_scales.detach()
                # update absorb model
                absorb_module = fetch_module(model, absorb)
                if len(absorb_module.weight.shape) == 1:
                    absorb_module.weight.div_(scales)  # for LayerNorm
                else:
                    absorb_module.weight.div_(scales.view(-1, 1))
                if absorb_module.bias is not None:
                    absorb_module.bias.div_(scales.view(-1))
                # update absorbed model
                for name in absorbed:
                    absorbed_module = fetch_module(model, name)
                    absorbed_module.weight.mul_(scales.view(1, -1))

            if mse_range:
                logger.info("Searching the best clip range with AWQ algorithm")
                best_error = float('inf')
                best_clip_ratio = None
                n_grid = 100
                max_shrink = 0.1
                history = []
                org_stat = {_m: module.state_dict() for _m, module in absorbed_modules.items()}

                for name, module in absorbed_modules.items():
                    for i_s in range(int(max_shrink * n_grid)):
                        ratio = (1 - i_s / n_grid) # 1, 0.95-0.55
                        module.weight.data = quant_weight(
                            module.weight.data,
                            num_bits=weight_config[name]['bits'], 
                            group_size=weight_config[name]['group_size'], 
                            scheme=weight_config[name]['scheme'],
                            quantile=ratio,
                        )

                        loss = 0
                        for n, block in blockes.items():
                            if n in name:
                                # preprocess input with existing scale
                                if auto_scale:
                                    new_args, new_kwargs = _update_input_with_scale(
                                                    layer_args[n], layer_kwargs[n], scales)
                                else:
                                    new_args, new_kwargs = layer_args[n], layer_kwargs[n]
                                out = block(*new_args, **new_kwargs)
                                if isinstance(out, tuple):
                                    out = out[0]
                                loss += (org_out[n] - out).float().pow(2).mean().item()  # float prevents overflow
                        history.append(loss)
                        is_best = loss < best_error
                        if is_best:
                            best_error = loss
                            best_clip_ratio = ratio
                        module.load_state_dict(org_stat[name])

                    logger.debug("The loss history of different clip range:{}".format(history))
                    weight_config[name]['quantile'] = best_clip_ratio
                    logger.debug("The best clip ratio for {}:{}".format(name, best_clip_ratio))

    # apply quantization and clip
    logger.info("Quantizing the AWQ optimized fp32 model")
    model = rtn_quantize(
        model, 
        num_bits=-1, 
        weight_config=weight_config, 
        return_int=return_int,
        sym_full_range=sym_full_range,
    )
    logger.info("AWQ quantization is done.")
    return model

def teq_quantize(model, weight_config={}, absorb_to_layer={}, extra_config={},
        dataloader= None, calib_func=None, example_inputs=None):
    """Run weight-only quantization with """
    assert isinstance(model, torch.nn.Module), "only support torch module"
    logger.info("TEQ quantizing start.")
    if example_inputs is None:
        if dataloader is None: # pragma: no cover
            assert False, "Please provide dataloader or example_inputs for TEQ algorithm."
        try:
            for idx, (input, label) in enumerate(dataloader):
                example_inputs = input
                break
        except: # pragma: no cover
            for idx, input in enumerate(dataloader):
                example_inputs = input
                break

    from .teq import TEQuantizer
    teq_quantizer = TEQuantizer(model, weight_config, absorb_to_layer, extra_config, example_inputs)

    # 1. wrapper tuning scale to model
    teq_quantizer.add_tuning_scale()

    # 2. tuning
    # custom train function, there calls calib_func
    if calib_func: # pragma: no cover
        calib_func(teq_quantizer.model)
    else:
        if dataloader is None: # pragma: no cover
            assert False, "Please provide dataloader to train."
        teq_quantizer.train(dataloader)

    # 3. apply scale to model
    teq_quantizer.transform()

    # 4. get quantized model
    teq_quantizer.quantize()

    #quantization_data = gptq_quantizer.execute_quantization()
    logger.info("TEQ quantizing done.")
    return teq_quantizer.model


def quant_weight_w_scale(weight, scale, zp, group_size=-1):
    """Quant and dequant tensor with group size.

    Args:
        weight: input weight
        scale: scale
        zp: zero point
        group_size (int, optional): how many elements share one scale/zp. Defaults to -1.

    Returns:
        output: int weight.
    """
    device = weight.device
    scale = scale.to(device)
    if zp is not None:
        zp = zp.to(device)
    if group_size == -1:
        return torch.round(weight/scale) if zp is None else torch.round(weight/scale + zp)
    int_weight = torch.zeros(weight.shape).to(device)
    leng = weight.shape[1] // group_size
    tail_flag = False if weight.shape[1] % group_size == 0 else True
    for i in range(leng):
        int_weight_tmp = weight[:, i*group_size: (i+1)*group_size] / scale[:, i].unsqueeze(1)
        if zp is not None:
            int_weight_tmp += zp[:, i].unsqueeze(1)
        int_weight[:, i*group_size: (i+1)*group_size] = torch.round(int_weight_tmp)
    if tail_flag:
        int_weight_tmp = weight[:, leng*group_size:] / scale[:, -1].unsqueeze(1)
        if zp is not None:
            int_weight_tmp += zp[:, -1].unsqueeze(1)
        int_weight[:, leng*group_size:] = torch.round(int_weight_tmp)
    return int_weight
