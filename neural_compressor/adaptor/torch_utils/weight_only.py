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

from copy import deepcopy
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
    if num_bits == 1:  # pragma: no cover
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
    if num_bits <= 0:
        return weight
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


def search_clip(m, num_bits, group_size, scheme, sym_full_range):
    """Search best clip range of each linears in current block.

    Args:
        m (torch.nn.Module): torch module.
        num_bits (int, optional): num bits.
        group_size (int, optional): how many elements share one scale/zp.
        scheme (str, optional): sym or asym.
        sym_full_range (bool, optional): Choose sym range whether use -2**(bits-1).
    
    Returns:
        best_clip_ratio (float): best percentile of clip
    
    """
    org_weight = m.weight.data
    logger.info("Searching the best clip range with RTN algorithm")
    best_error = float('inf')
    best_clip_ratio = None
    n_grid = 200
    max_shrink = 0.2
    history = []
    for i_s in range(int(max_shrink * n_grid)):
        ratio = (1 - i_s / n_grid) # 1, 0.805-1.0
        cur_weight = quant_weight(
            m.weight.data,
            num_bits=num_bits, 
            group_size=group_size, 
            scheme=scheme,
            full_range=sym_full_range,
            quantile=ratio,
        )
        loss = (org_weight - cur_weight).float().pow(2).mean().item()
        history.append(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_clip_ratio = ratio
    logger.debug("The loss history of different clip range:{}".format(history))
    logger.debug("The best clip ratio is {}".format(best_clip_ratio))
    return best_clip_ratio

def rtn_quantize(model, num_bits=4, group_size=32, scheme="asym", 
                 quantile=1.0, weight_config={}, return_int=False, 
                 sym_full_range=False, mse_range=False, **kwargs):
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
        mse_range (bool, optional):  Whether search clip range.
                                     Defaults to True.

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
            logger.info(f"Skip {name}")
            continue
        weight = m.weight
        if mse_range:
            quantile = search_clip(m, num_bits, group_size, scheme, sym_full_range)
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

def gptq_quantize(model, weight_config={}, dataloader=None, device=None):
    """Run weight-only quantization with """
    # TODO: unify weight_config keys, add docstring, and support default config
    assert isinstance(model, torch.nn.Module), "only support torch module"
    from .gptq import GPTQuantizer
    gptq_quantizer = GPTQuantizer(model, weight_config, dataloader, device)
    fp32_modified_model, gptq_config = gptq_quantizer.execute_quantization()
    logger.info("GPTQ quantizing done.")
    return fp32_modified_model, gptq_config


@torch.no_grad()
def awq_quantize(model, bits=4,  group_size=32, scheme='asym', weight_config={}, 
                 example_inputs=None, dataloader=None, n_samples=128, calib_func=None,
                 auto_scale=True, mse_range=True, folding=False, return_int=False, 
                 sym_full_range=False):
    """Quant the model with Activation-aware Weight quantization(AWQ) method.

    Args:
        model (torch.nn.Module): torch model.
        example_inputs: example_inputs.
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
                    'fc1': ['fc1', 'fc2', 'fc3']
                } # in this case, fc2 and fc3 need to share the same scale. fc1 is self absorbed.
                # self absorb module will replace with MulLinear, which contains torch.mul and module.
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
    from .awq import ActAwareWeightQuant
    assert isinstance(model, torch.nn.Module), "only support torch module"
    awq = ActAwareWeightQuant(
        model, 
        example_inputs=example_inputs, 
        calib_func=calib_func, 
        dataloader=dataloader, 
        n_samples=n_samples,
        bits=bits,  
        group_size=group_size, 
        scheme=scheme, 
        sym_full_range=sym_full_range, 
        weight_config=weight_config
    )
    qdq_model = awq.quantize(
        auto_scale=auto_scale,
        mse_range=mse_range,
        folding=folding,
        return_int=return_int,
    )
    return qdq_model


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
