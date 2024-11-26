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
from copy import deepcopy
from typing import Optional, OrderedDict, Union

from ...utils import logger
from ...utils.utility import LazyImport
from .util import set_module

tqdm = LazyImport("tqdm")
torch = LazyImport("torch")


NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
FP4_BNB = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -0.0625, 0, 0.0625, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.0625, 0, 0.0625, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# the order is the same as float list, bit value range is [-7, 7]
# 1111 = -1, 1110 = -2, 1101= -3, ...

NF4_BIT = [7, 1, 2, 3, 4, 5, 6, 0, -8, -7, -6, -5, -4, -3, -2, -1]
FP4_BNB_BIT = [-5, -6, -3, -4, -1, -2, -7, 0, 1, 6, 7, 4, 5, 2, 3]
FP4_E2M1_BIT = [-1, -2, -3, -4, -5, -6, -7, 0, 1, 2, 3, 4, 5, 6, 7]

FLOAT_MAPPING = {"nf4": NF4, "fp4": FP4_BNB, "fp4_e2m1_bnb": FP4_BNB, "fp4_e2m1": FP4_E2M1}
INT_MAPPING = {"nf4": NF4_BIT, "fp4": FP4_BNB_BIT, "fp4_e2m1_bnb": FP4_BNB_BIT, "fp4_e2m1": FP4_E2M1_BIT}


def quantize_4bit(tensor, quantile=1.0, data_type="nf4", return_int=False):
    """Quantize tensor to NF4/FP4 data type.

    Args:
        tensor: input tensor
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        data_type (str, optional): data type. Defaults to 'nf4'.
        return_int (bool, optional): whether return int data. Defaults to False.

    Returns:
        q_tensor: fake quantized tensor
    """
    assert data_type in FLOAT_MAPPING, "unexpected data type."
    allow_data = FLOAT_MAPPING[data_type]
    allow_data_bit = INT_MAPPING[data_type]
    # get scale and update tensor
    scale = tensor.abs().max(1)[0] * quantile / max(allow_data)
    scale.unsqueeze_(dim=-1)
    tensor.div_(scale)
    mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
    q_tensor = torch.zeros_like(tensor)
    for i in range(len(allow_data)):
        data = allow_data_bit[i] if return_int else allow_data[i]
        if i == 0:
            q_tensor += torch.where(tensor <= mid_data[i], data, 0)
        elif i == len(allow_data) - 1:
            q_tensor += torch.where(tensor > mid_data[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_data[i - 1] < tensor) & (tensor <= mid_data[i]), data, 0)
    tensor.copy_(q_tensor)
    if return_int:
        return tensor, scale, None
    return tensor.mul_(scale)


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
    maxq = torch.tensor(2**num_bits - 1)
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
    weight.div_(scale)
    weight.round_()
    weight.add_(zp)
    weight.clamp_(0, maxq)
    if return_int:
        return weight, scale, zp
    weight.sub_(zp)
    return weight.mul_(scale)


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
    minq = torch.tensor(-(2 ** (num_bits - 1))).to(weight.device)
    if num_bits == 1:  # pragma: no cover
        maxq = torch.tensor(2 ** (num_bits - 1))
        minq = torch.tensor(2 ** (num_bits - 1) - 1)
    max_val = torch.max(weight, 1)[0]
    min_val = torch.min(weight, 1)[0]
    flip_flag = torch.abs(max_val) > torch.abs(min_val)
    wmax = torch.max(torch.abs(max_val), torch.abs(min_val))
    wmax = wmax * quantile
    tmp = wmax == 0
    wmax[tmp] = +1
    if full_range:
        # use -8, 8 to make sure amax is not changed after fake quant
        scale = wmax / (-minq)
        tmp = scale * flip_flag.int()
        scale -= 2 * tmp  # set negetive scale with flip_flag
    else:
        scale = wmax / maxq
    scale.unsqueeze_(dim=-1)
    weight.div_(scale)
    weight.round_()
    weight.clamp_(minq, maxq)
    if return_int:
        return weight, scale, None
    return weight.mul_(scale)


def qdq_weight_actor(weight, num_bits, scheme, quantile=1.0, data_type="int", return_int=False, full_range=False):
    """Quant and dequant tensor per channel.

    Args:
        weight : input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        data_type (str, optional): select from int, nf4, fp4. Defaults to int.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        output: qdq weight
    """
    assert num_bits > 0, "num_bits should be larger than 0"
    if "int" not in data_type and num_bits == 4:
        return quantize_4bit(weight, quantile=quantile, data_type=data_type, return_int=return_int)
    if scheme == "sym":
        return qdq_weight_sym(weight, num_bits, quantile, return_int, full_range)
    else:
        return qdq_weight_asym(weight, num_bits, quantile, return_int)


def quant_weight(
    weight, num_bits=4, group_size=-1, scheme="asym", quantile=1.0, data_type="int", return_int=False, full_range=False
):
    """Quant and dequant tensor with group size. It is an in-place op.

    Args:
        weight: input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to -1.
        scheme (str, optional): sym or asym. Defaults to "asym".
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        data_type (str, optional): select from int, nf4, fp4. Defaults to int.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        output: qdq weight.
    """
    if num_bits <= 0:  # pragma: no cover
        return weight
    if group_size == -1 or weight.shape[1] < group_size:
        return qdq_weight_actor(
            weight,
            num_bits,
            scheme=scheme,
            quantile=quantile,
            return_int=return_int,
            full_range=full_range,
            data_type=data_type,
        )

    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        orig_weight = weight
        weight = weight.reshape(-1, group_size)
        if return_int:
            weight, scale, zp = qdq_weight_actor(
                weight,
                num_bits,
                scheme=scheme,
                quantile=quantile,
                return_int=True,
                full_range=full_range,
                data_type=data_type,
            )
            weight = weight.reshape(orig_shape)
            orig_weight.copy_(weight)
            scale = scale.reshape(orig_shape[0], -1)
            if zp is not None:
                zp = zp.reshape(orig_shape[0], -1)
            return orig_weight, scale, zp
        else:
            qdq_weight_actor(
                weight, num_bits, scheme=scheme, data_type=data_type, quantile=quantile, full_range=full_range
            )
            weight = weight.reshape(orig_shape)
            orig_weight.copy_(weight)
            return orig_weight
    else:
        split_index = weight.shape[1] // group_size * group_size
        orig_weight = weight
        weight1 = weight[:, :split_index]
        weight1 = weight1.reshape(-1, group_size)
        if return_int:
            weight1, scale1, zp1 = qdq_weight_actor(
                weight1,
                num_bits,
                scheme=scheme,
                data_type=data_type,
                quantile=quantile,
                return_int=True,
                full_range=full_range,
            )
            scale1 = scale1.reshape(orig_shape[0], -1)
            if zp1 is not None:
                zp1 = zp1.reshape(orig_shape[0], -1)
        else:
            qdq_weight_actor(
                weight1, num_bits, scheme=scheme, quantile=quantile, data_type=data_type, full_range=full_range
            )
        weight1 = weight1.reshape(orig_shape[0], split_index)
        weight2 = weight[:, split_index:]
        if return_int:
            weight2, scale2, zp2 = qdq_weight_actor(
                weight2,
                num_bits,
                scheme=scheme,
                data_type=data_type,
                quantile=quantile,
                return_int=True,
                full_range=full_range,
            )
            orig_weight.copy_(torch.cat([weight1, weight2], dim=1))
            scale = torch.cat([scale1, scale2], dim=1)
            if zp2 is not None:
                zp = torch.cat([zp1, zp2], dim=1)
            else:
                zp = None
            return orig_weight, scale, zp
        else:
            weight2 = qdq_weight_actor(
                weight2, num_bits, scheme=scheme, data_type=data_type, quantile=quantile, full_range=full_range
            )
            orig_weight.copy_(torch.cat([weight1, weight2], dim=1))
            return orig_weight


def search_clip(m, num_bits=4, group_size=32, scheme="asym", data_type="int", enable_full_range=False):
    """Search best clip range of each linears in current block.

    Args:
        m (torch.nn.Module): torch module.
        num_bits (int, optional): num bits.
        group_size (int, optional): how many elements share one scale/zp.
        scheme (str, optional): sym or asym.
        data_type (str, optional): select from int, nf4, fp4. Defaults to int.
        enable_full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        best_clip_ratio (float): best percentile of clip
    """
    org_weight = m.weight.data.clone()
    logger.info("Searching the best clip range with RTN algorithm")
    best_error = float("inf")
    best_clip_ratio = None
    n_grid = 200
    max_shrink = 0.2
    history = []
    for i_s in range(int(max_shrink * n_grid)):
        ratio = 1 - i_s / n_grid  # 1, 0.805-1.0
        quant_weight(
            m.weight.data,  # in-place mode
            num_bits=num_bits,
            group_size=group_size,
            scheme=scheme,
            data_type=data_type,
            full_range=enable_full_range,
            quantile=ratio,
        )
        loss = (org_weight - m.weight.data).float().pow(2).mean().item()
        m.weight.data.copy_(org_weight)
        history.append(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_clip_ratio = ratio
    logger.debug("The loss history of different clip range:{}".format(history))
    logger.debug("The best clip ratio is {}".format(best_clip_ratio))
    return best_clip_ratio


def rtn_quantize(
    model,
    num_bits=4,
    group_size=32,
    scheme="asym",
    quantile=1.0,
    weight_config={},
    return_int=False,
    data_type="int",
    enable_full_range=False,
    enable_mse_search=False,
    group_dim=1,
    **kwargs,
):
    """Quant the model with round to nearst method.

    Args:
        model: torch module
        num_bits: num bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to 32.
        scheme (str, optional): sym or asym. Defaults to "asym".
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        data_type (str, optional): select from int, nf4, fp4. Defaults to int.
        weight_config (dict, optional): specific layer wise configurations. Defaults to {}.
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
        enable_full_range (bool, optional): Choose sym range whether use -2**(bits-1).
                                     Defaults to False.
        enable_mse_search (bool, optional):  Whether search clip range.
                                     Defaults to True.
        group_dim (int, optional):   0 means splitting output channel,
                                     1 means splitting input channel. Defaults to 1.

    Returns:
        model: fake quantized torch module
    """
    assert isinstance(model, torch.nn.Module), "only support torch module"
    supported_layers = (torch.nn.Linear,)
    if return_int:
        compression_dtype = kwargs.get("compression_dtype", torch.int32)
        compression_dim = kwargs.get("compression_dim", 1)
        scale_dtype = kwargs.get("scale_dtype", torch.float32)
        device = kwargs.get("device", "cpu")
        use_optimum_format = kwargs.get("use_optimum_format", True)
    with torch.no_grad():
        for name, m in model.named_modules():
            if not isinstance(m, supported_layers):
                continue
            orig_dtype = next(m.parameters()).dtype
            if orig_dtype != torch.float:
                m = m.float()
            if name in weight_config:  # pragma: no cover
                num_bits = weight_config[name]["bits"]
                group_size = weight_config[name]["group_size"]
                scheme = weight_config[name]["scheme"]
                quantile = weight_config[name].get("quantile", 1.0)
            logger.debug(f"RTN quantized module:{name, m}")
            log_msg = (
                f"RTN quantization config: num_bits={num_bits}, group_size={group_size}, "
                + f"scheme={scheme}, quantile={quantile}"
            )
            if data_type != "int":
                log_msg += f", dtype={data_type}"
            elif scheme == "sym":  # nf4/fp4 is always [-7,7]
                log_msg += f", enable_full_range={enable_full_range}"
            logger.debug(log_msg)
            if num_bits <= 0:
                logger.info(f"Skip {name}")
                continue
            # contiguous is not an in-place op and returns Tensor instead of Parameter, so set it back to m.weight.data.
            # transpose should be executed on Parameter level because Param.data.t_() is not an in-place op.
            # Parameter.T is an in-place op while Tensor.T is not.
            m.weight.data = m.weight.t_().data.contiguous() if group_dim == 0 else m.weight.data
            if enable_mse_search:
                quantile = search_clip(m, num_bits, group_size, scheme, data_type, enable_full_range)
            if return_int:
                from .model_wrapper import WeightOnlyLinear

                _, scale, zp = quant_weight(
                    m.weight.data,
                    num_bits,
                    group_size,
                    scheme,
                    quantile,
                    data_type=data_type,
                    return_int=True,
                    full_range=enable_full_range,
                )
                if group_dim == 0:
                    m.weight.t_()
                scale = scale.t_().contiguous() if group_dim == 0 else scale
                zp = zp.t_().contiguous() if group_dim == 0 and zp is not None else zp
                new_module = WeightOnlyLinear(
                    m.in_features,
                    m.out_features,
                    num_bits,
                    group_size,
                    dtype=data_type,
                    zp=zp is not None,
                    bias=m.bias is not None,
                    compression_dtype=compression_dtype,
                    compression_dim=compression_dim,
                    scale_dtype=scale_dtype,
                    device=device,
                    use_optimum_format=use_optimum_format,
                )
                new_module.pack(m.weight.data, scale, zp, m.bias)
                if name == "":
                    return new_module
                else:
                    set_module(model, name, new_module)
            else:
                quant_weight(
                    m.weight.data,
                    num_bits,
                    group_size,
                    scheme,
                    quantile,
                    data_type=data_type,
                    full_range=enable_full_range,
                )
                if group_dim == 0:
                    m.weight.t_()
            if orig_dtype != torch.float:
                m = m.to(orig_dtype)
    return model


def gptq_quantize(
    model,
    weight_config={},
    dataloader=None,
    nsamples=128,
    use_max_length=True,
    pad_max_length=2048,
    device=None,
    layer_wise=False,
    model_path=None,
):
    """Run weight-only quantization with."""
    # TODO: unify weight_config keys, add docstring, and support default config
    assert isinstance(model, torch.nn.Module), "only support torch module"
    if layer_wise:
        assert model_path is not None, "model_path should not be None when use layer_wise mode"
    from .gptq import GPTQuantizer

    gptq_quantizer = GPTQuantizer(
        model, weight_config, dataloader, nsamples, use_max_length, pad_max_length, device, layer_wise=layer_wise
    )
    fp32_modified_model, gptq_config = gptq_quantizer.execute_quantization(model_path=model_path)
    logger.info("GPTQ quantizing done.")
    return fp32_modified_model, gptq_config


@torch.no_grad()
def awq_quantize(
    model,
    bits=4,
    group_size=32,
    scheme="asym",
    weight_config={},
    example_inputs=None,
    dataloader=None,
    n_samples=128,
    calib_func=None,
    enable_auto_scale=True,
    enable_mse_search=True,
    folding=False,
    return_int=False,
    enable_full_range=False,
    data_type="int",
):
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
        enable_auto_scale (bool, optional): whether enable scale for salient weight. Defaults to True.
        enable_mse_search (bool, optional):  whether enable clip for weight by checking mse. Defaults to True.
        calib_func: a custom inference function to replace dataloader and iters.
        n_blocks: split model into block number to avoid OOM.
        return_int (bool, optional): Choose return fp32 or int32 model.
                                     Defaults to False.
        enable_full_range (bool, optional): Choose sym range whether use -2**(bits-1).

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
        enable_full_range=enable_full_range,
        weight_config=weight_config,
        data_type=data_type,
    )
    qdq_model = awq.quantize(
        enable_auto_scale=enable_auto_scale,
        enable_mse_search=enable_mse_search,
        folding=folding,
        return_int=return_int,
    )
    return qdq_model


def teq_quantize(
    model, weight_config={}, absorb_to_layer={}, extra_config={}, dataloader=None, calib_func=None, example_inputs=None
):
    """Run weight-only quantization with."""
    assert isinstance(model, torch.nn.Module), "only support torch module"
    logger.info("TEQ quantizing start.")
    if example_inputs is None:
        if dataloader is None:  # pragma: no cover
            assert False, "Please provide dataloader or example_inputs for TEQ algorithm."
        try:
            for idx, (input, label) in enumerate(dataloader):
                example_inputs = input
                break
        except:  # pragma: no cover
            for idx, input in enumerate(dataloader):
                example_inputs = input
                break

    from .teq import TEQuantizer

    teq_quantizer = TEQuantizer(model, weight_config, absorb_to_layer, extra_config, example_inputs)

    # 1. wrapper tuning scale to model
    teq_quantizer.add_tuning_scale()

    # 2. tuning
    # custom train function, there calls calib_func
    if calib_func:  # pragma: no cover
        calib_func(teq_quantizer.model)
    else:
        if dataloader is None:  # pragma: no cover
            assert False, "Please provide dataloader to train."
        teq_quantizer.train(dataloader)

    # 3. apply scale to model
    teq_quantizer.transform()

    # 4. get quantized model
    teq_quantizer.quantize()

    # quantization_data = gptq_quantizer.execute_quantization()
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
        return weight.div_(scale).round_() if zp is None else weight.div_(scale).add_(zp).round_()
    int_weight = torch.zeros(weight.shape).to(device)
    leng = weight.shape[1] // group_size
    tail_flag = False if weight.shape[1] % group_size == 0 else True
    for i in range(leng):
        int_weight_tmp = weight[:, i * group_size : (i + 1) * group_size].div_(scale[:, i].unsqueeze(1))
        if zp is not None:
            int_weight_tmp.add_(zp[:, i].unsqueeze(1))
        int_weight[:, i * group_size : (i + 1) * group_size].copy_(int_weight_tmp.round_())
    if tail_flag:
        int_weight_tmp = weight[:, leng * group_size :].div_(scale[:, -1].unsqueeze(1))
        if zp is not None:
            int_weight_tmp.add_(zp[:, -1].unsqueeze(1))
        int_weight[:, leng * group_size :].copy_(int_weight_tmp.round_())
    return int_weight


def autoround_quantize(
    model,
    tokenizer=None,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    weight_config: dict = {},
    enable_full_range: bool = False,  ##for symmetric, TODO support later
    batch_size: int = 8,
    amp: bool = True,
    device=None,
    lr_scheduler=None,
    dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
    enable_quanted_input: bool = True,
    enable_minmax_tuning: bool = True,
    lr: float = None,
    minmax_lr: float = None,
    low_gpu_mem_usage: bool = False,
    iters: int = 200,
    seqlen: int = 2048,
    nsamples: int = 128,
    sampler: str = "rand",
    seed: int = 42,
    nblocks: int = 1,
    gradient_accumulate_steps: int = 1,
    not_use_best_mse: bool = False,
    dynamic_max_gap: int = -1,
    data_type: str = "int",  ##only support int for now
    scale_dtype: str = "fp16",
    to_quant_block_names: list = None,
    act_bits: int = 32,
    act_group_size: int = None,
    act_sym: bool = None,
    act_dynamic: bool = True,
    use_layer_wise: bool = False,
    **kwargs,
):
    """Run autoround weight-only quantization.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data. If none is provided, a dataloader must be supplied.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (bool): Whether symmetric quantization is to be used (default is False).
        weight_config (dict): Configuration for weight quantization (default is an empty dictionary).
        weight_config={
                    'layer1':##layer_name
                        {
                            'data_type': 'int',
                            'bits': 4,
                            'group_size': 32,
                            'sym': False,
                            'act_data_type': None,
                            'act_bits': 32,
                            'act_sym': None,
                            'act_dynamic': True,
                        }
                    ...,
                }
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for tuning (default is "auto").
        lr_scheduler: The learning rate scheduler to be used.
        dataset (str): The default dataset name (default is "NeelNanda/pile-10k").
        enable_quanted_input (bool): Whether to use the output of the previous quantized block as
                                the input for the current block (default is True).
        enable_minmax_tuning (bool): Whether to enable weight min-max tuning (default is True).
        lr (float): The learning rate (default is None, will be set to 1.0/iters).
        minmax_lr (float): The learning rate for min-max tuning (default is None, it will be set to lr automatically).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is False).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Data length of the sequence for tuning (default is 2048).
        nsamples (int): Number of samples (default is 128).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).
        nblocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        data_type (str): The data type to be used (default is "int").
        scale_dtype (str): The data type of quantization scale to be used (default is "float32"), different kernels
                           have different choices.
        to_quant_block_names (list): A list whose elements are list of block's layer names to be quantized.
        act_bits (int): Number of bits for activation quantization. Default is 32.
        act_group_size (int): Group size for activation quantization. Default is None.
        act_sym (bool): Whether to use symmetric activation quantization. Default is None.
        act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
    Returns:
        The quantized model.
    """
    from auto_round import AutoRound  # pylint: disable=E0401

    rounder = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=bits,
        group_size=group_size,
        sym=sym,
        layer_config=weight_config,
        enable_full_range=enable_full_range,  ##for symmetric, TODO support later
        batch_size=batch_size,
        amp=amp,
        device=device,
        lr_scheduler=lr_scheduler,
        dataset=dataset,
        enable_quanted_input=enable_quanted_input,
        enable_minmax_tuning=enable_minmax_tuning,
        lr=lr,
        minmax_lr=minmax_lr,
        low_gpu_mem_usage=low_gpu_mem_usage,
        iters=iters,
        seqlen=seqlen,
        nsamples=nsamples,
        sampler=sampler,
        seed=seed,
        nblocks=nblocks,
        gradient_accumulate_steps=gradient_accumulate_steps,
        not_use_best_mse=not_use_best_mse,
        dynamic_max_gap=dynamic_max_gap,
        data_type=data_type,  ## only support data_type
        scale_dtype=scale_dtype,
        to_quant_block_names=to_quant_block_names,
        act_bits=act_bits,
        act_group_size=act_group_size,
        act_sym=act_sym,
        act_dynamic=act_dynamic,
        low_cpu_mem_usage=use_layer_wise,
        **kwargs,
    )
    qdq_model, weight_config = rounder.quantize()
    return qdq_model, weight_config
