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


import torch
from torch.nn import functional as F

from neural_compressor.common.logger import DEBUG, Logger, level
from neural_compressor.torch.utils import set_module

logger = Logger().get_logger()


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


def quantize_4bit(tensor, quantile=1.0, data_type="nf4", return_int=False, **kwargs):
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
    if "scale" in kwargs:
        scale = kwargs["scale"]
    else:
        scale = tensor.abs().max(1)[0] * quantile / max(allow_data)
        scale.unsqueeze_(dim=-1)
    tensor = tensor / scale
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
    double_quant = kwargs.get("double_quant", False)
    if return_int or double_quant:
        return q_tensor, scale, None
    return q_tensor * scale


def qdq_weight_asym(weight, num_bits=4, quantile=1.0, return_int=False, **kwargs):
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
    q = torch.clamp(torch.round(weight / scale) + zp, 0, maxq)
    double_quant = kwargs.get("double_quant", False)
    if return_int or double_quant:
        return q, scale, zp
    return scale * (q - zp)


def qdq_weight_sym(weight, num_bits=4, quantile=1.0, return_int=False, full_range=False, **kwargs):
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
    q = torch.clamp(torch.round(weight / scale), minq, maxq)
    double_quant = kwargs.get("double_quant", False)
    if return_int or double_quant:
        return q, scale, None
    return scale * q


def qdq_weight_actor(
    weight, num_bits, scheme, quantile=1.0, data_type="int", return_int=False, full_range=False, **kwargs
):
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

    if data_type in FLOAT_MAPPING.keys():
        return quantize_4bit(weight, quantile=quantile, data_type=data_type, return_int=return_int, **kwargs)
    if scheme == "sym":
        return qdq_weight_sym(weight, num_bits, quantile, return_int, full_range, **kwargs)
    else:
        return qdq_weight_asym(weight, num_bits, quantile, return_int, **kwargs)


def quant_weight(
    weight,
    num_bits=4,
    group_size=-1,
    scheme="asym",
    quantile=1.0,
    data_type="int",
    return_int=False,
    full_range=False,
    **kwargs,
):
    """Quant and dequant tensor with group size.

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
    double_quant = kwargs.get("double_quant", False)
    if num_bits <= 0:  # pragma: no cover
        return weight
    # case 1, group size = -1
    if group_size == -1 or weight.shape[1] < group_size:
        group_size = weight.shape[1]
    # case 2, reshape based on group size
    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        weight = qdq_weight_actor(
            weight,
            num_bits,
            scheme=scheme,
            quantile=quantile,
            return_int=return_int,
            full_range=full_range,
            data_type=data_type,
            **kwargs,
        )
        if return_int or double_quant:
            weight, scale, zp = weight
            weight = weight.reshape(orig_shape)
            scale = scale.reshape(orig_shape[0], -1)
            if zp is not None:
                zp = zp.reshape(orig_shape[0], -1)
            q_state = weight, scale, zp
        else:
            return weight.reshape(orig_shape)
    else:
        # case 3, process left part split by group size
        split_index = weight.shape[1] // group_size * group_size
        weight1 = weight[:, :split_index]
        weight1 = weight1.reshape(-1, group_size)
        weight1 = qdq_weight_actor(
            weight1,
            num_bits,
            scheme=scheme,
            quantile=quantile,
            return_int=return_int,
            full_range=full_range,
            data_type=data_type,
            **kwargs,
        )
        if return_int or double_quant:
            weight1, scale1, zp1 = weight1
            scale1 = scale1.reshape(orig_shape[0], -1)
            if zp1 is not None:
                zp1 = zp1.reshape(orig_shape[0], -1)
        weight1 = weight1.reshape(orig_shape[0], split_index)
        weight2 = weight[:, split_index:]
        weight2 = qdq_weight_actor(
            weight2,
            num_bits,
            scheme=scheme,
            data_type=data_type,
            quantile=quantile,
            return_int=return_int,
            full_range=full_range,
            **kwargs,
        )
        if return_int or double_quant:
            weight2, scale2, zp2 = weight2
            weight = torch.cat([weight1, weight2], dim=1)
            scale = torch.cat([scale1, scale2], dim=1)
            zp = None if zp2 is None else torch.cat([zp1, zp2], dim=1)
            q_state = (weight, scale, zp)
        else:
            weight = torch.cat([weight1, weight2], dim=1)
            return weight
    if double_quant:
        weight, scale, zp = q_state
        double_quant_dtype = kwargs.get("double_quant_dtype", "fp32")
        double_quant_num_bits = kwargs.get("double_quant_num_bits", 8)
        double_quant_scheme = kwargs.get("double_quant_scheme", "sym")
        double_quant_group_size = kwargs.get("double_quant_group_size", 256)
        double_quant_return_int = kwargs.get("double_quant_return_int", return_int)
        # process scale
        orig_scale_shape = scale.shape
        scale = scale.reshape(1, -1)
        scale = quant_weight(
            scale,
            double_quant_num_bits,
            double_quant_group_size,
            scheme=double_quant_scheme,
            quantile=1.0,
            data_type=double_quant_dtype,
            return_int=double_quant_return_int,
            full_range=False,
            double_quant=False,
        )
        if return_int:
            if double_quant_return_int:
                scale, hyper_scale, hyper_zp = scale
                scale = scale.reshape(orig_scale_shape)
                return weight, (scale, hyper_scale, hyper_zp), zp
            else:
                scale = scale.reshape(orig_scale_shape)
                return weight, scale, zp
        else:
            scale = scale.reshape(orig_scale_shape)
            if weight.shape[1] % group_size != 0:
                if zp is not None:
                    weight1 = weight1.reshape(-1, group_size) - zp[:, :-1].reshape(-1, 1)
                    weight2 = weight2 - zp[:, -1].reshape(-1, 1)
                else:
                    weight1 = weight1.reshape(-1, group_size)
                weight1 = weight1 * scale[:, :-1].reshape(-1, 1)
                weight1 = weight1.reshape(orig_shape[0], -1)
                weight2 = weight2 * scale[:, -1].reshape(-1, 1)
                weight = torch.cat([weight1, weight2], dim=1)
            else:
                if zp is not None:
                    weight = weight.reshape(-1, group_size) - zp.reshape(-1, 1)
                else:
                    weight = weight.reshape(-1, group_size)
                weight = weight * scale.reshape(-1, 1)
                weight = weight.reshape(orig_shape[0], -1)
            return weight
    else:
        return q_state


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
    org_weight = m.weight.data
    logger.info("Searching the best clip range with RTN algorithm")
    best_error = float("inf")
    best_clip_ratio = None
    n_grid = 200
    max_shrink = 0.2
    history = []
    for i_s in range(int(max_shrink * n_grid)):
        ratio = 1 - i_s / n_grid  # 1, 0.805-1.0
        cur_weight = quant_weight(
            m.weight.data,
            num_bits=num_bits,
            group_size=group_size,
            scheme=scheme,
            data_type=data_type,
            full_range=enable_full_range,
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
                            'dtype': 'int',
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
    supported_layers = ["Linear"]
    double_quant_dtype = kwargs.get("double_quant_dtype", "fp32")
    double_quant_config = {
        "double_quant": False if double_quant_dtype == "fp32" else True,
        "double_quant_dtype": double_quant_dtype,
        "double_quant_num_bits": kwargs.get("double_quant_num_bits", 8),
        "double_quant_scheme": kwargs.get("double_quant_scheme", "sym"),
        "double_quant_group_size": kwargs.get("double_quant_group_size", 256),
    }
    if return_int:
        compression_dtype = kwargs.get("compression_dtype", torch.int32)
        compression_dim = kwargs.get("compression_dim", 1)
        scale_dtype = kwargs.get("scale_dtype", torch.float32)
        device = kwargs.get("device", "cpu")
    for name, m in model.named_modules():
        if m.__class__.__name__ not in supported_layers:
            continue
        if name in weight_config:  # pragma: no cover
            data_type = weight_config[name].get("dtype", "int")
            num_bits = weight_config[name]["bits"]
            group_size = weight_config[name]["group_size"]
            scheme = weight_config[name]["scheme"]
            quantile = weight_config[name].get("quantile", 1.0)
        log_msg = (
            f"RTN quantization config: num_bits={num_bits}, group_size={group_size}, "
            + f"scheme={scheme}, quantile={quantile}"
        )
        if data_type != "int":
            log_msg += f", dtype={data_type}"
        elif scheme == "sym":  # nf4/fp4 is always [-7,7]
            log_msg += f", enable_full_range={enable_full_range}"
        if data_type == "fp32":
            continue
        logger.debug(f"RTN quantized module:{name, m}")
        logger.debug(log_msg)
        weight = m.weight.T if group_dim == 0 else m.weight
        if enable_mse_search:
            quantile = search_clip(m, num_bits, group_size, scheme, data_type, enable_full_range)
        if return_int:
            int_weight, scale, zp = quant_weight(
                weight,
                num_bits,
                group_size,
                scheme,
                quantile,
                data_type=data_type,
                return_int=True,
                full_range=enable_full_range,
                **double_quant_config,
            )
            int_weight = int_weight.T if group_dim == 0 else int_weight
            scale = scale.T if group_dim == 0 else scale
            zp = zp.T if group_dim == 0 and zp is not None else zp
            from neural_compressor.torch.quantization.layers import WeightOnlyLinear

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
            )
            new_module.pack(int_weight, scale, zp, m.bias)
            if name == "":
                return new_module
            else:
                set_module(model, name, new_module)
        else:
            q_weight = quant_weight(
                weight,
                num_bits,
                group_size,
                scheme,
                quantile,
                data_type=data_type,
                full_range=enable_full_range,
                **double_quant_config,
            )
            q_weight = q_weight.T if group_dim == 0 else q_weight
            m.weight.data.copy_(q_weight)
    return model


def quant_weight_w_scale(weight, scale, zp, group_size=-1, dtype="int"):
    """Quant and dequant tensor with group size.

    Args:
        weight: input weight
        scale: scale
        zp: zero point
        group_size (int, optional): how many elements share one scale/zp. Defaults to -1.
        dtype: data type, for NF4 FP4

    Returns:
        output: int weight.
    """
    device = weight.device
    scale = scale.to(device)
    # NF4 FP4
    if dtype in FLOAT_MAPPING.keys():
        int_weight = quantize_4bit(
            weight,
            quantile=1.0,
            data_type=dtype,
            return_int=True,
            scale=scale,
        )[0]
        return int_weight
    # INT
    if zp is not None:
        zp = zp.to(device)
    if group_size == -1:
        return torch.round(weight / scale) if zp is None else torch.round(weight / scale + zp)
    int_weight = torch.zeros(weight.shape).to(device)
    leng = weight.shape[1] // group_size
    tail_flag = False if weight.shape[1] % group_size == 0 else True
    for i in range(leng):
        int_weight_tmp = weight[:, i * group_size : (i + 1) * group_size] / scale[:, i].unsqueeze(1)
        if zp is not None:
            int_weight_tmp += zp[:, i].unsqueeze(1)
        int_weight[:, i * group_size : (i + 1) * group_size] = torch.round(int_weight_tmp)
    if tail_flag:
        int_weight_tmp = weight[:, leng * group_size :] / scale[:, -1].unsqueeze(1)
        if zp is not None:
            int_weight_tmp += zp[:, -1].unsqueeze(1)
        int_weight[:, leng * group_size :] = torch.round(int_weight_tmp)
    return int_weight


from neural_compressor.torch.quantization.config import RTNWeightQuantConfig


def apply_rtn_on_single_module(module: torch.nn.Module, quant_config: RTNWeightQuantConfig) -> torch.nn.Module:
    # TODO (Yi) remove it
    enable_full_range = quant_config.enable_full_range
    enable_mse_search = quant_config.enable_mse_search
    group_dim = quant_config.group_dim
    dtype = quant_config.weight_dtype
    num_bits = quant_config.weight_bits
    scheme = "sym" if quant_config.weight_sym else "asym"
    group_size = quant_config.weight_group_size
    return_int = quant_config.return_int
    double_quant_dtype = quant_config.double_quant_dtype
    double_quant_num_bits = quant_config.double_quant_bits
    double_quant_scheme = "sym" if quant_config.double_quant_sym else "asym"
    double_quant_group_size = quant_config.double_quant_group_size
    return rtn_quantize(
        module,
        num_bits,
        group_size,
        scheme,
        return_int=return_int,
        data_type=dtype,
        enable_full_range=enable_full_range,
        enable_mse_search=enable_mse_search,
        group_dim=group_dim,
        double_quant_dtype=double_quant_dtype,
        double_quant_scheme=double_quant_scheme,
        double_quant_num_bits=double_quant_num_bits,
        double_quant_group_size=double_quant_group_size,
    )
