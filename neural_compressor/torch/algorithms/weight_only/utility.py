# Copyright (c) 2024 Intel Corporation
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

import torch

from neural_compressor.torch.utils import logger

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


def quantize_4bit(tensor, quantile=1.0, dtype="nf4", return_int=False, **kwargs):
    """Quantize tensor to NF4/FP4 data type.

    Args:
        tensor: input tensor
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        dtype (str, optional): data type. Defaults to 'nf4'.
        return_int (bool, optional): whether return int data. Defaults to False.

    Returns:
        q_tensor: fake quantized tensor
    """
    assert dtype in FLOAT_MAPPING, "unexpected data type."
    allow_data = FLOAT_MAPPING[dtype]
    allow_data_bit = INT_MAPPING[dtype]
    # get scale and update tensor
    if "scale" in kwargs:
        scale = kwargs["scale"]
    else:
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
    double_quant = kwargs.get("double_quant", False)
    if return_int or double_quant:
        return tensor, scale, None
    return tensor.mul_(scale)


def qdq_weight_asym(weight, bits=4, quantile=1.0, return_int=False, **kwargs):
    """Quant and dequant tensor with asym schema.

    Args:
        weight:  input weight
        bits (int, optional): bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.

    Returns:
        output: qdq weight
    """
    maxq = torch.tensor(2**bits - 1)
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
    weight.clamp_(0, maxq)
    double_quant = kwargs.get("double_quant", False)
    if return_int or double_quant:
        return weight, scale, zp
    weight.sub_(zp)
    return weight.mul_(scale)


def qdq_weight_sym(weight, bits=4, quantile=1.0, return_int=False, full_range=False, **kwargs):
    """Quant and dequant tensor with sym schema.

    Args:
        weight : input weight
        bits (int, optional): bits. Defaults to 4.
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
    # assert bits > 1, "symmetric scheme only supports bits > 1"
    maxq = torch.tensor(2 ** (bits - 1) - 1).to(weight.device)
    minq = torch.tensor(-(2 ** (bits - 1))).to(weight.device)
    if bits == 1:  # pragma: no cover
        maxq = torch.tensor(2 ** (bits - 1))
        minq = torch.tensor(2 ** (bits - 1) - 1)
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
    double_quant = kwargs.get("double_quant", False)
    if return_int or double_quant:
        return weight, scale, None
    return weight.mul_(scale)


def qdq_weight_actor(weight, bits, scheme, quantile=1.0, dtype="int", return_int=False, full_range=False, **kwargs):
    """Quant and dequant tensor per channel. It is an in-place op.

    Args:
        weight : input weight
        bits (int, optional): bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        dtype (str, optional): select from int, nf4, fp4. Defaults to int.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        output: qdq weight
    """
    assert bits > 0, "bits should be larger than 0"

    if dtype in FLOAT_MAPPING.keys():
        return quantize_4bit(weight, quantile=quantile, dtype=dtype, return_int=return_int, **kwargs)
    if scheme == "sym":
        return qdq_weight_sym(weight, bits, quantile, return_int, full_range, **kwargs)
    else:
        return qdq_weight_asym(weight, bits, quantile, return_int, **kwargs)


def quant_tensor(
    weight,
    bits=4,
    group_size=-1,
    scheme="asym",
    quantile=1.0,
    dtype="int",
    return_int=False,
    full_range=False,
    **kwargs,
):
    """Quant and dequant tensor with group size.

    Args:
        weight: input weight
        bits (int, optional): bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to -1.
        scheme (str, optional): sym or asym. Defaults to "asym".
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        dtype (str, optional): select from int, nf4, fp4. Defaults to int.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        output: qdq weight.
    """
    double_quant = kwargs.get("double_quant", False)
    if bits <= 0:  # pragma: no cover
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
            bits,
            scheme=scheme,
            quantile=quantile,
            return_int=return_int,
            full_range=full_range,
            dtype=dtype,
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
            bits,
            scheme=scheme,
            quantile=quantile,
            return_int=return_int,
            full_range=full_range,
            dtype=dtype,
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
            bits,
            scheme=scheme,
            dtype=dtype,
            quantile=quantile,
            return_int=return_int,
            full_range=full_range,
            **kwargs,
        )
        if return_int or double_quant:
            weight2, scale2, zp2 = weight2
            weight.copy_(torch.cat([weight1, weight2], dim=1))
            scale = torch.cat([scale1, scale2], dim=1)
            zp = None if zp2 is None else torch.cat([zp1, zp2], dim=1)
            q_state = (weight, scale, zp)
        else:
            weight.copy_(torch.cat([weight1, weight2], dim=1))
            return weight
    if double_quant:
        weight, scale, zp = q_state
        double_quant_dtype = kwargs.get("double_quant_dtype", "fp32")
        double_quant_bits = kwargs.get("double_quant_bits", 8)
        double_quant_scheme = kwargs.get("double_quant_scheme", "sym")
        double_quant_group_size = kwargs.get("double_quant_group_size", 256)
        double_quant_return_int = kwargs.get("double_quant_return_int", return_int)
        # process scale
        orig_scale_shape = scale.shape
        scale = scale.reshape(1, -1)
        scale = quant_tensor(
            scale,
            dtype=double_quant_dtype,
            bits=double_quant_bits,
            group_size=double_quant_group_size,
            scheme=double_quant_scheme,
            quantile=1.0,
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
                    weight1 = weight1.reshape(-1, group_size).sub_(zp[:, :-1].reshape(-1, 1))
                    weight2 = weight2.sub_(zp[:, -1].reshape(-1, 1))
                else:
                    weight1 = weight1.reshape(-1, group_size)
                weight1 = weight1.mul_(scale[:, :-1].reshape(-1, 1))
                weight1 = weight1.reshape(orig_shape[0], -1)
                weight2 = weight2.mul_(scale[:, -1].reshape(-1, 1))
                weight = torch.cat([weight1, weight2], dim=1)
            else:
                if zp is not None:
                    weight = weight.reshape(-1, group_size) - zp.reshape(-1, 1)
                else:
                    weight = weight.reshape(-1, group_size)
                weight = weight.mul_(scale.reshape(-1, 1))
                weight = weight.reshape(orig_shape[0], -1)
            return weight
    else:
        return q_state


def search_clip(m, bits=4, group_size=32, scheme="asym", dtype="int", enable_full_range=False):
    """Search best clip range of each linear in current block.

    Args:
        m (torch.nn.Module): torch module.
        bits (int, optional): num bits.
        group_size (int, optional): how many elements share one scale/zp.
        scheme (str, optional): sym or asym.
        dtype (str, optional): select from int, nf4, fp4. Defaults to int.
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
        cur_weight = quant_tensor(
            m.weight.data,
            dtype=dtype,
            bits=bits,
            group_size=group_size,
            scheme=scheme,
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
            dtype=dtype,
            return_int=True,
            scale=scale,
        )[0]
        return int_weight
    # INT
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
