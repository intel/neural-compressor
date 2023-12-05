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


from enum import Enum, IntEnum
import torch
from torch.nn import functional as F
from torch.autograd import Function

from neural_compressor.common.logger import DEBUG, Logger, level
from neural_compressor.torch.utils import set_module

logger = Logger().get_logger()

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

class ElemFormat(Enum):
    int8 = 1
    int4 = 2
    int2 = 3
    fp8_e5m2 = 4
    fp8_e4m3 = 5
    fp6_e3m2 = 6
    fp6_e2m3 = 7
    fp4 = 8
    fp4_e2m1 = 8
    float16 = 9
    fp16 = 9
    bfloat16 = 10
    bf16 = 10

    @staticmethod
    def from_str(s):
        assert(s != None), "String elem_format == None"
        s = s.lower()
        if hasattr(ElemFormat, s):
            return getattr(ElemFormat, s)
        else:
            raise Exception("Undefined elem format", s)
        
    @staticmethod
    def is_bf(s):
        if isinstance(s, str):
            assert(s != None), "String elem_format == None"
            s = s.lower()
            if hasattr(ElemFormat, s):
                return getattr(ElemFormat, s).value == 10
        elif isinstance(s, int):
            return s == 10
        
        raise Exception("Undefined elem format", s)

    @staticmethod
    def is_fp(s):
        if isinstance(s, str):
            assert(s != None), "String elem_format == None"
            s = s.lower()
            if hasattr(ElemFormat, s):
                return 4 <= getattr(ElemFormat, s).value <= 9
        elif isinstance(s, int):
            return 4 <= s <= 9
        
        raise Exception("Undefined elem format", s)


    @staticmethod
    def is_int(s):
        if isinstance(s, str):
            assert(s != None), "String elem_format == None"
            s = s.lower()
            if hasattr(ElemFormat, s):
                return 1 <= getattr(ElemFormat, s).value <= 3
        elif isinstance(s, int):
            return 1 <= s <= 3
        
        raise Exception("Undefined elem format", s)

class RoundingMode(IntEnum):
    nearest = 0
    floor = 1
    even = 2

    @staticmethod
    def string_enums():
        return [s.name for s in list(RoundingMode)]

def _get_min_norm(ebits):
    """ Valid for all float formats """
    emin = 2 - (2 ** (ebits - 1))
    return 0 if ebits == 0 else 2 ** emin

def _get_max_norm(ebits, mbits):
    """ Valid only for floats that define NaN """
    assert(ebits >= 5), "invalid for floats that don't define NaN"
    emax = 0 if ebits==0 else 2**(ebits - 1) - 1
    return 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)

_FORMAT_CACHE = {}
def _get_format_params(fmt):
    """ Allowed formats:
        - intX:         2 <= X <= 32, assume sign-magnitude, 1.xxx representation
        - floatX/fpX:   16 <= X <= 28, assume top exp is used for NaN/Inf
        - bfloatX/bfX:  9 <= X <= 32
        - fp4,                  no NaN/Inf
        - fp6_e3m2/e2m3,        no NaN/Inf 
        - fp8_e4m3/e5m2,        e5m2 normal NaN/Inf, e4m3 special behavior

        Returns:
          ebits: exponent bits
          mbits: mantissa bits: includes sign and implicit bits
          emax: max normal exponent
          max_norm: max normal number
          min_norm: min normal number
    """
    if type(fmt) is str:
        fmt = ElemFormat.from_str(fmt)

    if fmt in _FORMAT_CACHE:
        return _FORMAT_CACHE[fmt]

    if fmt == ElemFormat.int8:
        ebits, mbits = 0, 8
        emax = 0
    elif fmt == ElemFormat.int4:
        ebits, mbits = 0, 4
        emax = 0
    elif fmt == ElemFormat.int2:
        ebits, mbits = 0, 2
        emax = 0
    elif fmt == ElemFormat.fp8_e5m2:
        ebits, mbits = 5, 4
        emax = 2**(ebits - 1) - 1
    elif fmt == ElemFormat.fp8_e4m3:
        ebits, mbits = 4, 5
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp6_e3m2:
        ebits, mbits = 3, 4
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp6_e2m3:
        ebits, mbits = 2, 5
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp4:
        ebits, mbits = 2, 3
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.float16:
        ebits, mbits = 5, 12
        emax = 2**(ebits - 1) - 1
    elif fmt == ElemFormat.bfloat16:
        ebits, mbits = 8, 9
        emax = 2**(ebits - 1) - 1
    else:
        raise Exception("Unknown element format %s" % fmt)
    
    if fmt != ElemFormat.fp8_e4m3:
        max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
    else:
        max_norm = 2**emax * 1.75  # FP8 has custom max_norm

    min_norm = _get_min_norm(ebits)

    _FORMAT_CACHE[fmt] = (ebits, mbits, emax, max_norm, min_norm)

    return ebits, mbits, emax, max_norm, min_norm

# Never explicitly compute 2**(-exp) since subnorm numbers have
# exponents smaller than -126
def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)

def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)

def _round_mantissa(A, bits, round, clamp=False):
    """
    Rounds mantissa to nearest bits depending on the rounding method 'round'
    Args:
      A     {PyTorch tensor} -- Input tensor
      round {str}            --  Rounding method
                                 "floor" rounds to the floor
                                 "nearest" rounds to ceil or floor, whichever is nearest
    Returns:
      A {PyTorch tensor} -- Tensor with mantissas rounded
    """

    if round == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round == "even":
        absA = torch.abs(A)
        # find 0.5, 2.5, 4.5 ...
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise Exception("Unrecognized round method %s" % (round))

    # Clip values that cannot be expressed by the specified number of bits
    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A

def _shared_exponents(A, method="max", axes=None, ebits=0):
    """
    Get shared exponents for the passed matrix A.
    Args:
      A      {PyTorch tensor} -- Input tensor
      method {str}            -- Exponent selection method.
                                 "max" uses the max absolute value
                                 "none" uses an exponent for each value (i.e., no sharing)
      axes   {list(int)}      -- List of integers which specifies the axes across which
                                 shared exponents are calculated.
    Returns:
      shared_exp {PyTorch tensor} -- Tensor of shared exponents
    """

    if method == "max":
        if axes is None:
            shared_exp = torch.max(torch.abs(A))
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    elif method == "none":
        shared_exp = torch.abs(A)
    else:
        raise Exception("Unrecognized shared exponent selection method %s" % (method))

    # log2(shared_exp) and truncate to integer
    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    # Restrict to [-emax, emax] range
    if ebits > 0:
        emax = 2**(ebits-1) - 1
        #shared_exp = torch.clamp(shared_exp, -emax, emax)
        # Overflow to Inf
        shared_exp[shared_exp > emax] = float("NaN")
        # Underflows are set to -127 which causes them to be
        # flushed to 0 later
        shared_exp[shared_exp < -emax] = -emax

    return shared_exp

def _reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension to apply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape

def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    # Undo tile reshaping
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A

def _quantize_elemwise_core(A, bits, exp_bits, max_norm, round='nearest',
                            saturate_normals=False, allow_denorm=True):
    """ Core function used for element-wise quantization
    Arguments:
      A         {PyTorch tensor} -- A tensor to be quantized
      bits      {int}            -- Number of mantissa bits. Includes
                                    sign bit and implicit one for floats
      exp_bits  {int}            -- Number of exponent bits, 0 for ints
      max_norm  {float}          -- Largest representable normal number
      round     {str}            -- Rounding mode: (floor, nearest, even)
      saturate_normals {bool}    -- If True, normal numbers (i.e., not NaN/Inf)
                                    that exceed max norm are clamped.
                                    Must be True for correct MX conversion.
      allow_denorm     {bool}    -- If False, flush denorm numbers in the
                                    elem_format to zero.
    Returns:
      quantized tensor {PyTorch tensor} -- A tensor that has been quantized
    """
    # Flush values < min_norm to zero if denorms are not allowed
    if not allow_denorm and exp_bits > 0:
        min_norm = _get_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(
            torch.abs(A) + (A == 0).type(A.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2**(exp_bits-1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of bits are in the integer portion of the number
    out = _safe_lshift(out, bits - 2, private_exp)

    out = _round_mantissa(out, bits, round, clamp=False)

    # Undo scaling
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                           torch.sign(out) * float("Inf"), out)

    # handle Inf/NaN
    out[A == float("Inf")] = float("Inf")
    out[A == -float("Inf")] = -float("Inf")
    out[A == float("NaN")] = float("NaN")

    return out

def _quantize_fp(A, exp_bits=None, mantissa_bits=None,
                 round='nearest', allow_denorm=True):
    """ Quantize values to IEEE fpX format. The format defines NaN/Inf
        and subnorm numbers in the same way as FP32 and FP16.
    Arguments:
        exp_bits        {int} -- number of bits used to store exponent
        mantissa_bits   {int} -- number of bits used to store mantissa, not
                                 including sign or implicit 1
        round           {str} -- Rounding mode, (floor, nearest, even)
    """
    # Shortcut for no quantization
    if exp_bits is None or mantissa_bits is None:
        return A

    max_norm = _get_max_norm(exp_bits, mantissa_bits+2)

    output = _quantize_elemwise_core(
            A, bits=mantissa_bits + 2, exp_bits=exp_bits,
            max_norm=max_norm, round=round, allow_denorm=allow_denorm)

    return output

def _quantize_bfloat(A, bfloat, round='nearest', allow_denorm=True):
    """ Quantize values to bfloatX format
    Arguments:
      bfloat      {int}       -- Total number of bits for bfloatX format,
                                 Includes 1 sign, 8 exp bits, and variable
                                 mantissa bits. Must be >= 9.
    """
    # Shortcut for no quantization
    if bfloat == 0 or bfloat == 32:
        return A

    max_norm = _get_max_norm(8, bfloat-7)

    return _quantize_elemwise_core(
            A, bits=bfloat-7, exp_bits=8, max_norm=max_norm, round=round,
            allow_denorm=allow_denorm)

def quantize_elemwise_op(A, mx_specs, round="even"):
    """A function used for element-wise quantization with mx_specs
    Arguments:
      A          {PyTorch tensor} -- a tensor that needs to be quantized
      mx_specs {dictionary}     -- dictionary to specify mx_specs
      round      {str}            -- Rounding mode, choose from (floor, nearest, even)
                                     (default: "nearest")
    Returns:
      quantized value {PyTorch tensor} -- a tensor that has been quantized
    """
    if mx_specs is None:
        return A

    out_dtype = mx_specs['out_dtype']
    elem_format = ElemFormat.from_str(out_dtype)
    ebits, mbits, _, _, _ = _get_format_params(elem_format)
    if ElemFormat.is_bf(out_dtype):
        A = _quantize_bfloat(A, bfloat=ebits+mbits-1, round=round,
                             allow_denorm=True)
    elif ElemFormat.is_fp(out_dtype):
        A = _quantize_fp(A, exp_bits=5, mantissa_bits=ebits+mbits-1 - 6,
                         round=round, allow_denorm=True)
    else:
        raise ValueError("Cannot set {} for output dtype.".format(out_dtype))
    return A

def _quantize_mx(
    A,
    scale_bits,
    elem_format,    # can be None for no quantization
    shared_exp_method="max",
    axes=None,
    block_size=32,
    round="nearest",
    flush_fp32_subnorms=False,
):
    """Function used for MX* quantization
    """
    # Shortcut for no quantization
    if elem_format == None:
        return A

    assert(scale_bits > 0)

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    # Perform tiling to the hardware vector size
    A, axes, orig_shape, padded_shape = _reshape_to_blocks(
        A, axes, block_size
    )

    ####################
    # Quantize
    ####################
    shared_exp_axes = [x + 1 for x in axes]

    # Get shared exponents
    shared_exp = _shared_exponents(
        A, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
    )

    # Flush subnormal FP32 inputs to zero
    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    # Offset the max exponent by the largest representable exponent
    # in the element data format
    shared_exp = shared_exp - emax

    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    A = A / (2**shared_exp)

    A = _quantize_elemwise_core(
            A, mbits, ebits, max_norm, round=round,
            allow_denorm=True, saturate_normals=True)

    A = A * (2**shared_exp)

    # Undo tile reshaping
    A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A

def quantize_mx_op(
    A,
    mx_specs: dict,
    elem_format=None,
    block_size=32,
    scale_bits=8,
    axes=None,
    round="nearest",
    expand_and_reshape=False,
):

    if elem_format == None:
        return A
    elif type(elem_format) is str:
        elem_format = ElemFormat.from_str(elem_format)

    return _quantize_mx(
            A, scale_bits,
            elem_format, block_size=block_size,
            axes=axes, round=round,
            shared_exp_method="max",
            flush_fp32_subnorms=False)

class MXLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, mx_specs=None):
        # element-wise quantize for input
        bf_in = quantize_elemwise_op(
            input, mx_specs=mx_specs, round="nearest"
        )

        # element-wise quantize for weight and bias
        bf_weight = quantize_elemwise_op(
            weight, mx_specs=mx_specs, round="nearest"
        )

        if bias is not None:
            bf_bias = quantize_elemwise_op(
                bias, mx_specs=mx_specs, round="nearest"
            )
        else:
            bf_bias = None


        # MX quantize everything along input size
        qis_input = quantize_mx_op(
            bf_in,
            mx_specs,
            elem_format=mx_specs['act_dtype'],
            axes=[-1],
            round="nearest",
        )
        qis_weight = quantize_mx_op(
            bf_weight,
            mx_specs,
            elem_format=mx_specs['weight_dtype'],
            axes=[-1],
            round="nearest",
        )

        # compute output
        output = F.linear(qis_input, qis_weight)
        output = quantize_elemwise_op(
            output, mx_specs=mx_specs, round="nearest"
        )

        if bias is not None:
            output = output + bf_bias
            output = quantize_elemwise_op(
                output, mx_specs=mx_specs, round="nearest"
            )

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Args:
            ctx: Pytorch convention.
            grad_output: A tensor of gradient of outputs
        Returns:
            grad_inputs: A tensor of gradient
        """
        return grad_outputs, None, None, None

class MXLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        mx_specs=None,
        name=None,
    ):
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = mx_specs
        super().__init__(in_features, out_features, bias)

    def apply_mx_specs(self, mx_specs):
        self.mx_none = mx_specs is None
        self.mx_specs = mx_specs

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super().forward(inputs)
        
        return MXLinearFunction.apply(inputs, self.weight, self.bias, self.mx_specs)

def mx_quantize(
    model,
    weight_dtype="fp8_e5m2",
    act_dtype="fp8_e5m2",
    out_dtype="bfloat16",
    config={},
    **kwargs,
):
    """Quant the model with round to nearst method.

    Args:
        model: torch module
        config (dict, optional): specific layer wise configurations. Defaults to {}.
            For example,
                config={
                    'fc2':
                        {
                            'bits': 4,
                            'group_size': 32,
                            'scheme': 'sym'
                            'gptq_perm': [1, 1, ...] # for gptq perm
                        }
                }

    Returns:
        model: fake quantized torch module
    """
    assert isinstance(model, torch.nn.Module), "only support torch module"
    supported_layers = ["Linear"]
    for name, m in model.named_modules():
        if m.__class__.__name__ not in supported_layers:
            continue
        if name in config:  # pragma: no cover
            weight_dtype = config[name]["weight_dtype"]
            act_dtype = config[name]["act_dtype"]
            out_dtype = config[name]["out_dtype"]
        logger.debug(f"MX quantized module:{name, m}")
        log_msg = (
            f"MX quantization config: weight_dtype={weight_dtype}, act_dtype={act_dtype}, "
            + f"out_dtype={out_dtype}"
        )
        logger.debug(log_msg)
        mx_specs = {
            "weight_dtype": weight_dtype,
            "act_dtype": act_dtype,
            "out_dtype": out_dtype,
        }
        tmp_stat = m.state_dict()
        new_module = MXLinear(
            m.in_features,
            m.out_features,
            bias=m.bias is not None,
            mx_specs=mx_specs,
            name=name,
        )
        new_module.load_state_dict(tmp_stat)
        if name == "":
            return new_module
        else:
            set_module(model, name, new_module)
    return model


from neural_compressor.torch.quantization.config import MXQuantConfig


def apply_mx_on_model(model: torch.nn.Module, configs_mapping: MXQuantConfig) -> torch.nn.Module:
    config = {}
    for (op_type, op_name), quant_config in configs_mapping.items():
        config[op_name] = quant_config.to_dict()
    return mx_quantize(model, config=config)

from typing import Dict, Tuple
from neural_compressor.torch.utils import fetch_module, register_algo, set_module
from neural_compressor.common.utility import MX_QUANT

@register_algo(name=MX_QUANT)
def mx_quantize_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], MXQuantConfig], *args, **kwargs
) -> torch.nn.Module:
    """The main entry to apply rtn quantization."""
    model = apply_mx_on_model(model, configs_mapping)
    return model

