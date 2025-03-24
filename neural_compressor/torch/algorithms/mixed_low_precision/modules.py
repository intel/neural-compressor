import math
from abc import abstractmethod

import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F

from ..weight_only.modules import HPUWeightOnlyLinear
from neural_compressor.torch.utils import accelerator, logger

cast_to_fp8_fcn = lambda x, dtype, scale_inv=None: torch.ops.hpu.cast_to_fp8_v2(x, scale_inv, False, False, dtype)[0]

class HPUMixedPrecisionLinear(HPUWeightOnlyLinear):
    """Weight and Activations quant (W4A8 gptq) Linear for HPU device."""

    def __init__(
        self, in_features, out_features, bias,
        **kwargs,
    ):
        """Init the HPUMixedPrecisionLinear object.
        """
        super(HPUMixedPrecisionLinear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        """The forward function of HPUMixedPrecisionLinear."""
        input_dtype = input.dtype
        output_shape = input.shape[:-1] + (self.out_features,)
        scales = self.scales
        scale_bf16_to_fp8 = self.scale_bf16_to_fp8
        qweight = self.qweight
        zeros = self.qzeros
        self.matmul_internal.scale_other = torch.nn.Parameter(scale_bf16_to_fp8)
        weight = torch.ops.hpu.convert_from_uint4(qweight, scales, zeros, torch.bfloat16)  # the uint4->fp8 is currently slower and with bugs. Jira ticket: https://jira.habana-labs.com/browse/SW-218009
        weight = cast_to_fp8_fcn(weight, torch.float8_e4m3fn)
        output = self.matmul_internal(input, weight)
        output = output.to(dtype=input_dtype).reshape(
            output_shape
        )  # A cast is needed here as for some reason the vecquant2matmul_faster_old still allocate a float32 output.
        output = output + self.bias if self.bias is not None else output
        return output

    @staticmethod
    def convert_from_weight_only(obj):
        bias = obj.bias is not None
        new_self = HPUMixedPrecisionLinear(obj.in_features, obj.out_features, bias)
        for attr, value in vars(obj).items():
            setattr(new_self, attr, value)
        new_self.matmul_internal.no_input_quant = True # flag for 8bit input, which shouldn't be quantized in matmul
        return new_self

