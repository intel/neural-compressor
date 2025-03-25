import math
from abc import abstractmethod

import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F

from ..weight_only.modules import HPUWeightOnlyLinear
from neural_compressor.torch.utils import accelerator, logger


class HPUMixedPrecisionLinear(HPUWeightOnlyLinear):
    """Weight and Activations quant (W4A8 gptq) Linear for HPU device."""

    def __init__(
        self, in_features, out_features,
        **kwargs,
    ):
        """Init the HPUMixedPrecisionLinear object.
        """
        super(HPUMixedPrecisionLinear, self).__init__(in_features, out_features)

    def forward(self, input):
        """The forward function of HPUMixedPrecisionLinear."""
        input_dtype = input.dtype
        output_shape = input.shape[:-1] + (self.out_features,)
        scales = self.scales
        qweight = self.qweight
        zeros = self.qzeros
        weight = torch.ops.hpu.convert_from_uint4(qweight, scales/self.matmul_internal.scale_other, zeros, torch.float8_e4m3fn)     # todo: div scales in init
        output = self.matmul_internal(input, weight)
        output = output.to(dtype=input_dtype).reshape(
            output_shape
        )  # A cast is needed here as for some reason the vecquant2matmul_faster_old still allocate a float32 output.
        output = output + self.bias if self.bias is not None else output
        return output

    @staticmethod
    def convert_from_weight_only(obj):
        new_self = HPUMixedPrecisionLinear(obj.in_features, obj.out_features)
        for attr, value in vars(obj).items():
            setattr(new_self, attr, value)
        new_self.matmul_internal.no_input_quant = True # flag for 8bit input, which shouldn't be quantized in matmul
        return new_self

