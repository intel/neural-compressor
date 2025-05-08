# Copyright (c) 2025 Intel Corporation
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

from neural_compressor.torch.algorithms.fp8_quant._core.fp_utils import FP8_143_SCALES, FP8_143_SCALES_TRAITS


class ScaleToPow2:
    def calc(self, scale):
        scale_pow2 = 2.0 ** torch.ceil(torch.log2(scale))
        return scale_pow2


class ScaleIdentity:
    def calc(self, scale):
        return scale

class ScaleHwAlignedFixed:
    def __init__(self, device_type):
        self.device_type = device_type
    def calc(self, scale):
        hw_aligned_single_scale = FP8_143_SCALES[self.device_type][0]
        return hw_aligned_single_scale

class ScaleUnit:
    def calc(self, scale):
        return 1.0

# Considering range of hw aligned scales: 2^a, 2^a+1,..., 2^b (a<b)
# we want to choose scale s for maxabs m such that 2^a <= s=2^x <= 2^b (for integer a<=x<=b)
# and also 2^(x-1) < m <= 2^x
# if m>=2^b then s=2^b, therefore min(_, 2^b)
# if m<=2^a then s=2^a, therefore max(_, 2^a) --> 2^a <= min(max(_,2^a),2^b) <=2^b
# if s^a<m<2^b then m as a positive number can be written as m=2^y (y=log2(m))
# if y is integer then y=ciel(y) we choose x=y so s=2^x=2^y=2^ciel(y)=2^ciel(log2(m))
# else we choose x=ciel(y) and a<=x-1<y<x<=b and s=2^x=2^ciel(y)=2^ciel(log2(m))
# for Gaudi2 the range is 16^-2..16^1 so we change 2 with 16 and remember that:
# 16 = 2^4, log16(m)=log2(m)/log2(16)=log2(m)/4, and we get:
# we choose s=16^ciel(log16(m))=2^4^ciel(log2(m)/4)=2^(4*ciel(log2(m)/4))=2^(ciel(log2(m)/4)*4)
class ScaleToHwAligned:
    def __init__(self, device_type):
        self.device_type = device_type
    def calc(self, scale):
        scale_pow2 = ScaleToPow2().calc(scale)
        min_scale, max_scale, scale_factor = FP8_143_SCALES_TRAITS[self.device_type]
        scale_pow2_hw = torch.minimum(
            torch.maximum(
                2.0 ** (torch.ceil(torch.log2(scale_pow2) / scale_factor) * scale_factor),
                torch.tensor(min_scale, dtype=scale.dtype, device=scale.device),
                ),
            torch.tensor(max_scale, dtype=scale.dtype, device=scale.device),
        )
        return scale_pow2_hw

