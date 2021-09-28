#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

from .op import Operator, operator_registry
from .tensor import Tensor


# The inputs must be two-dimensional matrices
@operator_registry(operator_type='_QuantizedFusedMatMulAndDequantize')
class _QuantizedFusedMatMulAndDequantize(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):

        # for baremetal, if QuantizedMatMulWithBiasAndDequantize has post op
        # its output_dtype maybe the origin dtype in attributes, float32
        # but if it has not, use int8 (s8) for performance consideration
        if framework == 'tensorflow':
            self._attr['output_dtype'] = 'fp32'  # 's8'
            transpose_a = node.attr['transpose_a'].b
            transpose_b = node.attr['transpose_b'].b
            if transpose_a:
                self._attr['src0_perm'] = '1,0'
            if not transpose_b:
                self._attr['src1_perm'] = '1,0'

            epsilon = node.attr['epsilon'].f
            if epsilon != 0.0:
                self._attr['epsilon'] = epsilon

            # if 'leakyrelu_alpha' in attributes.attr.keys():
            #     leakyrelu_alpha = attributes.attr['leakyrelu_alpha'].f
            # else:
            #     leakyrelu_alpha = None
            # if leakyrelu_alpha != None:
            #     self.attrs.attr['append_op'] = 'relu'
            #     self.attrs.attr['leakyrelu_alpha'] = leakyrelu_alpha
            fused_ops = node.attr['fused_ops'].list.s
            fused_ops = [str(i, encoding='utf-8') for i in fused_ops]
            self._attr['fused_ops'] = fused_ops
