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
@operator_registry(operator_type='Gemm')
class Gemm(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):
        self._op_type = 'MatMulWithBias'
        if framework == 'onnxruntime':
            transpose_a = False
            transpose_b = False
            alpha = 1.0
            beta = 1.0
            for attr in node.attribute:
                if attr.name == 'transA':
                    transpose_a = bool(attr.i)
                if attr.name == 'transB':
                    transpose_b = bool(attr.i)
                if attr.name == 'alpha':
                    alpha = attr.f
                if attr.name == 'beta':
                    beta = attr.f
            if transpose_a:
                self._attr['src0_perm'] = '1,0'
            # see OneDNN InnerProduct related requirements
            if not transpose_b:
                self._attr['src1_perm'] = '1,0'
            else:
                self._attr['src1_perm'] = '0,1'
            if alpha != 1.0:
                self._attr['alpha'] = alpha
            if beta != 1.0:
                self._attr['beta'] = beta
