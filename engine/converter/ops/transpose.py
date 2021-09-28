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
from ..graph_utils import list2str


# tf.transpose(a, perm=None, conjugate=False, name='transpose')
# The returned tensor's dimension i will correspond to the input dimension perm[i].
# If perm is not given, it is set to (n-1...0), where n is the rank of the input tensor.
# Hence by default, this operation performs a regular matrix transpose on 2-D input Tensors.
# If conjugate is True and a.dtype is either complex64 or complex128 then the values of a are
# conjugated and transposed.
# normally, we don't need the param 'conjugate'
@operator_registry(operator_type='Transpose')
class Transpose(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):

        if framework == "tensorflow":
            if self._input_tensors[1].source_op == []:
                dst_perm = list(self._input_tensors[1].data)
                self._attr['dst_perm'] = list2str(dst_perm)
                src_perm = [i for i in range(len(dst_perm))]
                self._attr['src_perm'] = list2str(src_perm)
                _ = self._input_tensors.pop()

        if framework == "onnxruntime":
            dst_perm = node.attribute[0].ints
            src_perm = [i for i in range(len(dst_perm))]
            self._attr['src_perm'] = list2str(src_perm)
            self._attr['dst_perm'] = list2str(dst_perm)
