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


# tf.strided_slice(input_, begin, end, strides=None, begin_mask=0, end_mask=0,
# ellipsis_mask=0,new_axis_mask=0, shrink_axis_mask=0, var=None, name=None)
@operator_registry(operator_type='StridedSlice')
class StridedSlice(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):

        if framework == 'tensorflow':
            self._attr['begin_mask'] = node.attr['begin_mask'].i
            self._attr['ellipsis_mask'] = node.attr['ellipsis_mask'].i
            self._attr['end_mask'] = node.attr['end_mask'].i
            self._attr['new_axis_mask'] = node.attr['new_axis_mask'].i
            self._attr['shrink_axis_mask'] = node.attr['shrink_axis_mask'].i

            begin = None
            end = None
            strides = None
            if self.input_tensors[1].source_op == []:
                begin = self.input_tensors[1]
                self._attr['begin'] = list2str(begin.data)
            if self.input_tensors[2].source_op == []:
                end = self.input_tensors[2]
                self._attr['end'] = list2str(end.data)
            if self.input_tensors[3].source_op == []:
                strides = self.input_tensors[3]
                self._attr['strides'] = list2str(strides.data)
