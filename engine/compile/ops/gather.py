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


@operator_registry(operator_type='Gather')
class Gather(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):
        self._op_type = 'Gather'
        if framework == 'tensorflow':
            self._attr['batch_dims'] = node.attr['batch_dims'].i
            try:
                axis = int(self._input_tensors[2].data)
            except BaseException:
                axis = 0
            self._attr['axis'] = axis
            
        if framework == 'onnxruntime':
            self._attr['batch_dims'] = 0
            if len(node.attribute) != 0:
                self._attr['axis'] = node.attribute[0].i
            else:
                self._attr['axis'] = 0
            

# tf.gather(params, indices, validate_indices=None, axis=None, batch_dims=0, name=None)
# argument validate_indices is deprecated
# indices must be an integer tensor of any dimension (usually 0-D or 1-D).
# Produces an output tensor with shape
# params.shape[:axis] + indices.shape[batch_dims:] + params.shape[axis + 1:]
# see: https://www.tensorflow.org/api_docs/python/tf/gather
@operator_registry(operator_type='GatherV2')
class GatherV2(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):
        self._op_type = 'Gather'
        if framework == 'tensorflow':
            self._attr['batch_dims'] = node.attr['batch_dims'].i
            try:
                axis = int(self._input_tensors[2].data)
            except BaseException:
                axis = 0
            _ = self._input_tensors.pop()
            self._attr['axis'] = axis
