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
import numpy as np
from ..graph_utils import list2str


# tf.reshape(tensor, shape, name=None)
# param shape is s Tensor. Must be one of the following types: int32, int64.
# defines the shape of the output tensor.
@operator_registry(operator_type='Reshape')
class Reshape(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):
        # if framework == 'tensorflow':
        if self.input_tensors[1].source_op == []:
            dst_shape = list2str(self._input_tensors[1].data)
            self._attr['dst_shape'] = dst_shape
            self.input_tensors.pop()
