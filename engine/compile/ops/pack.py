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


# tf 1.0 changes tf.pack -> tf.stack
# tf.stack(values, axis=0, name='stack')
# Packs the list of tensors in values into a tensor with
# rank one higher than each tensor in values,
# by packing them along the axis dimension.
# An int. The axis to stack along. Defaults to the first dimension. Negative values wrap around,
# so the valid range is [-(D+1), D]
# See also tf.concat, tf.tile, tf.repeat.
@operator_registry(operator_type='Pack')
class Pack(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):
        if framework == 'tensorflow':
            self._attr['axis'] = node.attr['axis'].i
            self._attr['N'] = node.attr['N'].i
