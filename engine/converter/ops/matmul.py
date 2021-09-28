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


@operator_registry(operator_type='MatMul')
class MatMul(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):
        if framework == 'tensorflow':
            self._attr['transpose_a'] = node.attr['transpose_a'].b
            self._attr['transpose_b'] = node.attr['transpose_b'].b
        if framework == 'onnxruntime':
            self._attr['transpose_a'] = False
            self._attr['transpose_b'] = False
