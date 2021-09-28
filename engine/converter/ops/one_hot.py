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
import copy


@operator_registry(operator_type='OneHot')
class OneHot(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):
        if framework == 'tensorflow':
            self._attr['axis'] = node.attr['axis'].i
            self._attr['depth'] = int(self._input_tensors[1].data)
            self._attr['on_value'] = int(self._input_tensors[2].data)  # float
            self._attr['off_value'] = int(self._input_tensors[3].data)  # float
            self._input_tensors = [copy.deepcopy(self._input_tensors[0])]
