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


# tf.cast(x, dtype, name=None)
# input dtype -> dest stype
@operator_registry(operator_type='Cast')
class Cast(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):

        if framework == 'tensorflow':
            from ..tf_utils import TF_DTYPE_ID
            self._attr['SrcT'] = TF_DTYPE_ID[node.attr['SrcT'].type]
            self._attr['DstT'] = TF_DTYPE_ID[node.attr['DstT'].type]
            self._attr['Truncate'] = node.attr['Truncate'].b
        if framework == 'onnxruntime':
            from ..onnx_utils import ONNX_DTYPE_ID
            self._attr['DstT'] = ONNX_DTYPE_ID[node.attribute[0].i]
