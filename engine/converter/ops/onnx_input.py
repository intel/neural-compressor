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
from ..graph_utils import names_from_input


# graph.input
@operator_registry(operator_type='ONNXINPUT')
class ONNXINPUT(Operator):
    def __init__(self):
        super().__init__()

    def extract(self, framework, node, model, nodes_dict):
        from ..onnx_utils import ONNX_DTYPE_ID
        self._name = node.name
        self._op_type = 'ONNXINPUT'
        output_tensor_name = names_from_input(self._name)[1]
        shape_len = len(node.type.tensor_type.shape.dim)
        shape = [-1] * shape_len
        dtype = ONNX_DTYPE_ID[node.type.tensor_type.elem_type]
        output_tensor = Tensor(
            name=output_tensor_name,
            shape=shape,
            dtype=dtype,
            source_op=[self._name],
            dest_op=nodes_dict[self._name].outputs,
        )

        self._output_tensors = [output_tensor]
