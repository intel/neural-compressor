#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx
from .base_operator import QuantOperatorBase
from .direct_q8 import QDQDirect8BitOp
from onnxruntime.quantization.quant_utils import QuantizedValueType
from onnx import onnx_pb as onnx_proto
from neural_compressor.adaptor.ox_utils.util import QuantizedValue


class QMaxPool(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        node = self.node
        assert (node.op_type == "MaxPool")

        if self.quantizer.opset_version < 12: # pragma: no cover
            return

        if len(self.quantizer.model.get_children(node)) == 0:
            return
        parent = self.quantizer.model.get_parents(node)[0]
        children = self.quantizer.model.get_children(node)
        if parent.op_type != 'DequantizeLinear' or \
            all([i.op_type != 'QuantizeLinear' for i in children]):
            return
        node.input[0] = parent.input[0]
        node.output[0] = node.output[0] + '_quantized'
        for child in children:
            if child.op_type == 'QuantizeLinear':
                self.quantizer.remove_nodes.append(child)
                for n in self.quantizer.model.get_children(child):
                    self.quantizer.model.replace_node_input(n,
                                    child.output[0], node.output[0])

        self.quantizer.remove_nodes.append(parent)

class QDQMaxPool(QDQDirect8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "MaxPool")

        # if version is less than 12, just no change
        if self.quantizer.opset_version < 12:
            return

        # Direct 8bits op
        super().quantize()