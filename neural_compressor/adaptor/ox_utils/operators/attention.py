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
from .qdq_base_operator import QDQOperatorBase
from onnxruntime.quantization.quant_utils import attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto
'''
    Quantize Attention
'''


class AttentionQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        '''
            parameter node: Attention node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Attention node
        '''
        node = self.node
        assert (node.op_type == "Attention")

        parents = self.quantizer.model.get_parents(node)
        quantized_name = []
        scale = []
        zp = []
        for parent in parents[:2]:
            if parent.op_type == 'DequantizeLinear':
                quantized_name.append(parent.input[0])
                scale.append(parent.input[1])
                zp.append(parent.input[2])
                self.quantizer.remove_nodes.append(parent)
            elif parent.op_type == 'DynamicQuantizeLinear':
                quantized_name.append(parent.output[0])
                scale.append(parent.output[1])
                zp.append(parent.output[2])
 
        inputs = []
        inputs.extend(quantized_name)
        inputs.append(node.input[2])
        inputs.extend(scale)
        inputs.append(node.input[3] if len(node.input) > 3 else "")
        inputs.extend(zp)
        inputs.extend([node.input[4] if len(node.input) > 4 else ""])

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qattention_node = onnx.helper.make_node("QAttention", inputs, node.output, 
                                                 node.name, **kwargs)
        self.quantizer.new_nodes.append(qattention_node)

        self.quantizer.remove_nodes.append(node)

class QDQAttention(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Attention")

        if self.quantizer.static:
            super().quantize()
        else:
            self.quantizer.quantize_inputs(node, [0, 1])
        node.name = node.name + "_quant"
