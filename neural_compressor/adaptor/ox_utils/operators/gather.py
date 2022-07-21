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
from onnxruntime.quantization.quant_utils import QuantizedValueType, attribute_to_kwarg
from onnx import onnx_pb as onnx_proto
from neural_compressor.adaptor.ox_utils.util import QuantizedValue
'''
    Quantize Gather
'''

class GatherConverter(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        node = self.node
        assert node.op_type in ["Gather"]

        if len(self.quantizer.model.get_children(node)) == 0:
            return
        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)

        if len(parents) == 0:
            return
        if any([i.op_type == 'DequantizeLinear' for i in parents]):
            inputs = []
            inputs.append(parents[0].input[0])
            inputs.append(node.input[1])

            gather_new_output = node.output[0] + "_quantized"

            kwargs = {}
            for attribute in node.attribute:
                kwargs.update(attribute_to_kwarg(attribute))

            gather_node = onnx.helper.make_node("Gather",
                                                inputs,
                                                [gather_new_output],
                                                node.name,
                                                **kwargs)
            self.quantizer.new_nodes.append(gather_node)
            if any([i.op_type  != 'QuantizeLinear' for i in children]):
                dq_inputs = []
                dq_inputs.append(gather_new_output)
                dq_inputs.extend(parents[0].input[1:])
                dq_node = onnx.helper.make_node("DequantizeLinear",
                                                dq_inputs,
                                                [node.output[0]],
                                                node.name + '_DequantizeLinear')
                self.quantizer.new_nodes.append(dq_node)
                
            for child in children:
                if child.op_type == 'QuantizeLinear':
                    self.quantizer.remove_nodes.append(child)
                    for n in self.quantizer.model.get_children(child):
                        self.quantizer.model.replace_node_input(n, 
                                        child.output[0], gather_new_output)
            self.quantizer.remove_nodes.extend([node, parents[0]])

class GatherQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Gather")

        if not self.quantizer.is_valid_quantize_weight(node.input[0]):
            return
        self.quantizer.quantize_inputs(node, [0])
        if not self.disable_qdq_for_node_output or self.quantizer != 'qdq':
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"
