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

from neural_compressor.adaptor.ox_utils.operators.ops import op_registry, Operator, qop_registry, QOperator
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg, ms_domain

@op_registry(op_types="Reshape, Transpose, Squeeze, Unsqueeze")
class Direct8BitOperator(Operator):
    def __init__(self, onnx_quantizer, onnx_node):
        super(Direct8BitOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        node = self.node
        if not self.quantizer.is_valid_quantize_weight(node.input[0]):
            return False
        return True
        
    def quantize(self):
        node = self.node
        self.quantizer.quantize_inputs(self.node, [0], direct_int8=True)
        if not self.disable_qdq_for_node_output or self.quantizer.mode != 'qdq':
            self.quantizer.quantize_outputs(self.node, direct_int8=True)
        node.name = node.name + "_quant"

    def convert_check(self, convert_format):
        node = self.node
        assert convert_format in ['static'], \
            "convert format for {} should be in ['static']".format(node.op_type)
            
        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)
        if (len(children) == 0 and len(parents) == 0) or \
            not node.name.endswith('_quant'):
            return False
        return True

    def convert(self, convert_format):
        node = self.node
       
        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)
        if any([i.op_type == 'DequantizeLinear' for i in parents]) and \
            any([i.op_type == 'QuantizeLinear' for i in children]):
            for parent in parents:
                if parent.op_type == 'DequantizeLinear':
                    self.node.input[0] = parent.input[0]
                    self.quantizer.remove_nodes.append(parents[0])
                    break
            for child in children:
                if child.op_type == 'QuantizeLinear':
                    self.quantizer.remove_nodes.append(child)
                    self.quantizer.model.replace_input_of_all_nodes(
                        child.output[0], node.output[0] + '_quantized')
            node.output[0] = node.output[0] + '_quantized' 
    
    def cast(self): # pragma: no cover
        node = self.node
        if node.input[0] not in [i.tensor_name for i in self.quantizer.new_value_info.values()]:
            return
        self.quantizer.dtype_cast(self.node, self.dtype)

@op_registry(op_types="Shape, Loop, Slice")
class DirectCastOperator(Operator): # pragma: no cover
    def __init__(self, onnx_quantizer, onnx_node):
        super(DirectCastOperator, self).__init__(onnx_quantizer, onnx_node)

    def cast(self):
        node = self.node
        if node.input[0] not in [i.tensor_name for i in self.quantizer.new_value_info.values()]:
            return
        self.quantizer.dtype_cast(self.node, self.dtype)

@qop_registry(op_types="Reshape, Transpose, Squeeze, Unsqueeze")
class QDirectOperator(QOperator):
    def __init__(self, onnx_node, children, initializers, channel_axis, exclude_output_quantization):
        super().__init__(onnx_node, children, initializers, channel_axis, exclude_output_quantization)

    def convert(self):
        node = self.node
        add_nodes = []
        inputs = []
        inits = []
        if all([i.op_type != 'DequantizeLinear' for i in self.children]):
            return False, add_nodes
        for child in self.children:
            if child.op_type == 'DequantizeLinear':
                in_dq = onnx.helper.make_node(
                    'DequantizeLinear',
                    [node.input[0], child.input[1], child.input[2]],
                    [node.name + '_in_dequant_' + str(i)])
                inputs.append(node.name + '_in_dequant_' + str(i))
                add_nodes.append(in_dq)
                break
        outputs = node.output
        kwargs = {}
        for attribute in node.attribute: # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        gather_node = onnx.helper.make_node(
            node.op_type, inputs,
            outputs, node.name + '_convert', **kwargs)
        add_nodes.append(gather_node)
        return True, add_nodes, inits