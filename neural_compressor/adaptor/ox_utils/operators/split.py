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
"""Split Operator."""

import onnx
from neural_compressor.adaptor.ox_utils.operators.ops import op_registry, Operator, QOperator, qop_registry
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg


@op_registry(op_types="Split")
class SplitOperator(Operator):
    """Split Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(SplitOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0])
        if not self.disable_qdq_for_node_output or self.quantizer != 'qdq':
            self.quantizer.quantize_outputs(self.node, direct_int8=True)
        node.name = node.name + "_quant"

    def convert_check(self, convert_format):
        """Check if conversion can be done."""
        node = self.node
        assert convert_format in ['static'], \
            "convert format for {} should be in ['static']".format(node.op_type)

        parent = self.quantizer.model.get_parents(node)[0]
        children = self.quantizer.model.get_children(node)
        if parent.op_type != 'DequantizeLinear' or len(children) == 0 or \
            not node.name.endswith('_quant'): # pragma: no cover
            return False
        return True

    def convert(self, convert_format):
        """Convert to QOperator format."""
        node = self.node

        parent = self.quantizer.model.get_parents(node)[0]
        kwargs = {}
        for attribute in node.attribute: # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))

        quantized_input_names = []
        quantized_input_names.append(parent.input[0])
        if len(node.input) > 1: # pragma: no cover
            quantized_input_names.extend(node.input[1:])
        outputs = []
        for output in node.output:
            if output in self.quantizer.model.input_name_to_nodes:
                child = self.quantizer.model.input_name_to_nodes[output][0]
                if child.op_type == 'QuantizeLinear':
                    self.quantizer.remove_nodes.append(child)
                    outputs.append(child.output[0])
                else: # pragma: no cover
                    outputs.append(output)
            else: # pragma: no cover
                outputs.append(output + '_quatized')

        quantized_node = onnx.helper.make_node(node.op_type, 
                                               quantized_input_names, 
                                               outputs,
                                               node.name, **kwargs)
        self.quantizer.new_nodes.append(quantized_node)
        self.quantizer.remove_nodes.extend([parent, node])

    def cast(self): # pragma: no cover
        """Cast node."""
        node = self.node
        if node.input[0] not in [i.tensor_name for i in self.quantizer.new_value_info.values()]:
            return
        self.quantizer.dtype_cast(self.node, self.dtype)

@qop_registry(op_types="Split")
class QSplitOperator(QOperator):
    """QSplit Operator."""

    def __init__(self, onnx_node, children, initializers):
        """Initialization."""
        super().__init__(onnx_node, children, initializers)

    def convert(self):
        """Convert to QDQ format."""
        node = self.node
        add_nodes = []
        inputs = []
        inits = []

        if all([child.op_type not in self.qop_list or \
                child.op_type != 'DequantizeLinear' for child in self.children]):
            return False, add_nodes, inits

        # input dq
        for child in self.children:
            if child.op_type == 'DequantizeLinear':
                in_dq = onnx.helper.make_node(
                    'DequantizeLinear',
                    [node.input[0], child.input[1], child.input[2]],
                    [node.name + '_in_dequant'],
                    node.name + '_in_dequant')
                inputs.append(node.name + '_in_dequant')
                add_nodes.append(in_dq)
                break

        outputs = []
        for i, out in enumerate(node.output):
            out_q = onnx.helper.make_node(
                'QuantizeLinear',
                [node.name + '_out_' + str(i), in_dq.input[1], in_dq.input[2]],
                [node.output[i]],
                node.name + '_out_quant_' + str(i))
        outputs.append([node.name + '_out_quant_' + str(i)])
        add_nodes.append(out_q)

        outputs = node.output
        kwargs = {}
        for attribute in node.attribute: # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))

        gather_node = onnx.helper.make_node(
            node.op_type, inputs,
            outputs, node.name + '_convert', **kwargs)
        add_nodes.append(gather_node)
        return True, add_nodes, inits