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
"""Base Operator."""

from neural_compressor.utils.utility import LazyImport
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg
onnx = LazyImport('onnx')

OPERATORS = {}
QOPERATORS= {}

def op_registry(op_types):
    """The class decorator used to register all Operator subclasses."""
    def decorator_op(cls):
        assert cls.__name__.endswith(
            'Operator'), "The name of subclass of Operator should end with \'Operator\' substring."
        if cls.__name__[:-len('Operator')] in OPERATORS: # pragma: no cover
            raise ValueError('Cannot have two operators with the same name.')
        for single_op_type in [op_type.strip() for op_type in op_types.split(',')]:
            OPERATORS[single_op_type] = cls
        return cls
    return decorator_op

def qop_registry(op_types):
    """The class decorator used to register all QOperator subclasses."""
    def decorator_op(cls):
        assert cls.__name__.endswith(
            'Operator'), "The name of subclass of QOperator should end with \'Operator\' substring."
        if cls.__name__[:-len('Operator')] in QOPERATORS: # pragma: no cover
            raise ValueError('Cannot have two operators with the same name.')
        for single_op_type in [op_type.strip() for op_type in op_types.split(',')]:
            if single_op_type.startswith('QLinear') or \
                single_op_type in ['QGemm', 'QAttention', 'QEmbedLayerNormalization', 'ArgMax', 
                                   'Reshape', 'Transpose', 'Squeeze', 'Unsqueeze', 'Gather',
                                   'MaxPool', 'Pad', 'Resize', 'Split']:
                QOPERATORS[single_op_type] = cls
        return cls
    return decorator_op


class Operator(object):
    """Base Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        self.quantizer = onnx_quantizer
        self.node = onnx_node
        if self.node.name in self.quantizer.config:
            self.dtype = self.quantizer.config[self.node.name]
        self.disable_qdq_for_node_output = True if onnx_node.op_type in \
            onnx_quantizer.op_types_to_exclude_output_quantization else False
        self.per_channel = False
        self.algorithm = 'minmax'
        self.weight_scheme = 'sym'
        self.weight_dtype = None
        self.activation_dtype = None
        self.activation_scheme = 'asym'
        if self.node.name in self.quantizer.config:
            if self.quantizer.config[self.node.name] != 'fp32':
                if 'weight' in self.quantizer.config[self.node.name].keys():
                    self.per_channel = self.quantizer.config[self.node.name]\
                        ['weight']['granularity'] == 'per_channel'
                    self.algorithm = self.quantizer.config[self.node.name]\
                        ['weight']['algorithm']
                    self.weight_scheme = self.quantizer.config[self.node.name]\
                        ['weight']['scheme']
                    self.weight_dtype = self.quantizer.config[self.node.name]\
                        ['weight']['dtype']
                if 'activation' in self.quantizer.config[self.node.name].keys():
                    self.activation_dtype = self.quantizer.config[self.node.name]\
                        ['activation']['dtype']
                    self.activation_scheme = self.quantizer.config[self.node.name]\
                        ['activation']['scheme']

    def quantize_check(self):
        """Check if quantizaion can be done."""
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node)
        if not self.disable_qdq_for_node_output or self.quantizer.mode != 'qdq':
            self.quantizer.quantize_outputs(node)

    def convert_check(self, convert_format):
        """Check if conversion can be done."""
        return True

    def convert(self, convert_format):
        """Convert to QOperator format."""
        return

    def cast(self): # pragma: no cover
        """Cast node."""
        self.quantizer.dtype_cast(self.node, self.dtype)

class QOperator(object):
    """Base QOperator."""

    def __init__(self, onnx_node, children, initializers):
        """Initialization."""
        self.node = onnx_node
        self.children = children
        self.initializers = initializers
        self.qop_list = ['QGemm', 'QAttention', 'QEmbedLayerNormalization',
                       'QLinearLeakyRelu', 'QLinearSigmoid', 'QLinearAdd','QLinearMul',
                       'QLinearConcat', 'QLinearConv', 'QLinearGlobalAveragePool',
                       'QLinearMatMul', 'QLinearAveragePool']

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

        # output q
        out_q = onnx.helper.make_node(
            'QuantizeLinear',
            [node.name + '_out', in_dq.input[1], in_dq.input[2]],
            node.output,
            node.name + '_out_quant')
        outputs = [node.name + '_out']
        add_nodes.append(out_q)

        kwargs = {}
        for attribute in node.attribute: # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))
            
        inputs.append(node.input[1:])
        new_node = onnx.helper.make_node(
            node.op_type, inputs,
            outputs, node.name + '_convert', **kwargs)
        add_nodes.append(new_node)
        return True, add_nodes, inits