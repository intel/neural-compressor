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
from onnxruntime.quantization.quant_utils import find_by_name, get_mul_node, \
                                                 QuantizedValueType
from onnx import onnx_pb as onnx_proto
from neural_compressor.adaptor.ox_utils.util import QuantizedValue
'''
    Used when quantize mode is QuantizationMode.IntegerOps.
'''


class MatMulInteger(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        node = self.node
        assert (node.op_type == "MatMul")

        parents = self.quantizer.model.get_parents(node)

        inputs = []
        quantized_name = []
        scale = []
        zp = []
        for parent in parents:
            if parent.op_type == 'DequantizeLinear':
                quantized_name.append(parent.input[0])
            else:
                quantized_name.append(parent.output[0])
            if parent.op_type == 'DynamicQuantizeLinear':
                scale.append(parent.output[1])
                zp.append(parent.output[2])
            else:
                scale.append(parent.input[1])
                zp.append(parent.input[2])
        inputs.extend(quantized_name)
        inputs.extend(zp)
        matmul_integer_output = node.output[0] + "_output_quantized"
        matmul_integer_node = onnx.helper.make_node("MatMulInteger",
                                                    inputs,
                                                    [matmul_integer_output], node.name)
        self.quantizer.new_nodes.append(matmul_integer_node)

        # Add cast operation to cast matmulInteger output to float.
        cast_op_output = matmul_integer_output + "_cast_output"
        cast_node = onnx.helper.make_node("Cast", [matmul_integer_output], [cast_op_output],
                                          matmul_integer_output + "_cast",
                                          to=onnx_proto.TensorProto.FLOAT)
        self.quantizer.new_nodes.append(cast_node)

        # Add mul operation to multiply scales of two inputs.
        scales_mul_op = node.name + "_scales_mul"

        scales_mul_node = find_by_name(scales_mul_op, self.quantizer.new_nodes)
        if scales_mul_node is None:
            scales_mul_node = get_mul_node([scale[0], scale[1]], 
                scales_mul_op + ":0", scales_mul_op)
            self.quantizer.new_nodes.append(scales_mul_node)

        scales_mul_op_output = scales_mul_node.output[0]

        # Add mul operation to multiply mul_scales_op result with output of MatMulInteger
        # and make the output of this node the same as output of original matmul node.
        output_scale_mul_op = node.name + "_output_scale_mul"
        self.quantizer.new_nodes.append(get_mul_node([cast_op_output, scales_mul_op_output], 
                                  node.output[0], 
                                  output_scale_mul_op))
        if parents[1].op_type == 'DequantizeLinear':
            self.quantizer.remove_nodes.append(parents[1])
        self.quantizer.remove_nodes.append(node)

'''
    Used when quantize mode is QuantizationMode.QLinearOps
'''


class QLinearMatMul(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        node = self.node
        assert (node.op_type == "MatMul")

        parents = self.quantizer.model.get_parents(node)
        if len(self.quantizer.model.get_children(node)) == 0:
            return
        child = self.quantizer.model.get_children(node)[0]

        qlinear_matmul_output = child.output[0]

        qlinear_matmul_inputs = []
        for parent in parents:
            qlinear_matmul_inputs.extend(parent.input)
        qlinear_matmul_inputs.extend(child.input[1:])

        qlinear_matmul_node = onnx.helper.make_node("QLinearMatMul", 
                                                    qlinear_matmul_inputs, 
                                                    [qlinear_matmul_output],
                                                    node.name)
        self.quantizer.new_nodes.append(qlinear_matmul_node)
        self.quantizer.remove_nodes.extend(parents)
        self.quantizer.remove_nodes.append(child)
        self.quantizer.remove_nodes.append(node)

class QDQMatMul(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "MatMul")

        self.quantizer.quantize_inputs(node, [0])
        if self.per_channel and find_by_name(node.input[1], self.quantizer.model.initializer()):
            self.quantizer.quantize_weights_per_channel(node, [1],
                                    self.weight_dtype, self.weight_scheme, -1)
        else:
            self.quantizer.quantize_inputs(node, [1])

        if not self.disable_qdq_for_node_output or self.quantizer.mode != 'qdq':
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"
