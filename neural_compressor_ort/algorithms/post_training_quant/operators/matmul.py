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
"""MatMul Operator."""

import onnx
from onnx import onnx_pb as onnx_proto

from neural_compressor_ort.algorithms.post_training_quant.operators.ops import op_registry, Operator
from neural_compressor_ort.algorithms.post_training_quant.utils import attribute_to_kwarg, find_by_name, ms_domain
from neural_compressor_ort.common.utils import DYNAMIC_QUANT, STATIC_QUANT


@op_registry(op_types="MatMul", mode=[DYNAMIC_QUANT])
class MatMulOperator(Operator):
    """MatMul Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(MatMulOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        if not all([self.quantizer.model.get_initializer(i) is None for i in node.input]):
            return True
        elif all([i not in self.quantizer.quantized_value_map for i in node.input]):
            return False
        else:
            return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0])
        if self.per_channel and find_by_name(node.input[1], self.quantizer.model.initializer()):
            self.quantizer.quantize_weights_per_channel(node, [1], self.weight_dtype, self.weight_scheme, 1)
        else:
            self.quantizer.quantize_inputs(node, [1])

        node.name = node.name + "_quant"

    def convert(self):
        """Convert to QOperator format."""
        node = self.node

        parents = self.quantizer.model.get_parents(node)

        inputs = []
        quantized_name = []
        scale = []
        zp = []
        for parent in parents:
            if parent.op_type == "DequantizeLinear":
                quantized_name.append(parent.input[0])
            else:
                quantized_name.append(parent.output[0])
            if parent.op_type == "DynamicQuantizeLinear":
                scale.append(parent.output[1])
                zp.append(parent.output[2])
            else:
                scale.append(parent.input[1])
                zp.append(parent.input[2])
        inputs.extend(quantized_name)
        inputs.extend(zp)
        matmul_integer_output = node.output[0] + "_output_quantized"
        matmul_integer_node = onnx.helper.make_node("MatMulInteger", inputs, [matmul_integer_output], node.name)
        self.quantizer.new_nodes.append(matmul_integer_node)

        # Add cast operation to cast matmulInteger output to float.
        cast_op_output = matmul_integer_output + "_cast_output"
        cast_node = onnx.helper.make_node(
            "Cast",
            [matmul_integer_output],
            [cast_op_output],
            matmul_integer_output + "_cast",
            to=onnx_proto.TensorProto.FLOAT,
        )
        self.quantizer.new_nodes.append(cast_node)

        # Add mul operation to multiply scales of two inputs.
        scales_mul_op = node.name + "_scales_mul"

        scales_mul_node = find_by_name(scales_mul_op, self.quantizer.new_nodes)
        if scales_mul_node is None:
            scales_mul_node = onnx.helper.make_node(
                "Mul", [scale[0], scale[1]], [scales_mul_op + ":0"], scales_mul_op
            )
            self.quantizer.new_nodes.append(scales_mul_node)

        scales_mul_op_output = scales_mul_node.output[0]

        # Add mul operation to multiply mul_scales_op result with output of MatMulInteger
        # and make the output of this node the same as output of original matmul node.
        output_scale_mul_op = node.name + "_output_scale_mul"
        self.quantizer.new_nodes.append(
            onnx.helper.make_node(
                "Mul", [cast_op_output, scales_mul_op_output], [node.output[0]], output_scale_mul_op
            )
        )
        if parents[1].op_type == "DequantizeLinear":
            self.quantizer.remove_nodes.append(parents[1])
        self.quantizer.remove_nodes.append(node)


@op_registry(op_types="MatMul", mode=[STATIC_QUANT])
class StaticMatMulOperator(MatMulOperator):
    """MatMul Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(MatMulOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0])
        if self.per_channel and find_by_name(node.input[1], self.quantizer.model.initializer()):
            self.quantizer.quantize_weights_per_channel(node, [1], self.weight_dtype, self.weight_scheme, 1)
        else:
            self.quantizer.quantize_inputs(node, [1])

        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"

    def convert(self):
        """Convert to QOperator format."""
        node = self.node

        parents = self.quantizer.model.get_parents(node)
        if len(self.quantizer.model.get_children(node)) == 0 or not node.name.endswith(
            "_quant"
        ):  # pragma: no cover
            return

        qlinear_matmul_inputs = []
        if self.disable_qdq_for_node_output:
            for i in range(len(parents[0].input)):
                qlinear_matmul_inputs.extend([parent.input[i] for parent in parents])
            qlinear_matmul_node = onnx.helper.make_node(
                "MatMulIntegerToFloat", qlinear_matmul_inputs, node.output, node.name, domain="com.microsoft"
            )
        else:
            child = self.quantizer.model.get_children(node)[0]
            qlinear_matmul_output = child.output[0]
            for parent in parents:
                qlinear_matmul_inputs.extend(parent.input)
            qlinear_matmul_inputs.extend(child.input[1:])
            qlinear_matmul_node = onnx.helper.make_node(
                "QLinearMatMul", qlinear_matmul_inputs, [qlinear_matmul_output], node.name
            )
            self.quantizer.remove_nodes.append(child)
        self.quantizer.new_nodes.append(qlinear_matmul_node)
        self.quantizer.remove_nodes.extend(parents)
        self.quantizer.remove_nodes.append(node)
