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

from neural_compressor.adaptor.ox_utils.operators.ops import Operator, QOperator, op_registry, qop_registry
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg, find_by_name


@op_registry(op_types="MatMul")
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

        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"

    def convert_check(self, convert_format):
        """Check if conversion can be done."""
        node = self.node
        assert convert_format in [
            "dynamic",
            "static",
        ], "convert format for {} should be in ['dynamic', 'static']".format(node.op_type)
        if not node.name.endswith("_quant"):
            return False
        return True

    def convert(self, convert_format):
        """Convert to QOperator format."""
        node = self.node

        if convert_format == "dynamic":
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
        elif convert_format == "static":
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


@qop_registry(op_types="QLinearMatMul")
class QMatMulOperator(QOperator):
    """QLinearMatMul Operator."""

    def __init__(self, onnx_node, children, initializers):
        """Initialization."""
        super().__init__(onnx_node, children, initializers)

    def convert(self):
        """Convert to QDQ format."""
        node = self.node
        add_nodes = []
        inits = []
        # input dq
        in_dq1 = onnx.helper.make_node(
            "DequantizeLinear", node.input[:3], [node.name + "_in_dequant1"], node.name + "_in_dequant1"
        )

        in_dq2 = onnx.helper.make_node(
            "DequantizeLinear", node.input[3:6], [node.name + "_in_dequant2"], node.name + "_in_dequant2"
        )
        inputs = [node.name + "_in_dequant1", node.name + "_in_dequant2"]

        add_nodes.extend([in_dq1, in_dq2])
        # output q
        out_q = onnx.helper.make_node(
            "QuantizeLinear", [node.name + "_out", node.input[6], node.input[7]], node.output, node.name + "_out_quant"
        )
        outputs = [node.name + "_out"]
        add_nodes.append(out_q)

        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))

        matmul_node = onnx.helper.make_node("MatMul", inputs, outputs, node.name + "_convert", **kwargs)
        add_nodes.append(matmul_node)
        return True, add_nodes, inits


@op_registry(op_types="FusedMatMul")
class FusedMatMulOperator(Operator):
    """FusedMatMul Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(FusedMatMulOperator, self).__init__(onnx_quantizer, onnx_node)
