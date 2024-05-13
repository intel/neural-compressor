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
"""Binary operator."""

import onnx

from neural_compressor_ort.algorithms.post_training_quant.operators.ops import op_registry, Operator
from neural_compressor_ort.algorithms.post_training_quant.utils import attribute_to_kwarg, find_by_name, ms_domain
from neural_compressor_ort.common.utils import DYNAMIC_QUANT, STATIC_QUANT


@op_registry(op_types="Add, Mul", mode=[STATIC_QUANT])
class BinaryOperator(Operator):
    """Binary operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(BinaryOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        data_found, _, _, _, _ = self.quantizer._get_quantization_params(node.output[0])
        if not data_found:
            return False
        if self.quantizer.backend == "TensorrtExecutionProvider":
            return True
        if not all([self.quantizer.is_valid_quantize_weight(i) for i in node.input]):
            return False
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, initializer_use_weight_qType=False)
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node
        children = self.quantizer.model.get_children(node)
        if len(children) == 0 or not node.name.endswith("_quant"):
            return False
        return True

    def convert(self):
        """Convert to QOperator format."""
        node = self.node
        parents = self.quantizer.model.get_parents(node)
        child = self.quantizer.model.get_children(node)[0]

        qlinear_binary_math_output = child.output[0]

        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qlinear_binary_math_inputs = []
        for parent in parents:
            qlinear_binary_math_inputs.extend(parent.input)
        qlinear_binary_math_inputs.extend(child.input[1:])

        qlinear_binary_math_node = onnx.helper.make_node(
            "QLinear" + node.op_type, qlinear_binary_math_inputs, [qlinear_binary_math_output], node.name, **kwargs
        )

        self.quantizer.new_nodes += [qlinear_binary_math_node]
        self.quantizer.remove_nodes.extend(parents)
        self.quantizer.remove_nodes.append(child)
        self.quantizer.remove_nodes.append(node)


@op_registry(op_types="Mod", mode=[STATIC_QUANT])
class BinaryDirect8BitOperator(Operator):
    """Binary operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(BinaryDirect8BitOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        data_found, _, _, _, _ = self.quantizer._get_quantization_params(node.output[0])
        if not data_found:
            return False
        if not all([self.quantizer.is_valid_quantize_weight(i) for i in node.input]):
            return False

        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, initializer_use_weight_qType=False)
        if not self.disable_qdq_for_node_output or self.quantizer.mode != "qdq":
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node

        children = self.quantizer.model.get_children(node)
        if len(children) == 0 or not node.name.endswith("_quant"):
            return False
        return True

    def convert(self):
        """Convert to QOperator format."""
        node = self.node
        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)
        if any([i.op_type == "DequantizeLinear" for i in parents]) and any(
            [i.op_type == "QuantizeLinear" for i in children]
        ):
            for idx, parent in enumerate(parents):
                if parent.op_type == "DequantizeLinear":
                    self.node.input[idx] = parent.input[0]
                    self.quantizer.remove_nodes.append(parent)
            for child in children:
                if child.op_type == "QuantizeLinear":
                    self.quantizer.remove_nodes.append(child)
                    self.quantizer.model.replace_input_of_all_nodes(child.output[0], node.output[0] + "_quantized")
            node.output[0] = node.output[0] + "_quantized"

@op_registry(op_types="Sum, Sub, Div, Pow, Equal, Greater, GreaterOrEqual, Less, LessOrEqual", mode=[STATIC_QUANT])
class Float16BinaryOperator(Operator):
    """Float16 Binary operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(Float16BinaryOperator, self).__init__(onnx_quantizer, onnx_node)
