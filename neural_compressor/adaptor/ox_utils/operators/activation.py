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
"""Activation operator."""

import onnx

from neural_compressor.adaptor.ox_utils.operators.ops import Operator, QOperator, op_registry, qop_registry
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg, ms_domain


@op_registry(op_types="LeakyRelu, Sigmoid")
class ActivationOperator(Operator):
    """Activation operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(ActivationOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        data_found, _, _, _, _ = self.quantizer._get_quantization_params(node.output[0])
        if not data_found:
            return False
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        super().quantize()
        node.name = node.name + "_quant"

    def convert_check(self, convert_format):
        """Check if conversion can be done."""
        node = self.node
        assert convert_format in ["static"], "convert format for {} should be in ['static']".format(node.op_type)

        children = self.quantizer.model.get_children(node)
        if len(children) == 0 or not node.name.endswith("_quant"):
            return False
        return True

    def convert(self, convert_format):
        """Convert to QOperator format."""
        node = self.node

        parent = self.quantizer.model.get_parents(node)[0]
        child = self.quantizer.model.get_children(node)[0]

        inputs = []
        inputs.extend(parent.input)
        inputs.extend(child.input[1:])

        qlinear_activation_output = child.output[0]
        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qlinear_activation_node = onnx.helper.make_node(
            "QLinear" + node.op_type, inputs, [qlinear_activation_output], node.name, **kwargs
        )

        self.quantizer.new_nodes.append(qlinear_activation_node)
        self.quantizer.remove_nodes.extend([parent, child, node])


@op_registry(op_types="Relu, Clip")
class RemovableActivationOperator(Operator):
    """Removable activation operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(RemovableActivationOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        if node.input[0] not in self.quantizer.quantized_value_map:
            return False
        return True

    def quantize(self):
        """Do quantization."""
        node = self.node
        if node.output[0] in [i.name for i in self.quantizer.model.model.graph.output]:
            self.quantizer.dequantize_tensor(node, node.input[0])
        else:
            self.quantizer.model.replace_input_of_all_nodes(node.output[0], node.input[0])
            self.quantizer.remove_nodes.append(node)


@qop_registry(op_types="QLinearLeakyRelu, QLinearSigmoid")
class QActivationOperator(QOperator):
    """INT8 activation operator in QOperator format."""

    def __init__(self, onnx_node, children, initializers):
        """Initialization."""
        super().__init__(onnx_node, children, initializers)

    def convert(self):
        """Convert to QDQ format."""
        node = self.node
        add_nodes = []
        inits = []
        # input dq
        in_dq = onnx.helper.make_node(
            "DequantizeLinear", node.input[:3], [node.name + "_in_dequant"], node.name + "_in_dequant"
        )
        inputs = [node.name + "_in_dequant"]
        add_nodes.append(in_dq)
        # output q
        out_q = onnx.helper.make_node(
            "QuantizeLinear", [node.name + "_out", node.input[3], node.input[4]], node.output, node.name + "_out_quant"
        )
        outputs = [node.name + "_out"]
        add_nodes.append(out_q)

        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))

        activation_node = onnx.helper.make_node(
            node.op_type.split("QLinear")[-1], inputs, outputs, node.name + "_convert", **kwargs
        )
        add_nodes.append(activation_node)
        return True, add_nodes, inits


@op_registry(op_types="Softmax, BiasGelu, Elu, Exp, FastGelu, Gelu, Softplus, Tanh")
class Float16ActivationOperator(Operator):
    """Float16 Activation operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(Float16ActivationOperator, self).__init__(onnx_quantizer, onnx_node)
