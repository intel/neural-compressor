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
"""GlobalAveragePool Operator."""

import onnx

from neural_compressor.adaptor.ox_utils.operators.ops import Operator, QOperator, op_registry, qop_registry
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg, ms_domain


@op_registry(op_types="GlobalAveragePool")
class GlobalAveragePoolOperator(Operator):
    """GlobalAveragePool Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(GlobalAveragePoolOperator, self).__init__(onnx_quantizer, onnx_node)

    def convert_check(self, convert_format):
        """Check if conversion can be done."""
        node = self.node
        assert convert_format in ["static"], "convert format for {} should be in ['static']".format(node.op_type)

        children = self.quantizer.model.get_children(node)
        if len(children) == 0:  # pragma: no cover
            return False
        return True

    def convert(self, convert_format):
        """Convert to QOperator format."""
        node = self.node

        parent = self.quantizer.model.get_parents(node)[0]
        child = self.quantizer.model.get_children(node)[0]

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        kwargs["channels_last"] = 0

        inputs = parent.input
        inputs.extend(child.input[1:])

        qnode = onnx.helper.make_node("QLinear" + node.op_type, inputs, child.output, node.name + "_quant", **kwargs)
        self.quantizer.new_nodes += [qnode]
        self.quantizer.remove_nodes.append(child)
        self.quantizer.remove_nodes.append(parent)
        self.quantizer.remove_nodes.append(node)


@qop_registry(op_types="QLinearGlobalAveragePool")
class QGlobalAveragePoolOperator(QOperator):
    """QLinearGlobalAveragePool Operator."""

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
        activation_node = onnx.helper.make_node("GlobalAveragePool", inputs, outputs, node.name + "_convert", **kwargs)
        add_nodes.append(activation_node)
        return True, add_nodes, inits
