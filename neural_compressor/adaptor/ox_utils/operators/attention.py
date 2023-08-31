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
"""Attention operator."""

import onnx

from neural_compressor.adaptor.ox_utils.operators.ops import Operator, QOperator, op_registry, qop_registry
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg, find_by_name, ms_domain


@op_registry(op_types="Attention")
class AttentionOperator(Operator):
    """Attention operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(AttentionOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0, 1, 2])
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
        """Convert QDQ mode to QOperator format."""
        node = self.node
        parents = self.quantizer.model.get_parents(node)
        quantized_name = []
        scale = []
        zp = []
        for parent in parents[:2]:
            if parent.op_type == "DequantizeLinear":
                quantized_name.append(parent.input[0])
                scale.append(parent.input[1])
                zp.append(parent.input[2])
                self.quantizer.remove_nodes.append(parent)
            elif parent.op_type == "DynamicQuantizeLinear":
                quantized_name.append(parent.output[0])
                scale.append(parent.output[1])
                zp.append(parent.output[2])

        inputs = []
        inputs.extend(quantized_name)
        inputs.append(node.input[2])
        inputs.extend(scale)
        inputs.append(node.input[3] if len(node.input) > 3 else "")
        inputs.extend(zp)
        if len(node.input) > 4:
            inputs.append(node.input[4])

        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qattention_node = onnx.helper.make_node("QAttention", inputs, node.output, node.name, **kwargs)
        self.quantizer.new_nodes.append(qattention_node)

        self.quantizer.remove_nodes.append(node)


@qop_registry(op_types="QAttention")
class QAttentionOperator(QOperator):
    """QAttention operator."""

    def __init__(self, onnx_node, children, initializers):
        """Initialization."""
        super().__init__(onnx_node, children, initializers)

    def convert(self):
        """Convert to QDQ format."""
        node = self.node
        add_nodes = []
        inputs = []
        inits = []
        if find_by_name(node.input[3], self.initializers) is None:
            return False, add_nodes, inits
        # input dq
        in_dq1 = onnx.helper.make_node(
            "DequantizeLinear",
            [node.input[0], node.input[3], node.input[6]],
            [node.name + "_in_dequant1"],
            node.name + "_in_dequant1",
        )

        in_dq2 = onnx.helper.make_node(
            "DequantizeLinear",
            [node.input[1], node.input[4], node.input[7]],
            [node.name + "_in_dequant2"],
            node.name + "_in_dequant2",
        )
        inputs = [node.name + "_in_dequant1", node.name + "_in_dequant2", node.input[2], node.input[5]]

        add_nodes.extend([in_dq1, in_dq2])

        outputs = node.output
        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        binary_node = onnx.helper.make_node("Attention", inputs, outputs, node.name + "_convert", **kwargs)
        add_nodes.append(binary_node)
        return True, add_nodes, inits
