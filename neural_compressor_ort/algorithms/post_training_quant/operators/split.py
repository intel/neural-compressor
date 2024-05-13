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

from neural_compressor_ort.algorithms.post_training_quant.operators.ops import op_registry, Operator
from neural_compressor_ort.algorithms.post_training_quant.utils import attribute_to_kwarg, find_by_name, ms_domain
from neural_compressor_ort.common.utils import DYNAMIC_QUANT, STATIC_QUANT


@op_registry(op_types="Split", mode=[STATIC_QUANT])
class SplitOperator(Operator):
    """Split Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(SplitOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_outputs(self.node, direct_int8=True)
        node.name = node.name + "_quant"

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node
        parent = self.quantizer.model.get_parents(node)[0]
        children = self.quantizer.model.get_children(node)
        if (
            parent.op_type != "DequantizeLinear" or len(children) == 0 or not node.name.endswith("_quant")
        ):  # pragma: no cover
            return False
        return True

    def convert(self):
        """Convert to QOperator format."""
        node = self.node

        parent = self.quantizer.model.get_parents(node)[0]
        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))

        quantized_input_names = []
        quantized_input_names.append(parent.input[0])
        if len(node.input) > 1:  # pragma: no cover
            quantized_input_names.extend(node.input[1:])
        outputs = []
        for output in node.output:
            if output in self.quantizer.model.input_name_to_nodes:
                child = self.quantizer.model.input_name_to_nodes[output][0]
                if child.op_type == "QuantizeLinear":
                    self.quantizer.remove_nodes.append(child)
                    outputs.append(child.output[0])
                else:  # pragma: no cover
                    outputs.append(output)
            else:  # pragma: no cover
                outputs.append(output + "_quantized")

        quantized_node = onnx.helper.make_node(node.op_type, quantized_input_names, outputs, node.name, **kwargs)
        self.quantizer.new_nodes.append(quantized_node)
        self.quantizer.remove_nodes.extend([parent, node])
