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
"""Gather Operator."""

import onnx

from neural_compressor_ort.algorithms.post_training_quant.operators.ops import op_registry, Operator
from neural_compressor_ort.algorithms.post_training_quant.utils import attribute_to_kwarg, find_by_name, ms_domain
from neural_compressor_ort.common.utils import DYNAMIC_QUANT, STATIC_QUANT


@op_registry(op_types="Gather, GatherElements, GatherND", mode=[DYNAMIC_QUANT, STATIC_QUANT])
class GatherOperator(Operator):
    """Gather Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(GatherOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        if not self.quantizer.is_valid_quantize_weight(node.input[0]):
            return False
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0])
        node.name = node.name + "_quant"

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node
        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)
        if len(children) == 0 or len(parents) == 0 or not node.name.endswith("_quant"):
            return False

        return True

    def convert(self):
        """Convert to QOperator format."""
        node = self.node

        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)

        if any([i.op_type == "DequantizeLinear" for i in parents]):
            from onnx import numpy_helper

            inputs = []
            inputs.append(parents[0].input[0])
            inputs.append(node.input[1])

            gather_new_output = node.output[0] + "_quantized"

            kwargs = {}
            for attribute in node.attribute:  # pragma: no cover
                kwargs.update(attribute_to_kwarg(attribute))

            gather_node = onnx.helper.make_node("Gather", inputs, [gather_new_output], node.name, **kwargs)
            self.quantizer.new_nodes.append(gather_node)
            if any([i.op_type != "QuantizeLinear" for i in children]):  # pragma: no cover
                dq_inputs = []
                dq_inputs.append(gather_new_output)
                dq_inputs.extend(parents[0].input[1:])
                dq_node = onnx.helper.make_node(
                    "DequantizeLinear", dq_inputs, [node.output[0]], node.name + "_DequantizeLinear"
                )
                self.quantizer.new_nodes.append(dq_node)

            out_scale = 1.0
            out_zp = 0
            for child in children:
                if child.op_type == "QuantizeLinear":
                    out_scale = numpy_helper.to_array(self.quantizer.model.get_initializer(child.input[1]))
                    out_zp = numpy_helper.to_array(self.quantizer.model.get_initializer(child.input[2]))
                    self.quantizer.remove_nodes.append(child)
                    for n in self.quantizer.model.get_children(child):
                        self.quantizer.model.replace_node_input(n, child.output[0], gather_new_output)

            # int8 weight will be recalculated for the first time
            if (
                any([child.op_type == "QuantizeLinear" for child in children])
                and self.quantizer.model.get_initializer(parents[0].input[0]) is not None
                and parents[0].input[0] not in self.quantizer.recalculate_quantized_value
            ):
                int8_tensor = numpy_helper.to_array(self.quantizer.model.get_initializer(parents[0].input[0]))
                in_scale = numpy_helper.to_array(self.quantizer.model.get_initializer(parents[0].input[1]))
                in_zp = numpy_helper.to_array(self.quantizer.model.get_initializer(parents[0].input[2]))
                new_int8_tensor = (((int8_tensor.astype("float32") - in_zp) * in_scale) / out_scale).round() + out_zp
                self.quantizer.model.set_initializer(parents[0].input[0], new_int8_tensor.astype(int8_tensor.dtype))
                self.quantizer.recalculate_quantized_value.append(parents[0].input[0])
            self.quantizer.remove_nodes.extend([node, parents[0]])
