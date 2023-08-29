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
"""Pad Operator."""

import onnx

from neural_compressor.adaptor.ox_utils.operators.ops import Operator, QOperator, op_registry, qop_registry
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg, quantize_nparray


@op_registry(op_types="Pad")
class PadOperator(Operator):
    """Pad Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(PadOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        # if opset version is less than 11, just no change
        if self.quantizer.opset_version < 11:  # pragma: no cover
            return False
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0])
        if not self.disable_qdq_for_node_output or self.quantizer.mode != "qdq":
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"

    def convert_check(self, convert_format):
        """Check if conversion can be done."""
        node = self.node
        assert convert_format in ["static"], "convert format for {} should be in ['static']".format(node.op_type)

        children = self.quantizer.model.get_children(node)
        if len(children) == 0 or not node.name.endswith("_quant"):  # pragma: no cover
            return False
        return True

    def convert(self, convert_format):
        """Convert to QOperator format."""
        node = self.node

        parent = self.quantizer.model.get_parents(node)[0]
        child = self.quantizer.model.get_children(node)[0]

        kwargs = {}
        for attribute in node.attribute:
            kv = attribute_to_kwarg(attribute)
            kwargs.update(kv)

        if "mode" not in kwargs or kwargs["mode"] == b"constant":
            if len(node.input) > 2:  # There is 3rd input 'constant_value'
                zp_tensor = self.quantizer.model.get_initializer(parent.input[2])
                scale_tensor = self.quantizer.model.get_initializer(parent.input[1])

                padding_constant_initializer = self.quantizer.model.get_initializer(node.input[2])
                if padding_constant_initializer is not None:
                    zp_array = onnx.numpy_helper.to_array(zp_tensor)
                    zp_value = zp_array.item() if zp_array.ndim == 0 else zp_array[0]
                    scale_array = onnx.numpy_helper.to_array(scale_tensor)
                    scale_value = scale_array.item() if scale_array.ndim == 0 else scale_array[0]
                    padding_constant_array = onnx.numpy_helper.to_array(padding_constant_initializer)
                    quantized_padding_constant_array = quantize_nparray(
                        self.weight_dtype, padding_constant_array, scale_value, zp_value
                    )
                    quantized_padding_constant_name = node.input[2] + "_quantized"
                    quantized_padding_constant_initializer = onnx.numpy_helper.from_array(
                        quantized_padding_constant_array, quantized_padding_constant_name
                    )
                    # Suppose this padding constant initializer only used by the node
                    self.quantizer.model.remove_initializer(padding_constant_initializer)
                    self.quantizer.model.add_initializer(quantized_padding_constant_initializer)
                    node.input[2] = quantized_padding_constant_name
                else:
                    self.quantizer.quantize_inputs(node, [2], False)
                    node.input[2] = node.input[2] + "_DequantizeLinear"
            else:
                # pad zero_point for original zero
                node.input.extend([parent.input[2]])

        # Create an entry for output quantized value
        node.input[0] = parent.input[0]
        node.output[0] = child.output[0]
        self.quantizer.remove_nodes.extend([parent, child])


@qop_registry(op_types="Pad")
class QPadOperator(QOperator):
    """QPad Operator."""

    def __init__(self, onnx_node, children, initializers):
        """Initialization."""
        super().__init__(onnx_node, children, initializers)
