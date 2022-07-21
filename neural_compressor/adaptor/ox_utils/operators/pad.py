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

import numpy
import onnx
from onnxruntime.quantization.quant_utils import QuantizedValueType, \
                                                 attribute_to_kwarg
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase
from neural_compressor.adaptor.ox_utils.util import QuantizedValue

class QDQPad(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Pad")

        # Only after version 11, it has the optional constant_value
        # If input[0] is not quantized, do not quanitize this node
        if self.quantizer.opset_version < 11:
            return

        self.quantizer.quantize_inputs(node, [0])
        if not self.disable_qdq_for_node_output or self.quantizer.mode != 'qdq':
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"

class QPad(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        node = self.node
        assert (node.op_type == "Pad")

        # Only after version 11, it has the optional constant_value
        # If input[0] is not quantized, do not quanitize this node
        if self.quantizer.opset_version < 11:
            return

        if len(self.quantizer.model.get_children(node)) == 0:
            return
        parent = self.quantizer.model.get_parents(node)[0]
        child = self.quantizer.model.get_children(node)[0]

        kwargs = {}
        for attribute in node.attribute:
            kv = attribute_to_kwarg(attribute)
            kwargs.update(kv)

        if 'mode' not in kwargs or kwargs['mode'] == b'constant':
            if len(node.input) > 2:  # There is 3rd input 'constant_value'
                zp_tensor = self.quantizer.model.get_initializer(parent.input[2])
                scale_tensor = \
                            self.quantizer.model.get_initializer(parent.input[1])

                padding_constant_initializer = self.quantizer.model.get_initializer(node.input[2])
                if padding_constant_initializer is not None:
                    zp_array = onnx.numpy_helper.to_array(zp_tensor)
                    zp_value = zp_array.item() if zp_array.ndim == 0 else zp_array[0]
                    scale_array = onnx.numpy_helper.to_array(scale_tensor)
                    scale_value = scale_array.item() if scale_array.ndim == 0 else scale_array[0]
                    padding_constant_array = \
                                          onnx.numpy_helper.to_array(padding_constant_initializer)
                    quantized_padding_constant_array = quantize_nparray(
                                                    self.weight_dtype,
                                                    padding_constant_array, scale_value, zp_value)
                    quantized_padding_constant_name = node.input[2] + "_quantized"
                    quantized_padding_constant_initializer = onnx.numpy_helper.from_array(
                        quantized_padding_constant_array, quantized_padding_constant_name)
                    # Suppose this padding constant initializer only used by the node
                    self.quantizer.model.remove_initializer(padding_constant_initializer)
                    self.quantizer.model.add_initializer(quantized_padding_constant_initializer)
                    node.input[2] = quantized_padding_constant_name
                else:
                    self.quantizer.quantize_inputs(node, [2], False)
                    node.input[2] = node.input[2] + '_DequantizeLinear'
            else:
                # pad zero_point for original zero
                node.input.extend([parent.input[2]])

        # Create an entry for output quantized value

        node.input[0] = parent.input[0]
        node.output[0] = child.output[0]
        self.quantizer.remove_nodes.extend([parent, child])

def quantize_nparray(qtype, arr, scale, zero_point, low=None, high=None):
    dtype = numpy.uint8 if qtype == "uint8" else numpy.int8
    cliplow = max(0 if dtype == numpy.uint8 else -127, -127 if low is None else low)
    cliphigh = min(255 if dtype == numpy.uint8 else 127, 255 if high is None else high)
    arr_fp32 = numpy.asarray((arr.astype(numpy.float32) / scale).round() + zero_point)
    numpy.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)
    return arr_fp32.astype(dtype)
