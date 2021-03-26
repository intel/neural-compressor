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
from onnxruntime.quantization.quant_utils import QuantizedValue, QuantizedValueType, \
                                                 attribute_to_kwarg
from .base_operator import QuantOperatorBase


class QPad(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Pad")

        # Only after version 11, it has the optional constant_value
        # If input[0] is not quantized, do not quanitize this node
        if (self.quantizer.opset_version < 11) or (node.input[0] not \
                                               in self.quantizer.quantized_value_map):
            super().quantize()
            return
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]

        kwargs = {}
        for attribute in node.attribute:
            kv = attribute_to_kwarg(attribute)
            kwargs.update(kv)

        if 'mode' not in kwargs or kwargs['mode'] == b'constant':
            if len(node.input) > 2:  # There is 3rd input 'constant_value'
                zp_tensor = self.quantizer.model.get_initializer(quantized_input_value.zp_name)
                scale_tensor = \
                            self.quantizer.model.get_initializer(quantized_input_value.scale_name)
                # if zp_tensor is None or scale_tensor is None:
                #     super().quantize()
                #     return

                padding_constant_initializer = self.quantizer.model.get_initializer(node.input[2])
                if padding_constant_initializer is not None:
                    zp_array = onnx.numpy_helper.to_array(zp_tensor)
                    zp_value = zp_array.item() if zp_array.ndim == 0 else zp_array[0]
                    scale_array = onnx.numpy_helper.to_array(scale_tensor)
                    scale_value = scale_array.item() if scale_array.ndim == 0 else scale_array[0]
                    padding_constant_array = \
                                          onnx.numpy_helper.to_array(padding_constant_initializer)
                    quantized_padding_constant_array = quantize_nparray(
                                                    self.activation_dtype,
                                                    padding_constant_array, scale_value, zp_value)
                    quantized_padding_constant_name = node.input[2] + "_quantized"
                    quantized_padding_constant_initializer = onnx.numpy_helper.from_array(
                        quantized_padding_constant_array, quantized_padding_constant_name)
                    # Suppose this padding constant initializer only used by the node
                    self.quantizer.model.remove_initializer(padding_constant_initializer)
                    self.quantizer.model.add_initializer(quantized_padding_constant_initializer)
                    node.input[2] = quantized_padding_constant_name
                else:
                    pad_value_qnodes = self.quantizer._get_quantize_input_nodes(node, 2, 
                                                                 self.activation_dtype)
                    self.quantizer.new_nodes += pad_value_qnodes
                    node.input[2] = pad_value_qnodes[0].output[0]
            else:
                # pad zero_point for original zero
                node.input.extend([quantized_input_value.zp_name])

        # Create an entry for output quantized value
        quantized_output_value = QuantizedValue(node.output[0], node.output[0] + "_quantized",
                                                quantized_input_value.scale_name, 
                                                quantized_input_value.zp_name,
                                                QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        node.input[0] = quantized_input_value.q_name
        node.output[0] = quantized_output_value.q_name
        self.quantizer.new_nodes += [node]

def quantize_nparray(qtype, arr, scale, zero_point, low=None, high=None):
    dtype = numpy.uint8 if qtype == "uint8" else numpy.int8
    cliplow = max(0 if dtype == numpy.uint8 else -127, -127 if low is None else low)
    cliphigh = min(255 if dtype == numpy.uint8 else 127, 255 if high is None else high)
    arr_fp32 = numpy.asarray((arr.astype(numpy.float32) / scale).round() + zero_point)
    numpy.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)
    return arr_fp32.astype(dtype)
