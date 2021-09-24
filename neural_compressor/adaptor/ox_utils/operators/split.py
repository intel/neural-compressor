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

import onnx
from onnxruntime.quantization.quant_utils import QuantizedValueType, \
                                                 attribute_to_kwarg
from .base_operator import QuantOperatorBase 
from neural_compressor.adaptor.ox_utils.util import QuantizedValue

class QSplit(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        quantized_input_names, zero_point_names, scale_names, nodes = \
                                           self.quantizer.quantize_inputs(node, [0])
        quantized_node_name = ""
        if node.name != "":
            quantized_node_name = node.name + "_quant"
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))

        # Output just derive the scale/zero from input
        quantized_output_names = []
        for output_name in node.output:
            quantized_output_name = output_name + "quantized"
            quantized_output_names.append(quantized_output_name)
            q_output = QuantizedValue(output_name, quantized_output_name, 
                                      scale_names[0], zero_point_names[0],
                                      QuantizedValueType.Input)
            self.quantizer.quantized_value_map[output_name] = q_output

        if len(node.input) > 1:
            quantized_input_names = quantized_input_names.extend(node.input[1:])
        quantized_node = onnx.helper.make_node(node.op_type, 
                                               quantized_input_names, 
                                               quantized_output_names,
                                               quantized_node_name, **kwargs)

        nodes.append(quantized_node)
        self.quantizer.new_nodes += nodes
