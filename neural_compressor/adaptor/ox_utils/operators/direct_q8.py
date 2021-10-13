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

from .base_operator import QuantOperatorBase
from neural_compressor.adaptor.ox_utils.util import QuantizedValue

# For operators that support 8bits operations directly, and output could
# reuse input[0]'s type, zeropoint, scale; For example,Transpose, Reshape, etc.
class Direct8BitOp(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        # Quantize when input[0] is quantized already. Otherwise keep it.
        if node.input[0] not in self.quantizer.quantized_value_map:
            self.quantizer.new_nodes += [node]
            return
            
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]
        quantized_output_value = QuantizedValue(node.output[0], node.output[0] + "_quantized",
                                                quantized_input_value.scale_name, 
                                                quantized_input_value.zp_name,
                                                quantized_input_value.value_type)
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value
        node.input[0] = node.input[0] + "_quantized"
        node.output[0] = node.output[0] + "_quantized"

        self.quantizer.new_nodes += [node]
