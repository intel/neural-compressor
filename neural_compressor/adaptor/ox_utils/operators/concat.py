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
from .base_operator import QuantOperatorBase
from onnxruntime.quantization.quant_utils import QuantizedValueType, \
        attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto
from neural_compressor.adaptor.ox_utils.util import QuantizedValue


class QLinearConcat(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        if all([item not in self.quantizer.quantized_value_map for item in node.input]):
            self.quantizer.new_nodes += [node]
            return

        if not all([item in self.quantizer.quantized_value_map for item in node.input]):
            super().quantize()
            return

        data_found, output_scale_name, output_zp_name, _, _ = \
                    self.quantizer._get_quantization_params(node.output[0])
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]
        quantized_output_value = QuantizedValue(node.output[0], node.output[0] + "_quantized",
                                                output_scale_name, output_zp_name,
                                                quantized_input_value.value_type)
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        (q_input_names, zero_point_names, scale_names, nodes) = self.quantizer.quantize_inputs(
                                                            node, [*range(0, len(node.input))])

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qnode_name = node.name + "_quant" if node.name != "" else ""

        qlconcat_inputs = [output_scale_name, output_zp_name]
        for i in range(0, len(q_input_names)):
            qlconcat_inputs.extend([q_input_names[i], scale_names[i], zero_point_names[i]])
        qlconcat_node = onnx.helper.make_node("QLinearConcat", 
                                              qlconcat_inputs, 
                                              [node.output[0] + "_quantized"], 
                                              qnode_name, 
                                              **kwargs)        

        self.quantizer.new_nodes += nodes
        self.quantizer.new_nodes += [qlconcat_node]
