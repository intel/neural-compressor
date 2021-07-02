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
from lpot.adaptor.ox_utils.util import QuantizedValue


class QLinearActivation(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def QuantizeClipRelu(self):
        node = self.node
        assert (node.op_type == "Relu" or node.op_type == 'Clip')

        # When mode is QLinearOps, the output quantization params are calculated 
        # based on outputs from activation nodes, therefore these nodes can be 
        # removed from the graph if they follow a quantized op.
        # If input to this node is not quantized then keep this node
        if node.input[0] not in self.quantizer.quantized_value_map:
            self.quantizer.new_nodes += [node]
            return

        quantized_value = self.quantizer.quantized_value_map[node.input[0]]
        self.quantizer.quantized_value_map[node.output[0]] = quantized_value

    def quantize(self):
        node = self.node
        if node.op_type == "Relu" or node.op_type == 'Clip':
            self.QuantizeClipRelu()
            return

        # No assert on op_type as it is controlled by registry
        # only try to quantize when given quantization parameters for it
        data_found, output_scale_name, output_zp_name, _, _ = self.quantizer.\
                                       _get_quantization_params(node.output[0])
        if not data_found:
            super().quantize()
            return

        quantized_input_names, zero_point_names, scale_names, nodes = self.quantizer.\
                                                            quantize_inputs(node, [0])

        qlinear_activation_output = node.output[0] + "_quantized"
        qlinear_activation_name = ""
        if node.name != "":
            qlinear_activation_name = node.name + "_quant"
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qlinear_activation_inputs = [quantized_input_names[0], scale_names[0], 
                                     zero_point_names[0], output_scale_name, output_zp_name]

        qlinear_activation_node = onnx.helper.make_node(
            "QLinear" + node.op_type, qlinear_activation_inputs,
            [qlinear_activation_output], qlinear_activation_name, **kwargs)

        # Create an entry for this quantized value
        q_output = QuantizedValue(node.output[0], qlinear_activation_output, output_scale_name,
                                  output_zp_name, QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        nodes.append(qlinear_activation_node)
        self.quantizer.new_nodes += nodes

