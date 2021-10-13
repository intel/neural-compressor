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


class QLinearPool(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        # only try to quantize when given quantization parameters for it
        data_found, output_scale_name, output_zp_name, _, _ = \
            self.quantizer._get_quantization_params(node.output[0])
        if (not data_found):
            return super().quantize()

        # get quantized input tensor names, quantize input if needed
        quantized_input_names, input_zero_point_names, input_scale_names, nodes = \
            self.quantizer.quantize_inputs(node, [0])

        # Create an entry for output quantized value.
        qlinear_output_name = node.output[0] + "_quantized"
        quantized_output_value = QuantizedValue(
            node.output[0], qlinear_output_name, output_scale_name, output_zp_name, QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        # Create qlinear pool node for given type (AveragePool, etc)
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qlinear_node_name = node.name + "_quant" if node.name != "" else ""
        inputs = [quantized_input_names[0], input_scale_names[0], input_zero_point_names[0], \
                 output_scale_name, output_zp_name]
        qnode = onnx.helper.make_node(
            "QLinear" + node.op_type,
            inputs,
            [qlinear_output_name],
            qlinear_node_name,
            **kwargs)

        # add all newly created nodes
        nodes.append(qnode)
        self.quantizer.new_nodes += nodes
