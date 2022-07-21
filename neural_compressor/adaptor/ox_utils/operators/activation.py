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
from .qdq_base_operator import QDQOperatorBase
from onnxruntime.quantization.quant_utils import QuantizedValueType, \
                                                 attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto
from neural_compressor.adaptor.ox_utils.util import QuantizedValue


class QLinearActivation(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        node = self.node
        if node.op_type in ['Relu', 'Clip']:
            return
            
        if len(self.quantizer.model.get_children(node)) == 0:
            return
        # No assert on op_type as it is controlled by registry
        # only try to quantize when given quantization parameters for it
        parent = self.quantizer.model.get_parents(node)[0]
        child = self.quantizer.model.get_children(node)[0]

        inputs = []
        inputs.extend(parent.input)
        inputs.extend(child.input[1:])

        qlinear_activation_output = child.output[0]
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qlinear_activation_node = onnx.helper.make_node(
            "QLinear" + node.op_type, inputs,
            [qlinear_activation_output], node.name, **kwargs)

        self.quantizer.new_nodes.append(qlinear_activation_node)
        self.quantizer.remove_nodes.extend([parent, child, node])

class QDQRemovableActivation(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        if node.input[0] not in self.quantizer.quantized_value_map:
            return
        elif node.output[0] in [i.name for i in self.quantizer.model.model.graph.output]:
            self.quantizer.dequantize_tensor(node, node.input[0])
        else:
            self.quantizer.model.replace_input_of_all_nodes(node.output[0], node.input[0])
            self.quantizer.remove_nodes.append(node)

class QDQActivation(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        data_found, _, _, _, _ = self.quantizer._get_quantization_params(node.output[0])
        if not data_found:
            return
        super().quantize()
        node.name = node.name + "_quant"
