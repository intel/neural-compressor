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
from onnxruntime.quantization.quant_utils import attribute_to_kwarg, ms_domain, \
                                                 QuantizedValueType
from onnx import onnx_pb as onnx_proto
from neural_compressor.adaptor.ox_utils.util import QuantizedValue

class QLinearBinaryOp(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        node = self.node
        if len(self.quantizer.model.get_children(node)) == 0:
            return
        parents = self.quantizer.model.get_parents(node)
        if all([i.op_type != 'DequantizeLinear' for i in parents]):
            return
        child = self.quantizer.model.get_children(node)[0]

        qlinear_binary_math_output = child.output[0]

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qlinear_binary_math_inputs = []
        for parent in parents:
            qlinear_binary_math_inputs.extend(parent.input)
        qlinear_binary_math_inputs.extend(child.input[1:])

        qlinear_binary_math_node = onnx.helper.make_node("QLinear" + node.op_type, 
                                                         qlinear_binary_math_inputs,
                                                         [qlinear_binary_math_output], 
                                                         node.name,
                                                         **kwargs)

        self.quantizer.new_nodes += [qlinear_binary_math_node]
        self.quantizer.remove_nodes.extend(parents)
        self.quantizer.remove_nodes.append(child)
        self.quantizer.remove_nodes.append(node)

class QDQBinaryOp(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        data_found, _, _, _, _ = self.quantizer._get_quantization_params(node.output[0])
        if not data_found:
            return
 
        self.quantizer.quantize_inputs(node, initializer_use_weight_qType=False)
        if not self.disable_qdq_for_node_output or self.quantizer.mode != 'qdq':
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"
