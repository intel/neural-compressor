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
from .qdq_base_operator import QDQOperatorBase

class QDQConcat(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        inits = [i.name for i in self.quantizer.model.initializer()]
        if all([inp not in self.quantizer.quantized_value_map and inp not in inits \
            for inp in node.input]) or \
            not all([inp in self.quantizer.quantized_value_map or inp in inits \
            for inp in node.input]):
            return
        for idx, inp in enumerate(node.input):
            initializer_use_weight_qType = inp not in inits
            self.quantizer.quantize_inputs(node, [idx], initializer_use_weight_qType)
        if not self.disable_qdq_for_node_output or self.quantizer.mode != 'qdq':
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"

class QLinearConcat(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        node = self.node

        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)
        if len(children) == 0 or len(parents) == 0:
            return

        if all([i.op_type == 'DequantizeLinear' for i in parents]) and \
            any([i.op_type == 'QuantizeLinear' for i in children]):
            inputs = []

            inputs.extend([i for i in children if i.op_type == 'QuantizeLinear'][0].input[1:])
            for parent in parents:
                inputs.extend(parent.input)
                self.quantizer.remove_nodes.append(parent)
            for child in children:
                if child.op_type == 'QuantizeLinear':
                    self.quantizer.remove_nodes.append(child)
                    self.quantizer.model.replace_input_of_all_nodes(
                        child.output[0], node.output[0] + '_quantized')
            
            kwargs = {}
            for attribute in node.attribute:
                kwargs.update(attribute_to_kwarg(attribute))
            kwargs["domain"] = ms_domain
            qlconcat_node = onnx.helper.make_node("QLinearConcat", 
                                                  inputs, 
                                                  [node.output[0] + "_quantized"], 
                                                  node.name, 
                                                  **kwargs)        

            self.quantizer.new_nodes += [qlconcat_node]
            self.quantizer.remove_nodes.append(node)
