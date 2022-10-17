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
from neural_compressor.adaptor.ox_utils.operators.ops import op_registry, Operator
from neural_compressor.adaptor.ox_utils.util import find_by_name, ms_domain, \
    attribute_to_kwarg, is_B_transposed

@op_registry(op_types="Gemm")
class GemmOperator(Operator):
    def __init__(self, onnx_quantizer, onnx_node):
        super(GemmOperator, self).__init__(onnx_quantizer, onnx_node)
    
    def quantize_check(self):
        node = self.node
        if len(node.input) == 3 and \
            not find_by_name(node.input[2], self.quantizer.model.initializer()):
            from neural_compressor.utils import logger
            logger.warning("Bias of Gemm node '{}' is not constant. " \
                "Exclude this node can get better performance.".format(node.name))
            if self.quantizer.mode != 'qdq':
                return False
        return True

    def quantize(self):
        node = self.node
        self.quantizer.quantize_inputs(node, [0])
        if self.per_channel and find_by_name(node.input[1], self.quantizer.model.initializer()):
            self.quantizer.quantize_weights_per_channel(node, [1],
                self.weight_dtype, self.weight_scheme, 0 if is_B_transposed(node) else 1)
        else:
            self.quantizer.quantize_inputs(node, [1])

        if len(node.input) == 3 and \
            find_by_name(node.input[2], self.quantizer.model.initializer()):
            self.quantizer.quantize_bias_tensor(node)
            beta_attribute = [attr for attr in node.attribute if attr.name == "beta"]
            if len(beta_attribute):
                beta_attribute[0].f = 1.0

        if not self.disable_qdq_for_node_output or self.quantizer.mode != 'qdq':
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"

    def convert_check(self, convert_format):
        node = self.node
        assert convert_format in ['static'], \
            "convert format for {} should be in ['static']".format(node.op_type)
        
        children = self.quantizer.model.get_children(node)
        if len(children) == 0 or not node.name.endswith('_quant'):
            return False
        return True

    def convert(self, convert_format):
        node = self.node
        
        parents = self.quantizer.model.get_parents(node)
        child = self.quantizer.model.get_children(node)[0]
        qgemm_output = child.output[0]
        qgemm_inputs = []
        for parent in parents[:-1]:
            qgemm_inputs.extend(parent.input)
        qgemm_inputs.append(parents[-1].input[0])
        qgemm_inputs.extend(child.input[1:])

        kwargs = {}
        for attribute in node.attribute:
            if attribute.name != "beta":
                kwargs.update(attribute_to_kwarg(attribute))
                kwargs["domain"] = ms_domain

        qgemm_node = onnx.helper.make_node("QGemm", 
            qgemm_inputs, [qgemm_output], node.name, **kwargs)

        self.quantizer.new_nodes.append(qgemm_node)
        self.quantizer.remove_nodes.extend(parents)
        self.quantizer.remove_nodes.append(child)
        self.quantizer.remove_nodes.append(node)