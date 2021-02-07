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

class QuantOperatorBase:
    def __init__(self, onnx_quantizer, onnx_node):
        self.quantizer = onnx_quantizer
        self.node = onnx_node
        if self.node.name in self.quantizer.config:
            self.per_channel = False
            self.algorithm = 'minmax'
            self.scheme = 'sym'
            self.weight_dtype = None
            self.activation_dtype = None
            if self.quantizer.config[self.node.name] != 'fp32':
                if 'weight' in self.quantizer.config[self.node.name].keys():
                    self.per_channel = self.quantizer.config[self.node.name]\
                        ['weight']['granularity'] == 'per_channel'
                    self.algorithm = self.quantizer.config[self.node.name]\
                        ['weight']['algorithm']
                    self.scheme = self.quantizer.config[self.node.name]\
                        ['weight']['scheme']
                    self.weight_dtype = self.quantizer.config[self.node.name]\
                        ['weight']['dtype']
                if 'activation' in self.quantizer.config[self.node.name].keys():
                    self.activation_dtype = self.quantizer.config[self.node.name]\
                        ['activation']['dtype']

    def quantize(self):
        '''
        Given a node which does not support quantization(Conv, Matmul, Gather), this method
        checks whether the input to this node is quantized and adds a DequantizeLinear node
        to dequantize this input back to FP32
            parameter node: Current node
            parameter new_nodes_list: List of new nodes created before processing current node
            return: List of new nodes created
        '''
        nodes = []
        for index, node_input in enumerate(self.node.input):
            dequantize_node = self.quantizer._dequantize_value(node_input)
            if dequantize_node is not None:
                self.quantizer.new_nodes.append(dequantize_node)

        # Append the original node
        self.quantizer.new_nodes.append(self.node)
