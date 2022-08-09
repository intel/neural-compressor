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
        self.disable_qdq_for_node_output = True if onnx_node.op_type in \
            onnx_quantizer.op_types_to_exclude_output_quantization else False
        self.per_channel = False
        self.algorithm = 'minmax'
        self.weight_scheme = 'sym'
        self.weight_dtype = None
        self.activation_dtype = None
        self.activation_scheme = 'asym'
        if self.node.name in self.quantizer.config:
            if self.quantizer.config[self.node.name] != 'fp32':
                if 'weight' in self.quantizer.config[self.node.name].keys():
                    self.per_channel = self.quantizer.config[self.node.name]\
                        ['weight']['granularity'] == 'per_channel'
                    self.algorithm = self.quantizer.config[self.node.name]\
                        ['weight']['algorithm']
                    self.weight_scheme = self.quantizer.config[self.node.name]\
                        ['weight']['scheme']
                    self.weight_dtype = self.quantizer.config[self.node.name]\
                        ['weight']['dtype']
                if 'activation' in self.quantizer.config[self.node.name].keys():
                    self.activation_dtype = self.quantizer.config[self.node.name]\
                        ['activation']['dtype']
                    self.activation_scheme = self.quantizer.config[self.node.name]\
                        ['activation']['scheme']

    def convert(self):
        '''
        Given a node which does not support quantization(Conv, Matmul, Gather), this method
        checks whether the input to this node is quantized and adds a DequantizeLinear node
        to dequantize this input back to FP32
            parameter node: Current node
            parameter new_nodes_list: List of new nodes created before processing current node
            return: List of new nodes created
        '''
        return

