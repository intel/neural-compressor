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


import itertools
from .base_operator import QuantOperatorBase
from onnxruntime.quantization.quant_utils import QuantizedValue, QuantizedValueType, \
                                                    attribute_to_kwarg, quantize_nparray


class QDQOperatorBase(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)
        self.quantizer = onnx_quantizer
        self.node = onnx_node
        self.disable_qdq_for_node_output = True if onnx_node.op_type in \
            onnx_quantizer.op_types_to_exclude_output_quantization else False

    def quantize(self):
        node = self.node

        if self.disable_qdq_for_node_output:
            nodes_to_iterate = node.input
        else:
            nodes_to_iterate = itertools.chain(node.input, node.output)

        for tensor_name in nodes_to_iterate:
            self.quantizer.quantize_tensor(tensor_name)
