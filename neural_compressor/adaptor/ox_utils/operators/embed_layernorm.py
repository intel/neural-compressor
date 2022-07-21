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
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.quant_utils import QuantizedValueType, \
                                                 attribute_to_kwarg, ms_domain
'''
Quantize EmbedLayerNormalization
'''


class EmbedLayerNormalizationQuant(QuantOperatorBase): # pragma: no cover
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        node = self.node
        assert (node.op_type == "EmbedLayerNormalization")

        '''
        Pre-quantization EmbedLayerNorm inputs:
        [0] input_ids (int32)
        [1] segment_ids (int32)
        [2] word_embedding (float32)
        [3] position_embedding (float32)
        [4] segment_embedding (float32)
        [5] gamma (float32)
        [6] beta (float32)
        [7] mask (int32) (optional)
        '''

        parents = self.quantizer.model.get_parents(node)
        inputs = []
        # 'input_ids'
        inputs.extend([node.input[0]])
        # 'segment_ids'
        inputs.extend([node.input[1]])
        for parent in parents:
            inputs.append(parent.input[0])
        # 'mask' (optional)
        inputs.extend([node.input[7] if len(node.input) > 7 else ""])

        for parent in parents:
            inputs.append(parent.input[1])
        for parent in parents:
            inputs.append(parent.input[2])
 
        '''
        Quantized Input Tensor List
        [0] input_ids (int32)
        [1] segment_ids (int32)
        [2] word_embedding (uint8)
        [3] position_embedding (uint8)
        [4] segment_embedding (uint8)
        [5] gamma (uint8)
        [6] beta (uint8)
        [7] mask (int32) (optional)
        [8] word_embedding_scale (float)
        [9] position_embedding_scale (float)
        [10] segment_embedding_scale (float)
        [11] gamma_scale (float)
        [12] beta_scale (float)
        [13] word_embedding_zero_point (uint8)
        [14] position_embedding_zero_point (uint8)
        [15] segment_embedding_zero_point (uint8)
        [16] gamma_zero_point (uint8)
        [17] beta_zero_point (uint8)
        '''
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qembed_layer_norm_node = onnx.helper.make_node("QEmbedLayerNormalization", 
                                                       inputs, node.output,
                                                       node.name, **kwargs)
        self.quantizer.new_nodes.append(qembed_layer_norm_node)
        self.quantizer.remove_nodes.extend(parents)

class QDQEmbedLayerNormalization(QDQOperatorBase): # pragma: no cover
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "EmbedLayerNormalization")
        self.quantizer.quantize_inputs(node, [2, 3, 4, 5, 6])
        node.name = node.name + "_quant"
