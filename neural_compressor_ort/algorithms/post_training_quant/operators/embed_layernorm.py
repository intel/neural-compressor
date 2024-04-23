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
"""EmbedLayerNormalization Operator."""

import onnx

from neural_compressor_ort.algorithms.post_training_quant.operators.ops import op_registry, Operator
from neural_compressor_ort.algorithms.post_training_quant.utils.import attribute_to_kwarg, find_by_name, ms_domain
from neural_compressor_ort.common.utils import DYNAMIC_QUANT, STATIC_QUANT


@op_registry(op_types="EmbedLayerNormalization", mode=[DYNAMIC_QUANT])
class EmbedLayerNormalizationOperator(Operator):
    """EmbedLayerNormalization Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(EmbedLayerNormalizationOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [2, 3, 4, 5, 6])
        node.name = node.name + "_quant"

    def convert(self, convert_format):
        """Convert to QOperator format."""
        node = self.node

        parents = [i for i in self.quantizer.model.get_parents(node) if i.op_type == "DequantizeLinear"]
        inputs = []
        # 'input_ids'
        inputs.extend([node.input[0]])
        # 'segment_ids'
        inputs.extend([node.input[1]])
        for parent in parents:
            inputs.append(parent.input[0])
        # 'mask' (optional)
        if len(node.input) > 7:
            inputs.append(node.input[7])

        for parent in parents:
            inputs.append(parent.input[1])
        for parent in parents:
            inputs.append(parent.input[2])

        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qembed_layer_norm_node = onnx.helper.make_node(
            "QEmbedLayerNormalization", inputs, node.output, node.name, **kwargs
        )
        self.quantizer.new_nodes.append(qembed_layer_norm_node)
        self.quantizer.remove_nodes.extend(parents)
        self.quantizer.remove_nodes.append(node)