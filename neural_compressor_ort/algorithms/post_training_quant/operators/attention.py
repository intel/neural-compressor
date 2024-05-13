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
"""Attention operator."""

import onnx

from neural_compressor_ort.algorithms.post_training_quant.operators.ops import op_registry, Operator
from neural_compressor_ort.algorithms.post_training_quant.utils import attribute_to_kwarg, find_by_name, ms_domain
from neural_compressor_ort.common.utils import DYNAMIC_QUANT, STATIC_QUANT


@op_registry(op_types="Attention", mode=[DYNAMIC_QUANT, STATIC_QUANT])
class AttentionOperator(Operator):
    """Attention operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(AttentionOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0, 1, 2])
        node.name = node.name + "_quant"

    def convert(self):
        """Convert QDQ mode to QOperator format."""
        node = self.node
        parents = self.quantizer.model.get_parents(node)
        quantized_name = []
        scale = []
        zp = []
        for parent in parents[:2]:
            if parent.op_type == "DynamicQuantizeLinear":
                quantized_name.append(parent.output[0])
                scale.append(parent.output[1])
                zp.append(parent.output[2])
            elif parent.op_type == "DequantizeLinear":
                quantized_name.append(parent.input[0])
                scale.append(parent.input[1])
                zp.append(parent.input[2])
                self.quantizer.remove_nodes.append(parent)

        inputs = []
        inputs.extend(quantized_name)
        inputs.append(node.input[2])
        inputs.extend(scale)
        inputs.append(node.input[3] if len(node.input) > 3 else "")
        inputs.extend(zp)
        if len(node.input) > 4:
            inputs.append(node.input[4])

        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qattention_node = onnx.helper.make_node("QAttention", inputs, node.output, node.name, **kwargs)
        self.quantizer.new_nodes.append(qattention_node)

        self.quantizer.remove_nodes.append(node)
