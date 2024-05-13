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
"""ArgMax operator."""

from neural_compressor_ort.algorithms.post_training_quant.operators.ops import op_registry, Operator
from neural_compressor_ort.algorithms.post_training_quant.utils import attribute_to_kwarg, find_by_name, ms_domain
from neural_compressor_ort.common.utils import DYNAMIC_QUANT, STATIC_QUANT


@op_registry(op_types="ArgMax", mode=[STATIC_QUANT])
class ArgMaxOperator(Operator):
    """ArgMax operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(ArgMaxOperator, self).__init__(onnx_quantizer, onnx_node)

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node
        return True

    def convert(self):
        """Convert to quantized format."""
        node = self.node
        origin_name = node.input[0].split("_argmax_node")[0]

        if origin_name in self.quantizer.quantized_value_map:
            node.name = node.name + "_quant"
