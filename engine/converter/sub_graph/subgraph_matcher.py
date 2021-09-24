#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .pattern import supported_patterns, PATTERNS
from neural_compressor.utils import logger

EXECUTOR_TYPE = {
    "MatMulWithBias": "InnerProduct",
    "MatMulWithBiasAdd": "InnerProduct",
    "MatMulWithBiasGelu": "InnerProduct",
    "MatMulWithBiasTanh": "InnerProduct",
    "MatMulWithBiasRelu": "InnerProduct",
    "MatMul": "InnerProduct",
    "QuantizedMatMulWithBiasAndDequantize": "InnerProduct",
    "TransposeBatchMatMul": "Matmul",
    "BatchMatMul": "Matmul",
    "BatchMatMulV2": "Matmul",
    "Add": "BinaryAdd",
    "AddV2": "BinaryAdd",
    "AddWithAdd": "BinaryAdd",
    "QLinearAdd": "BinaryAdd",
    "Transpose": "Reorder",
    "GatherV2": "Gather",
    "ExpandDimsToReshape": "Reshape",
    "QuantizeV2": "Quantize",
    "QuantizeLinear": "Quantize",
    "OneHot": "Onehot",
    "LayerNormalization": "LayerNorm",
    "FusedGemm": "InnerProduct",
    "_QuantizedFusedMatMulAndDequantize": "InnerProduct",
    "_FusedMatMul": "InnerProduct",
    "_MklLayerNorm": "LayerNorm",
}


class SubGraphMatcher(object):
    def __call__(self, model):
        patterns_switch = {
            'LayerNorm': True,
            'TransposeBatchMatMul': True,
            'MatMulWithBiasGelu': True,
            'MatMulWithBiasAdd': True,
            'MatMulWithBiasTanh': True,
        }
        logger.info('Start to implement Sub-Graph matching and replacing...')
        for pattern in supported_patterns:
            if pattern in PATTERNS:
                if pattern in patterns_switch.keys() and not patterns_switch[pattern]:
                    continue
                else:
                    p_fusion = PATTERNS[pattern]()
                    model = p_fusion(model)

        rm_node_names = []
        rm_op_type = ['Identity']
        for i in range(len(model.nodes)):
            node = model.nodes[i]
            if node.op_type in rm_op_type:
                rm_node_names.append(node.name)
            else:
                if node.op_type in EXECUTOR_TYPE.keys():
                    op_type = EXECUTOR_TYPE[node.op_type]
                    model.nodes[i].op_type = op_type
        model.remove_nodes(rm_node_names)
        logger.info('Sub-Graph match and replace done...')

        return model
