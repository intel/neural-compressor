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
"""Helper functions to export onnx model from QLinear ops to QDQ."""
from neural_compressor.adaptor.ox_utils.util import find_by_name
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

numpy_helper = LazyImport("onnx.numpy_helper")


def check_model(model):
    """Check optype for input model.

    Args:
        model (ModelProto): onnx model.
    """
    has_integerop = False
    has_qlinearop = False
    for node in model.graph.node:
        if node.op_type.endswith("Integer"):
            has_integerop = True
        elif node.op_type.startswith("QLinear"):
            has_qlinearop = True
        elif node.op_type in ["QAttention", "QGemm", "QEmbedLayerNormalization"]:
            has_qlinearop = True
        elif node.op_type in ["Gather"]:
            input_data = find_by_name(node.input[0], model.graph.initializer)
            if input_data is not None and numpy_helper.to_array(input_data).dtype in ["int8", "uint8"]:
                has_qlinearop = True
    if has_integerop:
        logger.info("This model has Integer ops, these ops will be skipped.")
    if has_qlinearop:
        return True
    else:
        logger.info("This model has no QLinear ops, save the original model.")
        return False


def onnx_qlinear_to_qdq(
    model,
    input_name_to_nodes,
):
    """Export ONNX QLinearops model into QDQ model.

    Args:
        model (ModelProto): int8 onnx model.
        input_name_to_nodes (dict): the mapping of tensor name and its destination nodes.
    """
    from neural_compressor.adaptor.ox_utils.operators import QOPERATORS

    add_nodes = []
    remove_nodes = []
    inits = []
    if check_model(model):
        for node in model.graph.node:
            if node.op_type in QOPERATORS:
                if node.output[0] not in input_name_to_nodes:
                    continue
                children = []
                for out in node.output:
                    children.extend(input_name_to_nodes[node.output[0]])
                converter = QOPERATORS[node.op_type](node, children, model.graph.initializer)
                done, add_node, init = converter.convert()
                if done:
                    add_nodes.extend(add_node)
                    inits.extend(init)
                    remove_nodes.append(node)
    return add_nodes, remove_nodes, inits
