#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
"""Fuse small ops to LayerNorm Graph Rewriter."""

import re

from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import tensor_util

from neural_compressor.utils.utility import dump_elapsed_time


class FuseLayerNormOptimizer:  # pragma: no cover
    """Remap smaller ops into fused LayerNorm.

    Current fusion is only for the case, when LayerNormalization uses FusedBatcNormV3.
    And further restrict it to only 2D or 3D tensor inputs to keras LayerNormalization api.
    """

    def __init__(self, input_graph_def):
        """Constructor."""
        self.input_graph_def = input_graph_def

    @dump_elapsed_time("Pass FuseLayerNormOptimizer")
    def do_transformation(self):
        """The following pattern will be searched in the graph with additional constraints.

        Here * means any type of op.
        Subgraph:
                *(input)  *  * Const  *  Const                       FusedOp
                     x    |   x  |    |  x        Const              -------
                      x   |    x |    | x  Const   x
                      Reshape  Fill   Fill  x     x         *(input) *(gamma)  *(beta)
                         x      x      x   x     x                x     |      x
                          x    x      x   x     x                  x    |     x
                     F u s e d B a t c h N o r m V 3              _MklLayerNorm
                            x
                             x   *
                              x x
                           Reshape
                               x   *(gamma)
                                x x
                                Mul
                        *(beta) x
                           x   x
                         AddV2(output)
        Args:
            input_graph_def: A GraphDef containing a model.

        Returns:
            Modified graph with individual ops that made up of layer normalization
            fused to LayerNorm.

        Raises:
            ValueError: If the graph is badly formed with duplicate node names.
        """
        input_node_map = {}
        for node in self.input_graph_def.node:
            if node.name not in input_node_map:
                input_node_map[node.name] = node
            else:
                raise ValueError("Duplicate node names detected for ", node.name)

        nodes_to_skip = {}
        new_ops = []
        for node in self.input_graph_def.node:
            if node.op != "AddV2":
                continue

            # AddV2 (Mul, beta) or AddV2 (beta, Mul)
            input0_op = node_from_map(input_node_map, node.input[0])
            input1_op = node_from_map(input_node_map, node.input[1])

            if input0_op.op == "Mul":
                data_scale_mul_op = input0_op
                beta_op = input1_op
            elif input1_op.op == "Mul":
                beta_op = input0_op
                data_scale_mul_op = input1_op
            else:
                continue

            # Mul (Reshape, *gamma)
            input0_op = node_from_map(input_node_map, data_scale_mul_op.input[0])
            input1_op = node_from_map(input_node_map, data_scale_mul_op.input[1])
            if input0_op.op == "Reshape":
                post_reshape_op = input0_op
                gamma_op = input1_op
            elif input1_op.op == "Reshape":
                post_reshape_op = input1_op
                gamma_op = input0_op
            else:
                continue

            # Reshape (FusedBatchNormV3, *post_shape)
            input0_op = node_from_map(input_node_map, post_reshape_op.input[0])
            input1_op = node_from_map(input_node_map, post_reshape_op.input[1])
            if input0_op.op == "FusedBatchNormV3":
                fused_batch_norm_op = input0_op
                post_shape_op = input1_op
            elif input1_op.op == "FusedBatchNormV3":
                fused_batch_norm_op = input1_op
                post_shape_op = input0_op
            else:
                continue

            # LayerNorm uses FusedBatchNorm in training mode.
            if fused_batch_norm_op.attr["is_training"] == attr_value_pb2.AttrValue(b=False):
                continue

            # FusedBatchNormV3(Reshape, Fill, Fill, Mean, Variance)
            pre_reshape_op = node_from_map(input_node_map, fused_batch_norm_op.input[0])
            if pre_reshape_op.op != "Reshape":
                continue
            fill_scale_op = node_from_map(input_node_map, fused_batch_norm_op.input[1])
            if fill_scale_op.op != "Fill":
                continue
            fill_offset_op = node_from_map(input_node_map, fused_batch_norm_op.input[2])
            if fill_offset_op.op != "Fill":
                continue

            # FusedBatchNorm node should have mean/variance as empty constant
            mean_op = node_from_map(input_node_map, fused_batch_norm_op.input[3])
            if mean_op.op != "Const":
                continue
            variance_op = node_from_map(input_node_map, fused_batch_norm_op.input[4])
            if variance_op.op != "Const":
                continue
            mean_value = values_from_const(mean_op)
            if mean_value.any():
                continue
            variance_value = values_from_const(variance_op)
            if variance_value.any():
                continue

            # Reshape (*input, *pre_shape)
            input_op = node_from_map(input_node_map, pre_reshape_op.input[0])
            pre_shape_op = node_from_map(input_node_map, pre_reshape_op.input[1])

            # Fill Scale(*dims_fill_scale, unit_gamma)
            dims_fill_scale_op = node_from_map(input_node_map, fill_scale_op.input[0])
            unit_gamma_op = node_from_map(input_node_map, fill_scale_op.input[1])
            if unit_gamma_op.op != "Const":
                continue

            # Fill Offset(*dims_fill_scale, unit_gamma)
            dims_fill_offset_op = node_from_map(input_node_map, fill_offset_op.input[0])
            zero_beta_op = node_from_map(input_node_map, fill_offset_op.input[1])
            if zero_beta_op.op != "Const":
                continue

            nodes_to_skip[node.name] = True
            nodes_to_skip[data_scale_mul_op.name] = True
            nodes_to_skip[post_reshape_op.name] = True
            nodes_to_skip[fused_batch_norm_op.name] = True
            nodes_to_skip[fill_scale_op.name] = True
            nodes_to_skip[fill_offset_op.name] = True

            new_fused_layernorm_op = node_def_pb2.NodeDef()
            new_fused_layernorm_op.op = "_MklLayerNorm"
            new_fused_layernorm_op.name = node.name
            new_fused_layernorm_op.attr["T"].CopyFrom(node.attr["T"])
            new_fused_layernorm_op.input.extend([input_op.name, gamma_op.name, beta_op.name])

            new_ops.append(new_fused_layernorm_op)

        result_graph_def = graph_pb2.GraphDef()
        for node in self.input_graph_def.node:
            if node.name in nodes_to_skip:
                continue
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(node)
            retained_input = []
            for input_node in new_node.input:
                if not input_node.startswith("^") or input_node[1:] not in nodes_to_skip:
                    retained_input.append(input_node)
            new_node.input[:] = retained_input
            result_graph_def.node.append(new_node)

        result_graph_def.node.extend(new_ops)
        result_graph_def.versions.CopyFrom(self.input_graph_def.versions)
        return result_graph_def


def node_name_from_input(node_name):  # pragma: no cover
    """Strips off ports and other decorations to get the underlying node name."""
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        node_name = m.group(1)
    return node_name


def node_from_map(node_map, name):  # pragma: no cover
    """Pulls a node def from a dictionary for a given name.

    Args:
        node_map: Dictionary containing an entry indexed by name for every node.
        name: Identifies the node we want to find.

    Returns:
        NodeDef of the node with the given name.

    Raises:
        ValueError: If the node isn't present in the dictionary.
    """
    stripped_name = node_name_from_input(name)
    if stripped_name not in node_map:
        raise ValueError("No node named '%s' found in map." % name)
    return node_map[stripped_name]


def values_from_const(node_def):  # pragma: no cover
    """Extracts the values from a const NodeDef as a numpy ndarray.

    Args:
        node_def: Const NodeDef that has the values we want to access.

    Returns:
        Numpy ndarray containing the values.

    Raises:
        ValueError: If the node isn't a Const.
    """
    if node_def.op != "Const":
        raise ValueError("Can not extract constant value from a node that is not Const. Got:\n" f"{node_def}")
    input_tensor = node_def.attr["value"].tensor
    tensor_value = tensor_util.MakeNdarray(input_tensor)
    return tensor_value
