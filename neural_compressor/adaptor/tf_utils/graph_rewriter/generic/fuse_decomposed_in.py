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
"""Fuse Decomposed InstanceNorm Graph Rewriter."""

import re

from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util
from tensorflow.python.platform import tf_logging

from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper as helper
from neural_compressor.utils.utility import dump_elapsed_time


class FuseDecomposedINOptimizer:  # pragma: no cover
    """Fuse decomposed small ops into InstanceNorm."""

    def __init__(self, input_graph_def):
        """Initialization."""
        self.input_graph_def = input_graph_def

    @dump_elapsed_time("Pass FuseDecomposedINOptimizer")
    def do_transformation(self):
        """Find a group of ops that make up an instance/layer normalization pattern for fusion.

        In some models, the instance normalizatin is performed via a group of individual
        ops instead of using single InstanceNormalization op. This function identifies a
        pattern of instance normalization subgraph which is made of multiple ops and
        transforms the graph by replacing those individual ops with InstanceNormalization op.
        This will provide the opportunity to further fold the InstanceNormalization with
        convolution ops to reduce the computation steps during inference.
        This function currently recognizes instance normalization patterns described
        below, this could be extended if newer patterns are seen. Also, the fusion
        is only attempted if the input graph is in NHWC format or has no format set.

        The following pattern will be searched in the graph with additional
        constraints. Here * means any type of op.
        clang-format off
                     Subgraph for fusion
                     -------------------
          *(input)
           x    x x____________
           x    x              x
           x    x             Mean1                                      FusedOp
           x    x            x    x                                      -------
           x    x           x      x                           *(input)  Const  Const
           x    x          x        x                              x    (gamma) (beta)
           x    x         x          x                              x     x     x
           x    x        x           x                _MklFusedInstanceNorm/_MklLayerNorm
           x    x       x            x
           x   SquaredDiff  Const    x
            x      x      x          x
             x      x    x           x
              x     Mean0  Const     x
               x      x    x         x
                x  AddV2|Add         x
                 x       x    Const  x
                  x    Rsqrt (gamma) x
                   x        x x      x
                    x       Mul1     x
                     x      x   x    x
                      x     x    x   x
                       x    x     x  x
                        x   x Constx x
                         x  x (beta)Mul2
                          x x    x  x
                          Mul0   Sub
                             x   x
                         AddV2|Add(output)
        Args:
            input_graph_def: A GraphDef containing a model.

        Returns:
            Modified graph with individual ops that made up of instance normalization
            fused to InstanceNormalization.

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
            if node.op != "Add" and node.op != "AddV2":
                continue

            # Add (Mul0, Sub) or Add (Sub, Mul0)
            input0_op = node_from_map(input_node_map, node.input[0])
            input1_op = node_from_map(input_node_map, node.input[1])

            if input0_op.op == "Mul" and input1_op.op == "Sub":
                data_scale_mul_op = input0_op
                bias_mean_sub_op = input1_op
            elif input0_op.op == "Sub" and input1_op.op == "Mul":
                bias_mean_sub_op = input0_op
                data_scale_mul_op = input1_op
            else:
                continue

            # Mul0 (*input, Mul1)
            input_data_op = node_from_map(input_node_map, data_scale_mul_op.input[0])
            scale_op = node_from_map(input_node_map, data_scale_mul_op.input[1])

            # Mul1 (Rsqrt, Constant_gamma)
            if scale_op.op == "Mul":
                rsqrt_op = node_from_map(input_node_map, scale_op.input[0])
                gamma_op, gamma_reshape_op = bypass_reshape(input_node_map, scale_op.input[1])
                if rsqrt_op.op != "Rsqrt":
                    continue
                if gamma_op.op != "Const":
                    continue
            else:
                continue

            # Sub (Constant_beta, Mul2)
            beta_op, beta_reshape_op = bypass_reshape(input_node_map, bias_mean_sub_op.input[0])
            mean_scale_mul_op = node_from_map(input_node_map, bias_mean_sub_op.input[1])
            if mean_scale_mul_op.op != "Mul":
                continue
            if beta_op.op != "Const":
                continue

            # Common scale applies to both input and running mean
            if scale_op != node_from_map(input_node_map, mean_scale_mul_op.input[1]):
                continue

            mean_op, mean_reshape_op = bypass_reshape(input_node_map, mean_scale_mul_op.input[0])
            if mean_op.op != "Mean":
                continue

            # Add (variance-mean0, Constant_epsilon)
            variance_epsilon_add_op = node_from_map(input_node_map, rsqrt_op.input[0])
            if variance_epsilon_add_op.op != "Add" and variance_epsilon_add_op.op != "AddV2":
                continue

            variance_op, variance_reshape_op = bypass_reshape(input_node_map, variance_epsilon_add_op.input[0])
            epsilon_op = node_from_map(input_node_map, variance_epsilon_add_op.input[1])
            if epsilon_op.op != "Const":
                continue
            if variance_op.op != "Mean":
                continue

            epsilon = values_from_const(epsilon_op)

            # Mean (SquaredDifference, Constant_r_indices0)
            squared_diff_op, squared_reshape_op = bypass_reshape(input_node_map, variance_op.input[0])
            r_indices0_op = node_from_map(input_node_map, variance_op.input[1])
            if squared_diff_op.op != "SquaredDifference":
                continue
            if r_indices0_op.op != "Const":
                continue

            if input_data_op != node_from_map(input_node_map, squared_diff_op.input[0]):
                continue

            if mean_op != node_from_map(input_node_map, squared_diff_op.input[1]):
                continue

            if input_data_op != node_from_map(input_node_map, mean_op.input[0]):
                continue

            r_indices1_op = node_from_map(input_node_map, mean_op.input[1])
            if r_indices1_op.op != "Const":
                continue

            r_indices1 = values_from_const(r_indices1_op)
            if (
                r_indices1.tolist() != [1, 2]
                and r_indices1.tolist() != [2, 3]
                and r_indices1.tolist() != [1, 2, 3]
                and r_indices1.tolist() != [2, 3, 4]
            ):
                continue

            nodes_to_skip[node.name] = True
            nodes_to_skip[data_scale_mul_op.name] = True
            nodes_to_skip[bias_mean_sub_op.name] = True
            nodes_to_skip[mean_scale_mul_op.name] = True
            nodes_to_skip[scale_op.name] = True
            nodes_to_skip[rsqrt_op.name] = True
            nodes_to_skip[variance_epsilon_add_op.name] = True
            nodes_to_skip[squared_diff_op.name] = True
            nodes_to_skip[mean_op.name] = True
            nodes_to_skip[variance_op.name] = True
            if gamma_reshape_op is not None:
                nodes_to_skip[gamma_reshape_op.name] = True
            if beta_reshape_op is not None:
                nodes_to_skip[beta_reshape_op.name] = True
            if mean_reshape_op is not None:
                nodes_to_skip[mean_reshape_op.name] = True
            if variance_reshape_op is not None:
                nodes_to_skip[variance_reshape_op.name] = True

            if gamma_op is None:
                gamma_op = node_def_pb2.NodeDef()
                gamma_op.op = "Const"
                # Assign name with same root of Rsqrt op's name plus "gamma"
                m = re.search(r"(.*)/(.*)", scale_op.name)
                if m:
                    gamma_op.name = m.group(1) + "/gamma"
                else:
                    gamma_op.name = scale_op.name + "/gamma"
                gamma_op.attr["dtype"].CopyFrom(beta_op.attr["dtype"])
                beta_value = values_from_const(beta_op)
                gamma_op.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            1, beta_value.dtype.type, beta_value.shape, allow_broadcast=True
                        )
                    )
                )
                new_ops.append(gamma_op)

            new_fused_instancenorm_op = node_def_pb2.NodeDef()
            new_fused_instancenorm_op.op = "_MklFusedInstanceNorm"
            new_fused_instancenorm_op.name = node.name
            new_fused_instancenorm_op.attr["T"].CopyFrom(node.attr["T"])
            new_fused_instancenorm_op.attr["epsilon"].CopyFrom(attr_value_pb2.AttrValue(f=epsilon.tolist()))
            list_value = attr_value_pb2.AttrValue.ListValue(i=r_indices1.flatten())
            new_fused_instancenorm_op.attr["reduction_axes"].CopyFrom(attr_value_pb2.AttrValue(list=list_value))

            # Mean and variance values will be computed at runtime for fp32 & bf16 input.
            # Pass a "dummy" node for mean and variance.
            mean_variance_dim = tensor_util.MakeNdarray(gamma_op.attr["value"].tensor).shape[-1]
            dummy_mean_node = helper.create_constant_node(
                node.name + "_dummy_mean", [0.0] * mean_variance_dim, dtypes.float32
            )
            dummy_variance_node = helper.create_constant_node(
                node.name + "_dummy_variance", [1.0] * mean_variance_dim, dtypes.float32
            )
            new_fused_instancenorm_op.input.extend(
                [input_data_op.name, gamma_op.name, beta_op.name, dummy_mean_node.name, dummy_variance_node.name]
            )
            new_ops.append(dummy_mean_node)
            new_ops.append(dummy_variance_node)
            new_ops.append(new_fused_instancenorm_op)

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


def node_name_from_input(node_name):
    """Strips off ports and other decorations to get the underlying node name."""
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        node_name = m.group(1)
    return node_name


def node_from_map(node_map, name):
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


def values_from_const(node_def):
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


def valid_reshape_inputs(reshape_in0_ndef, reshape_in1_ndef):
    """Check if the inputs of the Reshape are valid."""
    if reshape_in0_ndef.op != "Const" or reshape_in1_ndef.op != "Const" or get_const_dim_count(reshape_in0_ndef) != 1:
        return False
    input0_vec_size = values_from_const(reshape_in0_ndef).shape[0]
    const_value = values_from_const(reshape_in1_ndef)
    shape_ndims = const_value.ndim
    if shape_ndims != 1:
        raise ValueError("Num of dims of the shape must be 1, got {}.".format(shape_ndims))
    for value in const_value.tolist()[:-1]:
        if value != 1:
            return False
    if const_value.tolist()[-1] != input0_vec_size:
        return False
    return True


def bypass_reshape(input_node_map, input_name):
    """Get Reshape input nodes."""
    reshape_ndef = None
    maybe_reshape_ndef = node_from_map(input_node_map, input_name)
    input_ndef = maybe_reshape_ndef
    if maybe_reshape_ndef.op == "Reshape":
        reshpae_input0_ndef = node_from_map(input_node_map, maybe_reshape_ndef.input[0])
        reshpae_input1_ndef = node_from_map(input_node_map, maybe_reshape_ndef.input[1])
        if (
            reshpae_input0_ndef.op == "Const"
            and reshpae_input1_ndef.op == "Const"
            and valid_reshape_inputs(reshpae_input0_ndef, reshpae_input1_ndef)
        ):
            input_ndef = reshpae_input0_ndef
            reshape_ndef = maybe_reshape_ndef
    return input_ndef, reshape_ndef


def get_const_dim_count(node_def):
    """Get the number of dimensions for a Const node.

    Args:
        node_def: Const NodeDef.

    Returns:
        Number of dimensions for the Const node.
    """
    const_value = values_from_const(node_def)
    return const_value.ndim
