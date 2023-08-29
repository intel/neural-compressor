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
"""Folding BatchNorm Graph Rewriter."""

import math

import numpy as np
from tensorflow.core.framework import attr_value_pb2, node_def_pb2
from tensorflow.python.framework import tensor_util

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class FoldBatchNormNodesOptimizer(GraphRewriterBase):
    """Folding BatchNorm nodes into Conv."""

    INPUT_ORDER = {
        # Order of inputs for BatchNormWithGlobalNormalization.
        "BatchNormWithGlobalNormalization": ["conv_op", "mean_op", "var_op", "beta_op", "gamma_op"],
        # Order of inputs for FusedBatchNorm.
        "FusedBatchNorm": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"],
        "FusedBatchNormV3": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"],
        "_FusedBatchNormEx": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"],
    }
    # Name of the attribute epsilon value is stored in.
    EPSILON_ATTR = {
        "BatchNormWithGlobalNormalization": "variance_epsilon",
        "FusedBatchNorm": "epsilon",
        "FusedBatchNormV3": "epsilon",
        "_FusedBatchNormEx": "epsilon",
    }

    def scale_after_normalization(self, node):
        """Check the scale_after_normalization attribute if the node is BatchNormWithGlobalNormalization.

        Args:
            node (nodedef): input nodedef object

        Returns:
            bool: True if the node op is not BatchNormWithGlobalNormalization else it
                    depends on the BatchNormWithGlobalNormalization attribute value of
                    `scale_after_normalization`.
        """
        if node.op == "BatchNormWithGlobalNormalization":
            return node.attr["scale_after_normalization"].b
        return True

    @dump_elapsed_time("Pass FoldBatchNormNodesOptimizer")
    def do_transformation(self):
        """Removes batch normalization ops by folding them into convolutions.

        Batch normalization during training has multiple dynamic parameters that are
        updated, but once the graph is finalized these become constants. That means
        there's an opportunity to reduce the computations down to a scale and
        addition, rather than the more expensive multiple ops, and even bake the
        scaling into the convolution weights. This function identifies the typical
        pattern of batch normalization subgraphs, and performs the transformation to
        fold the computations down into a simpler form. It currently only spots batch
        normalization that's performed by the BatchNormWithGlobalNormalization, FusedBatchNorm,
        FusedBatchNormV3 and _FusedBatchNormEx ops, and will need to be extended in the future to handle the
        newer style.

        Returns:
          Modified graph with BN ops removed, and modified weights.

        Raises:
          ValueError: If the graph is badly formed with duplicate node names.
        """
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        target_nodes = cur_graph.query_fusion_pattern_nodes(
            [
                ["Conv2D", "DepthwiseConv2dNative"],
                ("BiasAdd", "Add", "AddV2"),
                ["BatchNormWithGlobalNormalization", "FusedBatchNorm", "FusedBatchNormV3", "_FusedBatchNormEx"],
            ]
        )
        for node_combination in target_nodes:
            matched_node = node_combination[:-1]
            has_add_op = True if len(node_combination[-1]) == 3 else False
            conv_node = graph_info[Helper.node_name_from_input(matched_node[0])].node
            weights_node_name = graph_info[Helper.node_name_from_input(matched_node[0])].node.input[1]
            weights_node = graph_info[Helper.node_name_from_input(weights_node_name)].node
            bn_node = graph_info[Helper.node_name_from_input(matched_node[-1])].node

            # oneDNN enabled _FusedBatchNormEx only supports num_side_inputs == 0
            # and Relu/Identity activations.
            if bn_node.op == "_FusedBatchNormEx":
                if bn_node.attr["num_side_inputs"].i != 0:
                    continue
                if not (
                    bn_node.attr["activation_mode"].s == b"Identity" or bn_node.attr["activation_mode"].s == b"Relu"
                ):
                    continue

            if weights_node.op != "Const":
                self.logger.warning(
                    "Didn't find expected conv Constant input to '%s', "
                    "found %s instead. Maybe freeze_graph wasn't "
                    "run first?" % (bn_node.name, weights_node_name)
                )
                continue
            weights = Helper.values_from_const(weights_node)

            if conv_node.op == "Conv2D":
                channel_count = weights.shape[3]
            elif conv_node.op == "DepthwiseConv2dNative":
                channel_count = weights.shape[2] * weights.shape[3]

            mean_node_name = Helper.node_name_from_input(bn_node.input[self.INPUT_ORDER[bn_node.op].index("mean_op")])
            mean_node = graph_info[mean_node_name].node

            if mean_node.op != "Const":
                continue

            mean_value = Helper.values_from_const(mean_node)

            if has_add_op:
                bias_node_name = graph_info[Helper.node_name_from_input(matched_node[1])].node.input[1]
                bias_node = graph_info[Helper.node_name_from_input(bias_node_name)].node
                if bias_node.op != "Const":
                    continue

                if mean_value.shape != (channel_count,):
                    continue

                mean_value = mean_value - Helper.values_from_const(bias_node)
                cur_graph.remove_node(bias_node.name)
                cur_graph.remove_node(matched_node[1])

            if mean_value.shape != (channel_count,):
                self.logger.warning(
                    "Incorrect shape for mean, found {}, expected {}, "
                    "for node {}.".format(str(mean_value.shape), str((channel_count,)), conv_node.name)
                )
                continue
            var_node_name = Helper.node_name_from_input(bn_node.input[self.INPUT_ORDER[bn_node.op].index("var_op")])
            var_node = graph_info[var_node_name].node
            if var_node.op != "Const":
                continue
            var_value = Helper.values_from_const(var_node)

            if var_value.shape != (channel_count,):
                continue

            beta_node_name = Helper.node_name_from_input(bn_node.input[self.INPUT_ORDER[bn_node.op].index("beta_op")])
            beta_node = graph_info[beta_node_name].node
            if beta_node.op != "Const":
                continue
            beta_value = Helper.values_from_const(beta_node)

            if beta_value.shape != (channel_count,):
                continue

            gamma_node_name = Helper.node_name_from_input(bn_node.input[self.INPUT_ORDER[bn_node.op].index("gamma_op")])
            gamma_node = graph_info[gamma_node_name].node

            if gamma_node.op != "Const":
                continue
            gamma_value = Helper.values_from_const(gamma_node)

            if gamma_value.shape != (channel_count,):
                continue

            variance_epsilon_value = bn_node.attr[self.EPSILON_ATTR[bn_node.op]].f

            if self.scale_after_normalization(bn_node):
                scale_value = (1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)) * gamma_value
            else:
                scale_value = 1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)

            offset_value = (-mean_value * scale_value) + beta_value

            if conv_node.op == "Conv2D":
                original_shape = weights.shape
                tmp_shape = (original_shape[-1], int(weights.size / original_shape[-1]))
                tmp_order = [weights.ndim - 1] + [i for i in range(weights.ndim - 1)]
                scaled_weights = np.copy(weights).transpose(tmp_order).ravel().reshape(tmp_shape)
                reshape_scale = np.array(scale_value).reshape(len(scale_value), 1)
                scaled_weights = np.multiply(scaled_weights, reshape_scale).transpose().reshape(original_shape)
            elif conv_node.op == "DepthwiseConv2dNative":
                scaled_weights = np.copy(weights)
                it = np.nditer(scaled_weights, flags=["multi_index"], op_flags=["readwrite"])
                channel_multiplier = weights.shape[3]
                while not it.finished:
                    current_scale = scale_value[it.multi_index[2] * channel_multiplier + it.multi_index[3]]
                    it[0] *= current_scale
                    it.iternext()

            scaled_weights_node = node_def_pb2.NodeDef()
            scaled_weights_node.op = "Const"
            scaled_weights_node.name = weights_node_name + "_bn_offset"
            scaled_weights_node.attr["dtype"].CopyFrom(weights_node.attr["dtype"])
            scaled_weights_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(scaled_weights, weights.dtype.type, weights.shape)
                )
            )
            cur_graph.replace_const_node(scaled_weights_node, [conv_node.name], weights_node_name)

            offset_node = node_def_pb2.NodeDef()
            offset_node.op = "Const"
            offset_node.name = conv_node.name + "_bn_offset"
            offset_node.attr["dtype"].CopyFrom(mean_node.attr["dtype"])
            offset_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(offset_value, mean_value.dtype.type, offset_value.shape)
                )
            )
            bias_add_node = node_def_pb2.NodeDef()
            bias_add_node.op = "BiasAdd"
            bias_add_node.name = bn_node.name
            bias_add_node.attr["T"].CopyFrom(conv_node.attr["T"])
            bias_add_node.attr["data_format"].CopyFrom(conv_node.attr["data_format"])
            bias_add_node.input.extend([conv_node.name, offset_node.name])
            if bn_node.op == "_FusedBatchNormEx" and bn_node.attr["activation_mode"].s == b"Relu":
                # Create Relu op which takes Bias-Add as input.
                #  Conv2D/Depthwise-Conv2D                         Conv2D/Depthwise-Conv2D
                #      |                                                 |
                #  Bias-Add (originally _FusedBatchNormEx) <---->     Bias-Add
                #      |                                              |       \
                #   <some-node>                                  <some-node>   Relu
                relu_node = node_def_pb2.NodeDef()
                relu_node.op = "Relu"
                relu_node.name = bn_node.name + "_bn_relu"
                relu_node.attr["T"].CopyFrom(conv_node.attr["T"])
                relu_node.input.extend([bias_add_node.name])

            cur_graph.add_node(offset_node, [], [bias_add_node.name])
            cur_graph.add_node(
                bias_add_node, conv_node.name, graph_info[Helper.node_name_from_input(matched_node[-1])].outputs
            )
            if bn_node.op == "_FusedBatchNormEx" and bn_node.attr["activation_mode"].s == b"Relu":
                matchd_node_outputs = graph_info[Helper.node_name_from_input(matched_node[-1])].outputs
                cur_graph.add_node(offset_node, [], [bias_add_node.name])
                cur_graph.add_node(bias_add_node, conv_node.name, [relu_node.name])
                cur_graph.add_node(relu_node, bias_add_node.name, matchd_node_outputs)
            else:
                cur_graph.add_node(offset_node, [], [bias_add_node.name])
                cur_graph.add_node(
                    bias_add_node, conv_node.name, graph_info[Helper.node_name_from_input(matched_node[-1])].outputs
                )
            cur_graph.replace_const_node(scaled_weights_node, [conv_node.name], weights_node_name)

            cur_graph.remove_node(weights_node_name)
            cur_graph.remove_node(mean_node_name)
            cur_graph.remove_node(var_node_name)
            cur_graph.remove_node(beta_node_name)
            cur_graph.remove_node(gamma_node_name)

        return cur_graph.dump_graph()
