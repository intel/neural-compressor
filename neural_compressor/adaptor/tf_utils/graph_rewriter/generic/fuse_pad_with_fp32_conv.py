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
"""Fuse Pad into Conv Graph Rewriter."""

import tensorflow as tf
from tensorflow.python.framework import tensor_util

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.adaptor.tf_utils.util import version1_gt_version2

from ..graph_base import GraphRewriterBase


class FusePadWithFP32Conv2DOptimizer(GraphRewriterBase):
    """Fuse Pad op into Conv."""

    def __init__(self, model, excluded_op_names, inputs, cfg, new_api, itex_qdq_mode=False):
        """Initialization."""
        super().__init__(model)
        self.excluded_conv = excluded_op_names
        self.inputs = inputs
        self.cfg = cfg
        self.new_api = new_api
        self.itex_qdq_mode = itex_qdq_mode

    def do_transformation(self):
        """Fuse Pad op into Conv2D/DepthwiseConv2dNative/Conv3D."""
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()

        target_nodes = cur_graph.query_fusion_pattern_nodes(
            [["Pad"], ["Conv2D", "DepthwiseConv2dNative"], ("BiasAdd", "Add", "AddV2")]
        )

        padding_tensor_dict = {}
        for node_combination in target_nodes:
            conv_name = node_combination[1]

            pattern = node_combination[-1]

            if conv_name not in self.cfg:
                continue

            is_perchannel = self.cfg[conv_name][0]

            # Line 55 to line 65 should be removed once the TFDO enabling the single quantized
            # conv2D supporting.
            if len(pattern) == 2:
                # TODO we need to enable single quantizedconv2d with s8 input.
                if not is_perchannel and not cur_graph.has_positive_input(conv_name):
                    continue
                # TFDO has the limitation that the single QuantizedConv2DPerchannel doesn't
                # support padding_list filed.
                if is_perchannel:
                    continue

            if conv_name in self.excluded_conv:
                continue

            padding_tensor = None
            pad_node = None
            if node_combination[0] not in padding_tensor_dict:
                pad_node = graph_info[node_combination[0]].node
                if graph_info[pad_node.input[1]].node.op != "Const":
                    input_node = graph_info[pad_node.input[1]].node
                    if input_node.op == "DataFormatVecPermute":
                        parent_input_node = graph_info[input_node.input[0]].node
                        if parent_input_node.op == "Const":
                            padding_tensor = tensor_util.MakeNdarray(parent_input_node.attr["value"].tensor).flatten()
                        else:
                            continue
                    else:
                        continue
                else:
                    padding_tensor = tensor_util.MakeNdarray(
                        graph_info[pad_node.input[1]].node.attr["value"].tensor
                    ).flatten()
                padding_tensor_dict[node_combination[0]] = padding_tensor
            else:
                padding_tensor = padding_tensor_dict[node_combination[0]]

            if self.itex_qdq_mode:
                enabled_pad_conv2d = bool(
                    tf.version.VERSION == "1.15.0-up3" or version1_gt_version2(tf.version.VERSION, "2.7")
                )
            else:
                enabled_pad_conv2d = bool(tf.version.VERSION == "1.15.0-up3" or self.new_api)

            if any(padding_tensor) and not enabled_pad_conv2d:  # pragma: no cover
                continue

            if pad_node:
                if graph_info[pad_node.input[1]].node.op != "Const":
                    cur_graph.node_name_details[pad_node.name].node.input.remove(pad_node.input[1])
                    cur_graph.remove_node_with_single_input_output(pad_node.name)
                else:
                    cur_graph.remove_node_with_single_input_output(pad_node.name)
                    cur_graph.remove_node(pad_node.input[1])
            conv_node = graph_info[node_combination[1]].node
            # Helper.set_attr_int_list(conv_node, "padding_list", padding_tensor)
            # only when padding attr is explicit, the explicit_paddings is not empty

            if self.itex_qdq_mode:
                if any(padding_tensor) and enabled_pad_conv2d:  # pragma: no cover
                    Helper.set_attr_string(conv_node, "padding", b"EXPLICIT")
                    Helper.set_attr_int_list(conv_node, "explicit_paddings", padding_tensor)
            else:
                if any(padding_tensor) and enabled_pad_conv2d:  # pragma: no cover
                    Helper.set_attr_string(conv_node, "padding", b"EXPLICIT")
                    Helper.set_attr_int_list(conv_node, "explicit_paddings", padding_tensor)

        return cur_graph.dump_graph()
