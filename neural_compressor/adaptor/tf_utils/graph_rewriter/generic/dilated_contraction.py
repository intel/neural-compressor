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
"""Dilated Contraction Graph Rewriter."""

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class DilatedContraction(GraphRewriterBase):
    """Fuse the SpaceToBatchND + Conv + BatchToSpaceND pattern."""

    @dump_elapsed_time("Pass DilatedContraction")
    def do_transformation(self):
        """Dilated Contraction fusion."""
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        target_nodes = cur_graph.query_fusion_pattern_nodes(
            ["SpaceToBatchND", ["Conv2D", "DepthwiseConv2dNative"], "BatchToSpaceND"]
        )

        for node_combination in target_nodes:
            stob_node = graph_info[node_combination[0]].node
            contraction_node = graph_info[node_combination[1]].node
            btos_node = graph_info[node_combination[2]].node
            stob_padding_node = graph_info[stob_node.input[2]].node

            block_shape_node = graph_info[btos_node.input[1]].node
            crops_node = graph_info[btos_node.input[2]].node

            block_value = [i for i in tensor_util.MakeNdarray(block_shape_node.attr["value"].tensor).flat]
            new_dilation = [1, block_value[0], block_value[1], 1]
            # if padding input of SpaceToBatchND can't be directly fetched, we continue
            if stob_padding_node.op != "Const":
                continue
            padding_value = [i for i in tensor_util.MakeNdarray(stob_padding_node.attr["value"].tensor).flat]
            crops_value = [i for i in tensor_util.MakeNdarray(crops_node.attr["value"].tensor).flat]

            contraction_node.input[0] = stob_node.input[0]
            Helper.set_attr_int_list(contraction_node, "dilations", new_dilation)

            real_padding = [padding_value[i] - crops_value[i] for i in range(4)]
            explict_padding = [0, 0, 0, 0, 0, 0, 0, 0]
            data_format = contraction_node.attr["data_format"].s.decode()
            if any(real_padding):
                contraction_node.attr["padding"].s = "EXPLICIT".encode()
                assert data_format in ("NHWC", "NCHW")
                if data_format == "NHWC":
                    explict_padding[2] = real_padding[0]
                    explict_padding[3] = real_padding[1]
                    explict_padding[4] = real_padding[2]
                    explict_padding[5] = real_padding[3]
                else:
                    explict_padding[4] = real_padding[0]
                    explict_padding[5] = real_padding[1]
                    explict_padding[6] = real_padding[2]
                    explict_padding[7] = real_padding[3]
                Helper.set_attr_int_list(contraction_node, "explicit_paddings", explict_padding)

            contraction_node.attr.pop("_output_shapes")
            cur_graph.remove_node(stob_node.name)
            following_node_name = graph_info[node_combination[2]].outputs[0]
            following_node = graph_info[following_node_name].node

            following_node.input[0] = btos_node.input[0]
            cur_graph.remove_node(btos_node.name)

        return cur_graph.dump_graph()
