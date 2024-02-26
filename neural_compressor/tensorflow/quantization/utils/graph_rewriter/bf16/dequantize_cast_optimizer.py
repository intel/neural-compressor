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
"""Dequantize Cast Graph Rerewriter."""

import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.tensorflow.quantization.utils.graph_rewriter.graph_base import GraphRewriterBase
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.utils import SPR_BASE_VERSIONS, dump_elapsed_time


class DequantizeCastOptimizer(GraphRewriterBase):
    """Remove the Cast OP and set Dequantize output to B16 if the Cast OP output is BF16."""

    @dump_elapsed_time("Pass DequantizeCastOptimizer")
    def do_transformation(self):
        """Remove the redundant Cast OP after Dequantize.

        Args:
            input_graph_def (graphdef): graphdef object

        Returns:
            [graphdef]: optimized graph
        """
        # stock TF _MklDequantize doesn't support BF16 currently.
        # TODO remove this when spr-base upstream to stock TF.
        if tf.version.VERSION not in SPR_BASE_VERSIONS:
            return self.model

        DT_BFLOAT16 = attr_value_pb2.AttrValue(type=dtypes.bfloat16.as_datatype_enum)
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model
        graph_info = cur_graph.parse_graph()
        target_nodes = cur_graph.query_fusion_pattern_nodes([["Dequantize"], ["Cast"]])
        for i in target_nodes:
            dq_node = graph_info[i[0]].node
            dq_outputs = graph_info[i[0]].outputs

            if len(dq_outputs) > 1:
                continue
            cast_node = graph_info[i[1]].node
            cast_outputs = graph_info[i[1]].outputs
            all_cast_outputs_bf16 = True
            for cast_output in cast_outputs:
                cast_output_node = graph_info[cast_output].node
                if cast_output_node.attr["T"] != DT_BFLOAT16:  # des dtype of the cast must be bfloat16
                    all_cast_outputs_bf16 = False
            if not all_cast_outputs_bf16:
                continue
            dq_node.attr["dtype"].CopyFrom(DT_BFLOAT16)
            for cast_output in cast_outputs:
                successor_node = graph_info[cast_output].node
                replace_index = None
                for index, value in enumerate(successor_node.input):
                    if value == cast_node.name:
                        replace_index = index
                        break
                successor_node.input[replace_index] = dq_node.name

            # remove Cast node
            cur_graph.remove_node(cast_node.name)

        return cur_graph.dump_graph()
