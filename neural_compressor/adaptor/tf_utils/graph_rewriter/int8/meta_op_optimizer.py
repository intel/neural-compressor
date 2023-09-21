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
"""Meta OP Graph Rewriter."""


from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class MetaInfoChangingMemOpOptimizer(GraphRewriterBase):
    """Fuse the pattern like Dequantize + MetaOp + Quantize into MetaOp(set its type to int8).

    With such changes, the Quantize and Dequantize OP will removed for better performance.
    """

    def __init__(self, model):
        """Initialization."""
        super().__init__(model)

        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model

        self.graph_info = self.graph_analyzer.parse_graph()

    @dump_elapsed_time("Pass MetaOpOptimizer")
    def do_transformation(self):
        """Apply the fusion of Dequantize + MetaOp + QuantizeV2."""
        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(
            [["Dequantize"], ("Squeeze", "Reshape"), ("Squeeze", "Reshape"), ["QuantizeV2"]]
        )
        for i in target_nodes:
            if len(i[-1]) == 2:
                continue

            dequantize_node_name = i[0]
            if len(i[-1]) == 3:
                quantize_node_name = i[2]
            else:
                quantize_node_name = i[3]
                dequantize_node_name = i[0]
                if self.graph_info[i[1]].node.op == self.graph_info[i[2]].node.op:
                    continue
            deq_node = self.graph_info[dequantize_node_name].node
            quant_node = self.graph_info[quantize_node_name].node
            if len(self.graph_info[dequantize_node_name].outputs) != 1:
                continue

            if quant_node.attr["mode"].s.decode() == deq_node.attr["mode"].s.decode():
                deq_min_range = self.graph_info[dequantize_node_name].node.input[1]
                deq_max_range = self.graph_info[dequantize_node_name].node.input[2]
                quant_output_min = quantize_node_name + ":1"
                quant_output_max = quantize_node_name + ":2"
                if len(i[-1]) == 3:
                    quantize_input_name = i[1]
                else:
                    quantize_input_name = i[2]

                quantized_node_name = self.graph_info[quantize_node_name].outputs[0]
                # _QuantizedBatchMatMul requires T1 and T2 with qint8 type
                # _QuantizedFusedBatchNorm requires T with qint8 type
                if (
                    self.graph_info[quantized_node_name].node.op == "_QuantizedBatchMatMul"
                    or self.graph_info[quantized_node_name].node.op == "_QuantizedFusedBatchNorm"
                ) and self.graph_info[dequantize_node_name].node.attr["T"].type != dtypes.qint8.as_datatype_enum:
                    continue

                for index, value in enumerate(self.graph_info[quantized_node_name].node.input):
                    if value == quant_output_min:
                        self.graph_info[quantized_node_name].node.input[index] = deq_min_range

                    if value == quant_output_max:
                        self.graph_info[quantized_node_name].node.input[index] = deq_max_range

                    if index == 0:
                        self.graph_info[quantized_node_name].node.input[index] = quantize_input_name

                new_dtype = self.graph_info[dequantize_node_name].node.attr["T"].type
                for node_name in i[1:-1]:
                    self.graph_info[node_name].node.attr["T"].type = new_dtype

                if "T1" in self.graph_info[quantized_node_name].node.attr:
                    self.graph_info[quantized_node_name].node.attr["T1"].type = new_dtype

                if "Tinput" in self.graph_info[quantized_node_name].node.attr:
                    self.graph_info[quantized_node_name].node.attr["Tinput"].type = new_dtype

                if "Thost_inputs" in self.graph_info[quantized_node_name].node.attr:
                    self.graph_info[quantized_node_name].node.attr["Thost_inputs"].list.type[0] = new_dtype

                if "T" in self.graph_info[quantized_node_name].node.attr:
                    self.graph_info[quantized_node_name].node.attr["T"].type = new_dtype

                self.graph_info[i[1]].node.input[0] = self.graph_info[dequantize_node_name].node.input[0]
                self.graph_analyzer.remove_node(dequantize_node_name)
                self.graph_analyzer.remove_node(self.graph_info[quantize_node_name].node.input[1])
                self.graph_analyzer.remove_node(self.graph_info[quantize_node_name].node.input[2])
                self.graph_analyzer.remove_node(quantize_node_name)
        return GraphAnalyzer().dump_graph()
