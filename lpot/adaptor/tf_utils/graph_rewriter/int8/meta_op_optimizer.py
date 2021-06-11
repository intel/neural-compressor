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


from lpot.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class MetaInfoChangingMemOpOptimizer(GraphRewriterBase):
    """Fuse the pattern like Dequantize + MetaOp + Quantize into MetaOp(set its type to int8).
       With such changes, the Quantize and Dequantize OP will removed for better performance.
    """
    meta_op_type_list = ['Reshape', 'Squeeze']

    def __init__(self, model):
        super().__init__(model)

        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model

        self.graph_info = self.graph_analyzer.parse_graph()

    def _check_and_apply_transform(self, quantize_node_name):
        cur_node = self.graph_info[quantize_node_name].node
        res = [quantize_node_name]
        found_meta_op_flag = False
        while cur_node.input:
            pre_node_name = Helper.node_name_from_input(cur_node.input[0])
            pre_node = self.graph_info[pre_node_name].node
            pre_node_op = pre_node.op
            if pre_node_op in self.meta_op_type_list:
                res.insert(0, pre_node_name)
                cur_node = pre_node
            elif pre_node_op == 'Dequantize' and len(self.graph_info[pre_node_name].outputs) == 1:
                res.insert(0, pre_node_name)
                found_meta_op_flag = True
                break
            else:
                break

        dequantize_node_name = res[0]
        quantize_node_name = res[-1]
        deq_node = self.graph_info[dequantize_node_name].node
        quant_node = self.graph_info[quantize_node_name].node

        if found_meta_op_flag and len(res) > 2 and \
            quant_node.attr['mode'].s.decode() == deq_node.attr['mode'].s.decode():
            deq_min_range = self.graph_info[dequantize_node_name].node.input[1]
            deq_max_range = self.graph_info[dequantize_node_name].node.input[2]
            quant_output_min = quantize_node_name + ':1'
            quant_output_max = quantize_node_name + ':2'
            quantize_input_name = res[-2]

            quantized_node_name = self.graph_info[quantize_node_name].outputs[0]

            for index, value in enumerate(self.graph_info[quantized_node_name].node.input):
                if value == quant_output_min:
                    self.graph_info[quantized_node_name].node.input[index] = deq_min_range

                if value == quant_output_max:
                    self.graph_info[quantized_node_name].node.input[index] = deq_max_range

                if index == 0:
                    self.graph_info[quantized_node_name].node.input[index] = quantize_input_name

            new_dtype = self.graph_info[dequantize_node_name].node.attr['T'].type
            for node_name in res[1: -1]:
                self.graph_info[node_name].node.attr['T'].type = new_dtype

            if 'T1' in self.graph_info[quantized_node_name].node.attr:
                self.graph_info[quantized_node_name].node.attr['T1'].type = new_dtype

            if 'Tinput' in self.graph_info[quantized_node_name].node.attr:
                self.graph_info[quantized_node_name].node.attr['Tinput'].type = new_dtype

            if 'T' in self.graph_info[quantized_node_name].node.attr:
                self.graph_info[quantized_node_name].node.attr['T'].type = new_dtype

            self.graph_info[res[1]
                            ].node.input[0] = self.graph_info[dequantize_node_name].node.input[0]
            self.graph_analyzer.remove_node(dequantize_node_name)
            self.graph_analyzer.remove_node(self.graph_info[quantize_node_name].node.input[1])
            self.graph_analyzer.remove_node(self.graph_info[quantize_node_name].node.input[2])
            self.graph_analyzer.remove_node(quantize_node_name)

    @dump_elapsed_time("Pass MetaOpOptimizer")
    def do_transformation(self):

        deq_node = self.graph_analyzer.query_fusion_pattern_nodes([['QuantizeV2']])

        for i in deq_node:
            self._check_and_apply_transform(i[0])
        return GraphAnalyzer().dump_graph()
