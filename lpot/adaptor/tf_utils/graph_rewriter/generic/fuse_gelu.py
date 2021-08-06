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

import tensorflow as tf
from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper
from tensorflow.python.framework import dtypes


class FuseGeluOptimizer(GraphRewriterBase): # pragma: no cover
    """Fuse Sqrt + RealDiv + Erf + AddV2 + Mul + Mul into Gelu op.
    """

    def do_transformation(self):
        if tf.version.VERSION not in ('1.15.0-up2','1.15.0-up3'):
            return self.model

        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        #Below code is relative to expression on
        # https://github.com/IntelAI/models/blob/master/models/language_modeling/tensorflow/
        # bert_large/inference/generic_ops.py#L105
        target_nodes = cur_graph.query_fusion_pattern_nodes(
            [["Pow"], ["Mul"], ["AddV2"], ["Mul"], ["Tanh"], ["AddV2"], ["Mul"], ['Mul']])

        if not target_nodes:
            target_nodes = cur_graph.query_fusion_pattern_nodes(
            [["Pow"], ["Mul"], ["AddV2"], ["Mul"], ["Tanh"], ["AddV2"], ["Mul"]])

        for node_combination in target_nodes:
            match_node_length = len(node_combination)
            pow_node = graph_info[node_combination[0]].node
            mul_1_node = graph_info[node_combination[1]].node
            addv2_1_node = graph_info[node_combination[2]].node
            mul_2_node = graph_info[node_combination[3]].node
            tanh_node = graph_info[node_combination[4]].node
            addv2_2_node = graph_info[node_combination[5]].node
            if match_node_length == 8:
                mul_3_node = graph_info[node_combination[6]].node
            else:
                mul_3_node = graph_info[node_combination[7]].node

            gelu_input_name = None
            pow_const_node_name = None
            pow_value = None

            for i in pow_node.input:
                node_name = Helper.node_name_from_input(i)
                if graph_info[node_name].node.op != 'Const':
                    gelu_input_name = i

                if graph_info[node_name].node.op == 'Const':
                    pow_const_node_name = i
                    pow_value = graph_info[node_name].node.attr['value'].tensor.float_val[0]
                    break

            if pow_value != 3:
                continue
            mul_1_value = None
            mul_1_const_node_name = None
            for i in mul_1_node.input:
                i = Helper.node_name_from_input(i)
                if i != pow_node.name and graph_info[i].node.op == 'Const':
                    mul_1_const_node_name = i
                    mul_1_value = graph_info[i].node.attr['value'].tensor.float_val[0]
                    break
            if mul_1_value != 0.044714998453855515:
                continue

            mul_2_value = None
            mul_2_const_node_name = None
            for i in mul_2_node.input:
                i = Helper.node_name_from_input(i)
                if i != addv2_1_node.name and graph_info[i].node.op == 'Const':
                    mul_2_const_node_name = i
                    mul_2_value = graph_info[i].node.attr['value'].tensor.float_val[0]
                    break
            if mul_2_value != 0.7978845834732056:
                continue

            addv2_2_value = None
            addv2_2_const_node_name = None
            for i in addv2_2_node.input:
                i = Helper.node_name_from_input(i)
                if i != tanh_node.name and graph_info[i].node.op == 'Const':
                    addv2_2_const_node_name = i
                    addv2_2_value = graph_info[i].node.attr['value'].tensor.float_val[0]
                    break
            if addv2_2_value != 1:
                continue

            rest_mul_node = None
            if match_node_length == 8:
                for i in mul_3_node.input:
                    i = Helper.node_name_from_input(i)
                    if i != addv2_2_node.name:
                        rest_mul_node = graph_info[i].node
                        break

                if not rest_mul_node or rest_mul_node.op != 'Mul':
                    continue
            else:
                rest_mul_node = graph_info[node_combination[6]].node

            rest_mul_valid = False
            rest_mul_const_node_name = None
            for i in rest_mul_node.input:
                i = Helper.node_name_from_input(i)
                if graph_info[i].node.op == 'Const' and \
                    graph_info[i].node.attr['value'].tensor.float_val[0] == 0.5:
                    rest_mul_const_node_name = i
                    rest_mul_valid = True
                    break

            if not rest_mul_valid:
                continue

            cur_graph.remove_node(pow_const_node_name)
            cur_graph.remove_node(pow_node.name)
            cur_graph.remove_node(mul_1_node.name)
            cur_graph.remove_node(mul_1_const_node_name)
            cur_graph.remove_node(addv2_1_node.name)
            cur_graph.remove_node(mul_2_node.name)
            cur_graph.remove_node(mul_2_const_node_name)
            cur_graph.remove_node(tanh_node.name)
            cur_graph.remove_node(addv2_2_node.name)
            cur_graph.remove_node(addv2_2_const_node_name)
            cur_graph.remove_node(rest_mul_node.name)
            cur_graph.remove_node(rest_mul_const_node_name)

            original_last = graph_info[mul_3_node.name].outputs
            cur_graph.remove_node(mul_3_node.name)
            gelu_node = Helper.create_node("Gelu", mul_3_node.name, [gelu_input_name])
            Helper.set_attr_bool(gelu_node, "approximate", True)
            Helper.set_attr_dtype(gelu_node, "T", dtypes.float32)

            cur_graph.add_node(gelu_node, gelu_input_name, original_last)

        target_nodes = cur_graph.query_fusion_pattern_nodes(
            [["Sqrt"], ["RealDiv"], ["Erf"], ["AddV2"], ["Mul"], ["Mul"]])

        for node_combination in target_nodes:

            sqrt_node = graph_info[node_combination[0]].node
            realdiv_node = graph_info[node_combination[1]].node
            erf_node = graph_info[node_combination[2]].node
            addv2_node = graph_info[node_combination[3]].node
            mul1_node = graph_info[node_combination[4]].node
            mul2_node = graph_info[node_combination[5]].node

            sqrt_input_name = Helper.node_name_from_input(sqrt_node.input[0])
            sqrt_value = graph_info[sqrt_input_name].node.attr['value'].tensor.float_val[0]

            if sqrt_value != 2:
                continue

            addv2_value = None
            mul1_value = None
            gelu_input_name = None

            for i in realdiv_node.input:
                i = Helper.node_name_from_input(i)
                if i != sqrt_node.name:
                    gelu_input_name = i
                    break

            addv2_const_name = None
            for i in addv2_node.input:
                i = Helper.node_name_from_input(i)
                if i != erf_node.name:
                    addv2_value = graph_info[i].node.attr['value'].tensor.float_val[0]
                    addv2_const_name = i
                    break

            if addv2_value != 1:
                continue

            mul1_const_name = None
            for i in mul1_node.input:
                i = Helper.node_name_from_input(i)
                if i != addv2_node.name:
                    mul1_value = graph_info[i].node.attr['value'].tensor.float_val[0]
                    mul1_const_name = i
                    break

            if mul1_value != 0.5:
                continue

            cur_graph.remove_node(sqrt_node.input[0])
            cur_graph.remove_node(sqrt_node.name)
            cur_graph.remove_node(realdiv_node.name)
            cur_graph.remove_node(erf_node.name)
            cur_graph.remove_node(addv2_node.name)
            cur_graph.remove_node(mul1_node.name)
            cur_graph.remove_node(addv2_const_name)
            cur_graph.remove_node(sqrt_input_name)
            cur_graph.remove_node(mul1_const_name)

            original_last = graph_info[mul2_node.name].outputs
            cur_graph.remove_node(mul2_node.name)
            gelu_node = Helper.create_node("Gelu", mul2_node.name, [gelu_input_name])
            Helper.set_attr_bool(gelu_node, "approximate", False)
            Helper.set_attr_dtype(gelu_node, "T", dtypes.float32)

            cur_graph.add_node(gelu_node, gelu_input_name, original_last)

        return cur_graph.dump_graph()
