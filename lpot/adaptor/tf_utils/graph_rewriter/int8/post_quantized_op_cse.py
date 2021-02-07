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

import hashlib
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util
from lpot.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper



class PostCseOptimizer(GraphRewriterBase):
    """[summary]
    Remove duplicated nodes like shared quantizev2 and const to decrease the output model size.
    """
    control_op_types = ('Switch', 'Enter', 'Merge', 'NextIteration', 'Exit')

    def _gen_node_hash(self, graph_info, node):

        hash_str = node.op
        hash_str += str(len(node.input))
        for i in node.input:
            input_node = graph_info[Helper.node_name_from_input(i)].node
            if input_node.op == 'Const':
                float_tensor = (tensor_util.MakeNdarray(input_node.attr["value"].tensor))
                hash_str += str(float_tensor.flatten())
            else:
                hash_str += i

        attr_keys = sorted(node.attr)
        for i in attr_keys:
            hash_str += str(node.attr[i])

        return hashlib.md5(hash_str.encode('utf-8')).hexdigest()

    @dump_elapsed_time("Pass PostCseOptimizer")
    def do_transformation(self):
        GraphAnalyzer().graph = self.model

        graph_info = GraphAnalyzer().parse_graph()
        node_hash_info = {}
        loc_attr_node = []
        need_to_keep_const_node_name = []
        for _, v in graph_info.items():
            if '_class' in v.node.attr:
                loc_attr_node.append(v.node.attr['_class'].list.s[0].decode().split(':@')[-1])

        for node_name, i in graph_info.items():
            if node_name in loc_attr_node or i.node.op not in ('QuantizeV2', "Const"):
                continue

            hash_value = self._gen_node_hash(graph_info, i.node)

            if hash_value not in node_hash_info:
                node_hash_info[hash_value] = [node_name]

            if node_name not in node_hash_info[hash_value]:
                node_hash_info[hash_value].append(node_name)

        for _, v in node_hash_info.items():
            if len(v) == 1 or v[0] not in graph_info:
                continue
            node_type = graph_info[v[0]].node.op
            for j in v[1:]:
                if node_type == 'Const' and j in graph_info:
                    output_op_types = [
                        graph_info[out_name].node.op in self.control_op_types for out_name
                        in graph_info[j].outputs]
                    if any(output_op_types):
                        continue

                    next_node = graph_info[j].outputs[0]
                    matched_index = 0
                    for index, origin_input in enumerate(graph_info[next_node].node.input):
                        if origin_input == j:
                            matched_index = index
                            break

                    graph_info[next_node].node.input[matched_index] = v[0]
                    if v[0] not in need_to_keep_const_node_name:
                        need_to_keep_const_node_name.append(v[0])

                    graph_info[v[0]].outputs.append(next_node)
                    graph_info.pop(j)

                elif node_type == 'QuantizeV2':
                    # import pdb; pdb.set_trace()
                    next_node = graph_info[j].outputs[0]
                    quantize_v2_output_names = (j, j + ':1', j + ':2')

                    replace_index = [list(graph_info[next_node].node.input).index(i)
                                     for i in quantize_v2_output_names]

                    graph_info[next_node].node.input[replace_index[0]] = v[0]

                    graph_info[next_node].node.input[replace_index[1]] = v[0] + ':1'
                    graph_info[next_node].node.input[replace_index[2]] = v[0] + ':2'

                    graph_info[v[0]].outputs.append(next_node)

                    if graph_info[j].node.input[1] not in need_to_keep_const_node_name:
                        graph_info.pop(graph_info[j].node.input[1])

                    if graph_info[j].node.input[2] not in need_to_keep_const_node_name:

                        graph_info.pop(graph_info[j].node.input[2])

                    graph_info.pop(j)

                else:
                    self.logger.debug('Unknown Op type {}'.format(node_type))

        output_graph_def = graph_pb2.GraphDef()

        for _, node_info in graph_info.items():
            output_graph_def.node.extend([node_info.node])

        return output_graph_def
