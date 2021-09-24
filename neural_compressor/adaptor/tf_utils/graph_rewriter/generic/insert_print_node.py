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
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util as tu
from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class InsertPrintMinMaxNode(GraphRewriterBase):
    """InsertPrintMinMaxNode Pass for tensorflow sampling.
    """

    def __init__(self, model, pre_node_name, post_node_name, frame_name=None):
        super().__init__(model)
        self.pre_node_name = pre_node_name
        self.post_node_name = post_node_name
        self.signature = pre_node_name + post_node_name
        self.frame_name = frame_name

    def do_transformation(self):
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        insert_node_pairs = []
        top_node = graph_info[self.pre_node_name].node
        if top_node.op == 'ConcatV2':
            for i in range(top_node.attr['N'].i):
                insert_node_pairs.append([top_node.input[i], self.post_node_name])
        else:
            refresh_pre_node_name = graph_info[self.pre_node_name].node.input[0]
            # Check the Conv2D could be fused with previous Pad or not.
            # If so, we need to update the pre-node name correspondingly.
            refresh_pre_node = graph_info[Helper.node_name_from_input(refresh_pre_node_name)].node
            if refresh_pre_node.op == 'Pad' and top_node.op == 'Conv2D':
                pad_const_node_name = refresh_pre_node.input[1]
                pad_const_node = graph_info[pad_const_node_name].node
                padding_tensor = tu.MakeNdarray(pad_const_node.attr["value"].tensor).flatten()
                if not any(padding_tensor) or \
                    (any(padding_tensor) and tf.version.VERSION == '1.15.0-up3'):
                    refresh_pre_node_name = refresh_pre_node.input[0]

            insert_node_pairs.append([refresh_pre_node_name, self.post_node_name])

        output_names = []
        for node_pair_names in insert_node_pairs:
            for index, each_node_name in enumerate(node_pair_names):
                name_with_sig = each_node_name + self.signature
                node_name_prefix = name_with_sig.replace(":", "__port__").replace("^", "__hat__")
                reshape_dims_name = node_name_prefix + "_reshape_dims"
                reduction_dims_name = node_name_prefix + "_reduction_dims"

                reshape_dims_node = Helper.create_constant_node(
                    reshape_dims_name, -1, dtypes.int32, [1])

                reduction_dims_node = Helper.create_constant_node(
                    reduction_dims_name, 0, dtypes.int32, [1])

                if self.frame_name:         # pragma: no cover
                    reduction_dims_enter_node = Helper.create_node(
                        'Enter', reduction_dims_name+'_enter', [reduction_dims_name])
                    reshape_dims_enter_node = Helper.create_node(
                        'Enter', reshape_dims_name+'_enter', [reshape_dims_name])
                    Helper.set_attr_string(reduction_dims_enter_node,
                                           'frame_name', self.frame_name.encode())
                    Helper.set_attr_dtype(reduction_dims_enter_node, 'T', dtypes.int32)
                    # Helper.set_attr_bool(reduction_dims_enter_node, 'is_constant', True)
                    Helper.set_attr_int(reduction_dims_enter_node, 'parallel_iterations', 32)
                    Helper.set_attr_string(reshape_dims_enter_node,
                                           'frame_name', self.frame_name.encode())
                    Helper.set_attr_dtype(reshape_dims_enter_node, 'T', dtypes.int32)
                    # Helper.set_attr_bool(reshape_dims_enter_node, 'is_constant', False)
                    Helper.set_attr_int(reshape_dims_enter_node, 'parallel_iterations', 32)

                reshape_input_name = node_name_prefix + "_reshape_"

                if self.frame_name:
                    reshape_input_node = Helper.create_node("Reshape", reshape_input_name, [
                        each_node_name, reshape_dims_enter_node.name])
                else:
                    reshape_input_node = Helper.create_node("Reshape", reshape_input_name, [
                                                            each_node_name, reshape_dims_name])

                min_input_name = node_name_prefix + "_min"
                if self.frame_name:
                    min_input_node = Helper.create_node(
                        "Min", min_input_name,
                        [reshape_input_name, reduction_dims_enter_node.name])
                else:
                    min_input_node = Helper.create_node(
                        "Min", min_input_name, [reshape_input_name, reduction_dims_name])
                Helper.set_attr_dtype(min_input_node, "Tidx", dtypes.int32)
                Helper.set_attr_bool(min_input_node, "keep_dims", False)

                max_input_name = node_name_prefix + "_max"
                if self.frame_name:
                    max_input_node = Helper.create_node(
                        "Max", max_input_name,
                        [reshape_input_name, reduction_dims_enter_node.name])
                else:
                    max_input_node = Helper.create_node(
                        "Max", max_input_name, [reshape_input_name, reduction_dims_name])
                Helper.set_attr_dtype(max_input_node, "Tidx", dtypes.int32)
                Helper.set_attr_bool(max_input_node, "keep_dims", False)

                max_print_node = Helper.create_node(
                    "Print", node_name_prefix + "_print_max__{}".format(index),
                    [max_input_name + ':0', max_input_name+':0'])
                min_print_node = Helper.create_node(
                    "Print", node_name_prefix + "_print_min__{}".format(index),
                    [min_input_name+':0', min_input_name+':0'])

                if index == 0:
                    max_msg = ';{}_eightbit_max_{}__print__;__max:'.format(
                        self.pre_node_name, each_node_name)
                    min_msg = ';{}_eightbit_min_{}__print__;__min:'.format(
                        self.pre_node_name, each_node_name)
                    src_dt = graph_info[self.pre_node_name].node.attr["T"]
                else:
                    max_msg = ';{}_eightbit_requant_range__print__;__requant_max:'.format(
                        self.pre_node_name)
                    min_msg = ';{}_eightbit_requant_range__print__;__requant_min:'.format(
                        self.pre_node_name)
                    src_dt = graph_info[each_node_name].node.attr["T"]

                reshape_input_node.attr["T"].CopyFrom(src_dt)
                min_input_node.attr["T"].CopyFrom(src_dt)
                min_print_node.attr["T"].CopyFrom(src_dt)
                max_input_node.attr["T"].CopyFrom(src_dt)
                max_print_node.attr["T"].CopyFrom(src_dt)

                min_print_node.attr["message"].s = min_msg.encode()
                min_print_node.attr["first_n"].i = -1
                min_print_node.attr["summarize"].i = 1024

                max_print_node.attr["message"].s = max_msg.encode()
                max_print_node.attr["first_n"].i = -1
                max_print_node.attr["summarize"].i = 1024

                attr_u = [dtypes.as_dtype(src_dt.type).as_datatype_enum]
                min_print_node.attr["U"].list.CopyFrom(
                    attr_value_pb2.AttrValue.ListValue(type=attr_u))
                max_print_node.attr["U"].list.CopyFrom(
                    attr_value_pb2.AttrValue.ListValue(type=attr_u))
                if self.frame_name:
                    cur_graph.add_node(reshape_dims_node, None, [reshape_dims_enter_node.name])
                    cur_graph.add_node(reduction_dims_node, None, [reduction_dims_enter_node.name])
                    cur_graph.add_node(reshape_dims_enter_node, None, [reshape_input_name])
                    cur_graph.add_node(reduction_dims_enter_node, None,
                                       [max_input_name, min_input_name])
                else:
                    cur_graph.add_node(reshape_dims_node, None, [reshape_input_name])
                    cur_graph.add_node(reduction_dims_node, None, [max_input_name, min_input_name])
                cur_graph.add_node(max_input_node, None, [max_print_node.name])
                cur_graph.add_node(min_input_node, None, [min_print_node.name])
                cur_graph.add_node(min_print_node, min_input_name, [])
                cur_graph.add_node(max_print_node, max_input_name, [])

                cur_graph.add_node(reshape_input_node, each_node_name,
                                   [max_input_name, min_input_name])
                output_names.append(max_print_node.name)
                output_names.append(min_print_node.name)

        return cur_graph.dump_graph(), output_names
