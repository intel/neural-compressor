#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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


from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util

from .quantize_graph_base import QuantizeNodeBase
from .quantize_graph_common import QuantizeGraphHelper as helper


class FuseNodeStartWithPad(QuantizeNodeBase):
    def __init__(self, input_graph, output_node_names, perchannel,
                 start_node_name, device, _):
        super(FuseNodeStartWithPad,
              self).__init__(input_graph, output_node_names, perchannel,
                             start_node_name, device)

    def has_relu(self, node_name):
        for _, value in self.node_name_mapping.items():
            if value.node.name == node_name:
                break
            if value.node.op in ("Relu", "Relu6"):
                return True
        return False

    def _apply_pad_conv_fusion(self):
        for _, value in self.node_name_mapping.items():
            if value.node.op in ("Pad") and self.node_name_mapping[
                    value.
                    output[0]].node.op == "Conv2D" and self.node_name_mapping[
                        value.output[
                            0]].node.name == self.start_node_name and self._find_relu_node(
                                value.node):
                paddings_tensor = tensor_util.MakeNdarray(
                    self.node_name_mapping[value.node.input[1]].node.
                    attr["value"].tensor).flatten()
                if any(paddings_tensor):
                    new_node = node_def_pb2.NodeDef()
                    new_node.CopyFrom(value.node)
                    self.add_output_graph_node(new_node)
                else:
                    self.node_name_mapping[
                        value.output[0]].node.input[0] = value.node.input[0]
                    helper.set_attr_int_list(
                        self.node_name_mapping[value.output[0]].node,
                        "padding_list", paddings_tensor)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(value.node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        return 2  # pad + conv

    def apply_the_transform(self):
        self._get_op_list()

        self._apply_pad_conv_fusion()
        self._reset_output_node_maps()

        self.output_graph = self.remove_redundant_quantization(
            self.output_graph)
        # self.remove_dead_nodes(self.output_node_names)
        return self.output_graph
