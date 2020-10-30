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
from tensorflow.python.framework import dtypes

from .quantize_graph_base import QuantizeNodeBase
from .quantize_graph_common import QuantizeGraphHelper as helper


class FuseNodeStartWithPooling(QuantizeNodeBase):
    def __init__(self, input_graph, output_node_names, perchannel,
                 start_node_name, device, _):
        super(FuseNodeStartWithPooling,
              self).__init__(input_graph, output_node_names, perchannel,
                             start_node_name, device)

    def _add_pool_function(self, original_node, quantized_op_node):
        helper.set_attr_dtype(quantized_op_node, "T", dtypes.quint8)
        helper.copy_attr(quantized_op_node, "ksize",
                         original_node.attr["ksize"])
        helper.copy_attr(quantized_op_node, "strides",
                         original_node.attr["strides"])
        helper.copy_attr(quantized_op_node, "padding",
                         original_node.attr["padding"])

    def _apply_pool_quantization(self):
        for _, v in self.node_name_mapping.items():
            if v.node.name == self.start_node_name and self._find_relu_node(
                    v.node):
                self.eightbitize_single_input_tensor_node(
                    v.node, self._add_pool_function)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(v.node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        return 1

    def apply_the_transform(self):
        self._apply_pool_quantization()
        self._reset_output_node_maps()

        self.output_graph = self.remove_redundant_quantization(
            self.output_graph)
        # self.remove_dead_nodes(self.output_node_names)
        return self.output_graph
