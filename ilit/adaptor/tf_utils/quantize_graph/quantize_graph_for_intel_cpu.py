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

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile

from .quantize_graph_base import QuantizeGraphBase
from .quantize_graph_common import QuantizeGraphHelper
from .quantize_graph_conv import FuseNodeStartWithConv2d
from .quantize_graph_concatv2 import FuseNodeStartWithConcatV2
from .quantize_graph_matmul import FuseNodeStartWithMatmul
from .quantize_graph_pooling import FuseNodeStartWithPooling
from .quantize_graph_pad import FuseNodeStartWithPad


class QuantizeGraphForIntel(QuantizeGraphBase):
    """

    """

    def __init__(self, input_graph, output_node_names, op_wise_config, device):
        """Quantize Graph For Intel Cpu

        Arguments:
            input_graph {[type]} -- [description]
            rules {[type]} -- [description]
            output_node_names {[type]} -- [description]

        Keyword Arguments:
            debug {bool} -- [description] (default: {False})
        """
        super(QuantizeGraphForIntel, self).__init__(output_node_names)
        self.op_wise_config = op_wise_config
        # self.perchannel = perchannel
        if isinstance(input_graph, graph_pb2.GraphDef):
            self.input_graph = input_graph
        else:
            self.input_graph = graph_pb2.GraphDef()
            with gfile.Open(input_graph, 'rb') as f:
                self.input_graph.ParseFromString(f.read())

        self.input_graph = QuantizeGraphHelper().remove_training_nodes(
            self.input_graph, protected_nodes=output_node_names)

        self.device = device
        # self.excluded_ops = excluded_ops
        # self.excluded_nodes = excluded_nodes
        self.register_transformer("MaxPool", FuseNodeStartWithPooling)
        self.register_transformer("Conv2D", FuseNodeStartWithConv2d)
        self.register_transformer("DepthwiseConv2dNative",
                                  FuseNodeStartWithConv2d)
        self.register_transformer("AvgPool", FuseNodeStartWithPooling)
        self.register_transformer("ConcatV2", FuseNodeStartWithConcatV2)
        self.register_transformer("Pad", FuseNodeStartWithPad)
        self.register_transformer("MatMul", FuseNodeStartWithMatmul)

    def can_fused_ops(self, start_node):
        registered_transformer = self.transformers[start_node.op][0]
        worker = registered_transformer(
            self.input_graph, self.output_node_names,
            False, start_node.name, self.device, False)
        can_fused_nodes = worker.get_longest_fuse()
        if isinstance(can_fused_nodes, tuple):
            return can_fused_nodes[1]
        else:
            return []

    def do_transform(self):
        for _, node in enumerate(self.input_graph.node):
            if node in self.input_graph.node and node.op in self.transformers.keys(
            ) and node.name in self.op_wise_config:
                if len(self.transformers[node.op]) > 1:
                    last_fuse_ops_count = 0
                    last_longest_fuse_worker = None
                    for registered_transformer in self.transformers[node.op]:
                        worker = registered_transformer(
                            self.input_graph, self.output_node_names,
                            self.op_wise_config[node.name][0], node.name, self.device,
                            self.op_wise_config[node.name][2])
                        cur_fuse_op_count = worker.get_longest_fuse()

                        if cur_fuse_op_count > last_fuse_ops_count:
                            last_fuse_ops_count = cur_fuse_op_count
                            last_longest_fuse_worker = worker
                    self.input_graph = last_longest_fuse_worker.apply_the_transform(
                    )
                else:
                    for registered_transformer in self.transformers[node.op]:
                        worker = registered_transformer(
                            self.input_graph, self.output_node_names,
                            self.op_wise_config[node.name][0], node.name, self.device,
                            self.op_wise_config[node.name][2])
                        self.input_graph = worker.apply_the_transform()

        return self.remove_dead_nodes(self.input_graph, self.output_node_names)
