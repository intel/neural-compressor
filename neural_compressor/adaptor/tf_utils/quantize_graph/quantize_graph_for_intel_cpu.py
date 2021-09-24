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

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from neural_compressor.utils.utility import dump_elapsed_time

from .quantize_graph_base import QuantizeGraphBase
from .quantize_graph_common import QuantizeGraphHelper
from .quantize_graph_conv import FuseNodeStartWithConv2d
from .quantize_graph_concatv2 import FuseNodeStartWithConcatV2
from .quantize_graph_matmul import FuseNodeStartWithMatmul
from .quantize_graph_pooling import FuseNodeStartWithPooling

class QuantizeGraphForIntel(QuantizeGraphBase):
    """

    """

    def __init__(self, input_graph, output_node_names, op_wise_config, op_wise_sequences, device, \
                 fake_quant=False):
        """Quantize Graph For Intel Cpu

        Arguments:
            input_graph {[type]} -- [description]
            rules {[type]} -- [description]
            output_node_names {[type]} -- [description]

        Keyword Arguments:
            debug {bool} -- [description] (default: {False})
        """
        super().__init__(output_node_names)
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

        self.op_wise_seq = op_wise_sequences

        self.device = device
        self.fake_quant = fake_quant

        self.all_quantizable_node = []
        self.register_transformer("MaxPool", FuseNodeStartWithPooling)
        self.register_transformer("Conv2D", FuseNodeStartWithConv2d)
        self.register_transformer("DepthwiseConv2dNative", FuseNodeStartWithConv2d)
        self.register_transformer("AvgPool", FuseNodeStartWithPooling)
        self.register_transformer("ConcatV2", FuseNodeStartWithConcatV2)
        self.register_transformer("MatMul", FuseNodeStartWithMatmul)

    @dump_elapsed_time("Pass Quantization")
    def do_transform(self):
        count = 0
        remove_redundant_quant_flag = False
        all_node_length = len(self.op_wise_config)
        for _, node in enumerate(self.input_graph.node):
            if node in self.input_graph.node and node.op in self.transformers \
                and node.name in self.op_wise_config:
                count += 1
                if count == all_node_length:
                    remove_redundant_quant_flag = True

                self.input_graph, quantizable_node_names = self.transformers[node.op](
                    input_graph=self.input_graph,
                    patterns=self.op_wise_seq[node.op],
                    remove_redundant_quant_flag=remove_redundant_quant_flag,
                    op_wise_cfg=self.op_wise_config[node.name],
                    start_node_name=node.name, device=self.device, \
                    fake_quant=self.fake_quant).apply_the_transform()
                if quantizable_node_names:
                    if node.op in ('ConcatV2', 'MaxPool', 'AvgPool'):
                        self.all_quantizable_node.extend([[i] for i in quantizable_node_names])
                    else:
                        self.all_quantizable_node.append(quantizable_node_names)
        return self.remove_dead_nodes(self.input_graph, self.output_node_names), \
            self.all_quantizable_node
