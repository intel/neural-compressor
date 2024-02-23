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
"""Convert fp32 op to int8 and fuse the pattern."""

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.quantization.utils.quantize_graph_common import QuantizeGraphHelper
from neural_compressor.tensorflow.utils import dump_elapsed_time

from .quantize_graph_base import QuantizeGraphBase
from .quantize_graph_bn import FuseNodeStartWithFusedBatchNormV3
from .quantize_graph_concatv2 import FuseNodeStartWithConcatV2
from .quantize_graph_conv import FuseNodeStartWithConv2d
from .quantize_graph_matmul import FuseNodeStartWithMatmul
from .quantize_graph_pooling import FuseNodeStartWithPooling


class QuantizeGraphForIntel(QuantizeGraphBase):
    """Quantize the graph."""

    def __init__(
        self,
        input_graph,
        input_node_names,
        output_node_names,
        op_wise_config,
        op_wise_sequences,
        device,
        fake_quant=False,
        new_api=False,
        performance_only=False,
        itex_mode=False,
    ):
        """Quantize Graph For Intel Cpu."""
        super().__init__(output_node_names)
        self.op_wise_config = op_wise_config
        # self.perchannel = perchannel
        if isinstance(input_graph, graph_pb2.GraphDef):
            self.input_graph = input_graph
        else:
            self.input_graph = graph_pb2.GraphDef()
            with gfile.Open(input_graph, "rb") as f:
                self.input_graph.ParseFromString(f.read())

        input_output_names = input_node_names + output_node_names
        self.input_graph = QuantizeGraphHelper().remove_training_nodes(
            self.input_graph, protected_nodes=input_output_names
        )

        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.input_graph

        self.graph_info = self.graph_analyzer.parse_graph()
        self.graph_analyzer.get_frame_info()

        self.op_wise_seq = op_wise_sequences

        self.device = device
        self.fake_quant = fake_quant
        self.new_api = new_api
        self.performance_only = performance_only
        self.itex_mode = itex_mode

        self.all_quantizable_node = []
        self.exclude_node_names = []
        self.register_transformer("MaxPool", FuseNodeStartWithPooling)
        self.register_transformer("MaxPool3D", FuseNodeStartWithPooling)
        self.register_transformer("Conv2D", FuseNodeStartWithConv2d)
        self.register_transformer("Conv3D", FuseNodeStartWithConv2d)
        self.register_transformer("FusedBatchNormV3", FuseNodeStartWithFusedBatchNormV3)
        self.register_transformer("DepthwiseConv2dNative", FuseNodeStartWithConv2d)
        self.register_transformer("AvgPool", FuseNodeStartWithPooling)
        self.register_transformer("ConcatV2", FuseNodeStartWithConcatV2)
        self.register_transformer("MatMul", FuseNodeStartWithMatmul)

    @dump_elapsed_time("Pass Quantization")
    def do_transform(self):
        """Apply all the transformers to fuse into int8 op."""
        count = 0
        remove_redundant_quant_flag = False
        op_wise_config_name_list = list(self.op_wise_config.keys())
        all_node_length = len(self.op_wise_config)
        for _, node in enumerate(self.input_graph.node):
            if node in self.input_graph.node and node.op in self.transformers and node.name in self.op_wise_config:
                count += 1
                if count == all_node_length:
                    remove_redundant_quant_flag = True
                self.input_graph, quantizable_node_names, exclude_node_names = self.transformers[node.op](
                    input_graph=self.input_graph,
                    patterns=self.op_wise_seq[node.op],
                    remove_redundant_quant_flag=remove_redundant_quant_flag,
                    op_wise_cfg=self.op_wise_config[node.name],
                    op_wise_config_name_list=op_wise_config_name_list,
                    start_node_name=node.name,
                    device=self.device,
                    fake_quant=self.fake_quant,
                    new_api=self.new_api,
                    performance_only=self.performance_only,
                    itex_mode=self.itex_mode,
                    frame_info=self.graph_analyzer.parent_frame_details,
                ).apply_the_transform()
                if quantizable_node_names:
                    if node.op in ("ConcatV2", "MaxPool", "MaxPool3D", "AvgPool"):
                        self.all_quantizable_node.extend([[i] for i in quantizable_node_names])
                    else:
                        self.all_quantizable_node.append(quantizable_node_names)
                if exclude_node_names:
                    self.exclude_node_names.extend(exclude_node_names)

        return (
            self.remove_dead_nodes(self.input_graph, self.output_node_names),
            self.all_quantizable_node,
            self.exclude_node_names,
        )
