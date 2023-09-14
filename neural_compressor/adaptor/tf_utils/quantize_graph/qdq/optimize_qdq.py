#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
"""Fuse the DQ + OP + Q fusion pattern, convert fp32 op to int8."""

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile

from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper
from neural_compressor.utils.utility import dump_elapsed_time

from ..quantize_graph_base import QuantizeGraphBase
from .fuse_qdq_bn import FuseNodeStartWithFusedBatchNormV3
from .fuse_qdq_concatv2 import FuseNodeStartWithConcatV2
from .fuse_qdq_conv import FuseNodeStartWithConv2d
from .fuse_qdq_deconv import FuseNodeStartWithDeconv2d
from .fuse_qdq_in import FuseNodeStartWithFusedInstanceNorm
from .fuse_qdq_matmul import FuseNodeStartWithMatmul
from .fuse_qdq_pooling import FuseNodeStartWithPooling


class OptimizeQDQGraph(QuantizeGraphBase):
    """Apply the fusion DQ + OP + Q pattern."""

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
        """Optimize QDQ Graph."""
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

        self.op_wise_seq = op_wise_sequences

        self.device = device
        self.fake_quant = fake_quant
        self.new_api = new_api
        self.performance_only = performance_only
        self.itex_mode = itex_mode
        self.exclude_node_list = []

        self.all_quantizable_node = []
        self.register_transformer("MaxPool", FuseNodeStartWithPooling)
        self.register_transformer("MaxPool3D", FuseNodeStartWithPooling)
        self.register_transformer("Conv2D", FuseNodeStartWithConv2d)
        self.register_transformer("Conv3D", FuseNodeStartWithConv2d)
        self.register_transformer("DepthwiseConv2dNative", FuseNodeStartWithConv2d)
        self.register_transformer("FusedBatchNormV3", FuseNodeStartWithFusedBatchNormV3)
        self.register_transformer("_MklFusedInstanceNorm", FuseNodeStartWithFusedInstanceNorm)
        self.register_transformer("AvgPool", FuseNodeStartWithPooling)
        self.register_transformer("ConcatV2", FuseNodeStartWithConcatV2)
        self.register_transformer("MatMul", FuseNodeStartWithMatmul)
        self.register_transformer("BatchMatMul", FuseNodeStartWithMatmul)
        self.register_transformer("BatchMatMulV2", FuseNodeStartWithMatmul)
        self.register_transformer("Conv2DBackpropInput", FuseNodeStartWithDeconv2d)
        self.register_transformer("Conv3DBackpropInputV2", FuseNodeStartWithDeconv2d)

    def get_quantized_nodes(self):
        """Get the quantized Ops."""
        count = 0
        remove_redundant_quant_flag = False
        op_wise_config_name_list = list(self.op_wise_config.keys())
        all_node_length = len(self.op_wise_config)
        for _, node in enumerate(self.input_graph.node):
            if node in self.input_graph.node and node.op in self.transformers and node.name in self.op_wise_config:
                count += 1
                if count == all_node_length:
                    remove_redundant_quant_flag = True
                _, quantizable_nodes = self.transformers[node.op](
                    input_graph=self.input_graph,
                    patterns=self.op_wise_seq[node.op],
                    remove_redundant_quant_flag=remove_redundant_quant_flag,
                    op_wise_config_name_list=op_wise_config_name_list,
                    op_wise_cfg=self.op_wise_config[node.name],
                    start_node_name=node.name,
                    device=self.device,
                    fake_quant=self.fake_quant,
                    new_api=self.new_api,
                    performance_only=self.performance_only,
                    itex_mode=self.itex_mode,
                ).get_longest_fuse()

                if quantizable_nodes:
                    if quantizable_nodes[-1] == "QuantizeV2":
                        quantizable_nodes.pop()
                    if quantizable_nodes[0] == "Dequantize":
                        quantizable_nodes.pop(0)
                    if node.op in ("ConcatV2", "MaxPool", "MaxPool3D", "AvgPool"):
                        self.all_quantizable_node.extend([[i] for i in quantizable_nodes])
                    else:
                        self.all_quantizable_node.append(quantizable_nodes)
        return self.all_quantizable_node

    @dump_elapsed_time("Pass OptimizeQDQGraph")
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

                self.input_graph, exclude_nodes = self.transformers[node.op](
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
                ).apply_the_transform()

                if exclude_nodes:
                    self.exclude_node_list.extend(exclude_nodes)

        return self.remove_dead_nodes(self.input_graph, self.output_node_names), self.exclude_node_list
