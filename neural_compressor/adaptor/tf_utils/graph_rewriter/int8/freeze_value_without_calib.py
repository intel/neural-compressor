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
"""Freeze Value without calibration Graph Rewriter."""

from tensorflow.core.framework import attr_value_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper

from ..graph_base import GraphRewriterBase


class FreezeValueWithoutCalibTransformer(GraphRewriterBase):
    """Freeze value without calibration."""

    def __init__(self, model, max_min_data, postfix, th=0.95, device="gpu"):
        """Free Max/Min value into QuantizeV2 op.

        Args:
            model (graphdef): input model
            max_min_data (string list): the string context contains max/min values.
            postfix (string): the specified postfix to locate value.
            th (float, optional): The percentage of overall data.Defaults to 0.95.
            device (string, optional): The hardware device type, 'cpu' or 'gpu'.
        """
        super().__init__(model)
        self.data = max_min_data
        if 0.0 < th <= 1.0:
            self.threshold = th
        else:
            self.logger.warning("The threshold value for clipping is invalid, " "Reset it to 0.95 by default.")
            self.threshold = 0.95
        self.postfix = postfix
        self.device = device
        self.cur_graph = GraphAnalyzer()
        self.cur_graph.graph = self.model

        self.graph_info = self.cur_graph.parse_graph()

    def generate_output_graph(self, max_name_value):
        """Generate transformed graph for freeze_max/freeze_min transformation.

        :param max_name_value: target values
        :return: transformed graph
        """
        for node_name, value in max_name_value.items():
            node_name = node_name.replace(":", "__port__").replace("^", "__hat__")
            if node_name not in self.graph_info:
                continue
            new_node = node_def_pb2.NodeDef()
            new_node.op = "Const"
            new_node_postfix = "/frozen_{}_only".format("".join([x for x in self.postfix if x.isalpha()]))
            new_node.name = node_name + new_node_postfix
            new_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            new_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(float(value), dtypes.float32, []))
            )
            output_node_name = self.graph_info[node_name].outputs[0]
            self.cur_graph.replace_const_node(new_node, [Helper.node_name_from_input(output_node_name)], node_name)
            self.cur_graph.remove_node(node_name)

        return GraphAnalyzer().dump_graph()

    def generate_output_graph_ranges(self, max_name_value):
        """Generate transformed graph for freeze_max/freeze_min transformation.

        :param max_name_value: target values
        :return: transformed graph
        """
        for node_name, value in max_name_value.items():
            if node_name not in self.graph_info:
                continue

            min_node = node_def_pb2.NodeDef()
            min_node.op = "Const"
            min_node_postfix = "/frozen_min"
            min_node.name = node_name + min_node_postfix
            min_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            min_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(float(value[0]), dtypes.float32, []))
            )

            max_node = node_def_pb2.NodeDef()
            max_node.op = "Const"
            max_node_postfix = "/frozen_max"
            max_node.name = node_name + max_node_postfix
            max_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            max_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(float(value[1]), dtypes.float32, []))
            )
            output_node_name = self.graph_info[node_name].outputs[0]
            self.cur_graph.replace_const_node(
                min_node, [Helper.node_name_from_input(output_node_name)], node_name + ":0"
            )
            self.cur_graph.replace_const_node(
                max_node, [Helper.node_name_from_input(output_node_name)], node_name + ":1"
            )
            self.cur_graph.remove_node(node_name)

        return GraphAnalyzer().dump_graph()

    def do_transformation_without_calib(self):
        """Apply transformation without calibration."""
        if self.postfix == "__requant_min_max":
            range_data = self.data[self.postfix]
            return self.generate_output_graph_ranges(range_data)
        max_name_value = self.data[self.postfix]
        return self.generate_output_graph(max_name_value)
