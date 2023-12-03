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
"""Fuse QuantizedConv QuantizedDeConv with redundant Dequantize Graph Rewriter."""

from tensorflow.core.framework import attr_value_pb2, node_def_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper

from ..graph_base import GraphRewriterBase


class FuseConvRedundantDequantizeTransformer(GraphRewriterBase):
    """Fuse _QuantizedConv/_QuantizedDeConv with the successor Dequantize Op."""

    fuse_patterns = [
        [
            "_FusedQuantizedConv3D",
            "_FusedQuantizedConv2D",
            "_FusedQuantizedDepthwiseConv2D",
            "_FusedQuantizedDeconv2D",
            "_FusedQuantizedDeconv3D",
        ],
        ["Dequantize"],
    ]

    fuse_sum_op_types_str = (
        str([b"BiasAdd", b"Sum", b"Requantize"]),
        str([b"BiasAdd", b"Sum", b"Relu", b"Requantize"]),
        str([b"BiasAdd", b"Sum", b"LeakyRelu", b"Requantize"]),
        str([b"BiasAdd", b"Relu", b"Sum", b"Requantize"]),
        str([b"BiasAdd", b"LeakyRelu", b"Sum", b"Requantize"]),
    )

    def __init__(self, model, device="cpu"):
        """Initialization."""
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model
        self.graph_info = self.graph_analyzer.parse_graph()

    def do_transformation(self):
        """Fuse the _QuantizedConv Op with the following Dequantize op.

        The output of _QuantizedConv or is fp32 or bf16.

        Returns:
            [graphdef]: the optimized graphdef object
        """
        dtype_map_dict = {
            dtypes.qint8.as_datatype_enum: dtypes.qint8,
            dtypes.quint8.as_datatype_enum: dtypes.quint8,
            dtypes.float32.as_datatype_enum: dtypes.float32,
            dtypes.qint32.as_datatype_enum: dtypes.qint32,
            dtypes.bfloat16.as_datatype_enum: dtypes.bfloat16,
        }
        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(self.fuse_patterns)

        for i in target_nodes:
            quantized_node_name = i[0]
            quantized_node = self.graph_info[quantized_node_name].node
            dequantize_node_name = i[1]
            dequantize_node = self.graph_info[dequantize_node_name].node

            if len(self.graph_info[quantized_node_name].outputs) > 3:
                continue

            # QuantizedConv doesn't support {"BiasAdd", "Sum", "Activation", "Dequantize"},
            # {"BiasAdd", "Activation", "Sum", "Dequantize"} and {"BiasAdd", "Sum", "Dequantize"}
            if str(quantized_node.attr["fused_ops"].list.s) in self.fuse_sum_op_types_str:
                continue

            new_node = node_def_pb2.NodeDef()
            new_node.op = quantized_node.op
            fused_ops = str(quantized_node.attr["fused_ops"].list.s).replace("Requantize", "Dequantize")
            new_node.name = quantized_node.name + "_dequantize"
            for _, value in enumerate(quantized_node.input):
                new_node.input.append(value)

            if "Tinput" in quantized_node.attr:
                new_node.attr["Tinput"].CopyFrom(quantized_node.attr["Tinput"])
            if "Tfilter" in quantized_node.attr:
                new_node.attr["Tfilter"].CopyFrom(quantized_node.attr["Tfilter"])
            if "strides" in quantized_node.attr:
                new_node.attr["strides"].CopyFrom(quantized_node.attr["strides"])
            if "padding" in quantized_node.attr:
                new_node.attr["padding"].CopyFrom(quantized_node.attr["padding"])
            if "alpha" in quantized_node.attr:
                new_node.attr["alpha"].CopyFrom(quantized_node.attr["alpha"])
            if "Tbias" in quantized_node.attr:
                new_node.attr["Tbias"].CopyFrom(quantized_node.attr["Tbias"])
            if "data_format" in quantized_node.attr:
                new_node.attr["data_format"].CopyFrom(quantized_node.attr["data_format"])
            if "is_filter_const" in quantized_node.attr:
                new_node.attr["is_filter_const"].CopyFrom(quantized_node.attr["is_filter_const"])
            if "is_bias_const" in quantized_node.attr:
                new_node.attr["is_bias_const"].CopyFrom(quantized_node.attr["is_bias_const"])
            if "dilations" in quantized_node.attr:
                new_node.attr["dilations"].CopyFrom(quantized_node.attr["dilations"])
            if "explicit_paddings" in quantized_node.attr:
                new_node.attr["explicit_paddings"].CopyFrom(quantized_node.attr["explicit_paddings"])
            if "Tdevice_inputs" in quantized_node.attr:
                new_node.attr["Tdevice_inputs"].CopyFrom(quantized_node.attr["Tdevice_inputs"])
            if "Tdevice_outputs" in quantized_node.attr:
                new_node.attr["Tdevice_outputs"].CopyFrom(quantized_node.attr["Tdevice_outputs"])
            if "Thost_inputs" in quantized_node.attr:
                new_node.attr["Thost_inputs"].CopyFrom(quantized_node.attr["Thost_inputs"])
            Helper.set_attr_type_list(new_node, "Thost_outputs", [dequantize_node.attr["dtype"].type])
            new_node.attr["out_type"].CopyFrom(attr_value_pb2.AttrValue(type=dequantize_node.attr["dtype"].type))
            Helper.set_attr_string_list(new_node, "fused_ops", eval(fused_ops))
            if "Tsummand" in quantized_node.attr:
                Helper.set_attr_dtype(new_node, "Tsummand", dtype_map_dict[dequantize_node.attr["dtype"].type])

            top_node_name = Helper.node_name_from_input(quantized_node.input[0])
            if self.graph_info[dequantize_node_name].outputs:
                self.graph_analyzer.replace_single_node(
                    new_node,
                    [top_node_name],
                    quantized_node_name,
                    self.graph_info[dequantize_node_name].outputs,
                    dequantize_node_name,
                )
                self.graph_analyzer.remove_node(dequantize_node_name)
            else:
                self.graph_analyzer.remove_node(dequantize_node_name)

                new_node.name = dequantize_node_name
                self.graph_analyzer.replace_single_node(
                    new_node, [top_node_name], quantized_node_name, [], dequantize_node_name
                )

            self.graph_analyzer.remove_node(quantized_node_name)

        return self.graph_analyzer.dump_graph()
