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
"""Fuse QuantizedMatMul with redundant Dequantize Graph Rewriter."""

from tensorflow.core.framework import attr_value_pb2, node_def_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper

from ..graph_base import GraphRewriterBase


class FuseMatMulRedundantDequantizeTransformer(GraphRewriterBase):
    """Fuse _QuantizedMatMul with the successor Dequantize Op."""

    fuse_patterns = [["_QuantizedMatMul", "_QuantizedBatchMatMul"], ["Dequantize", "Cast"]]

    def __init__(self, model, device="cpu"):
        """Initialization."""
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model
        self.graph_info = self.graph_analyzer.parse_graph()

    def do_transformation(self):
        """Fuse the _QuantizedMatMul with the following Dequantize op.

        The output of _QuantizedMatMul or is fp32 or bf16.

        Returns:
            [graphdef]: the optimized graphdef object
        """
        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(self.fuse_patterns)

        for i in target_nodes:
            quantized_node_name = i[0]
            quantized_node = self.graph_info[quantized_node_name].node
            dequantize_node_name = i[1]
            dequantize_node = self.graph_info[dequantize_node_name].node

            if len(self.graph_info[quantized_node_name].outputs) > 3:  # pragma: no cover
                need_drop = False
                for output in self.graph_info[quantized_node_name].outputs:
                    if self.graph_info[output].node.op != "Dequantize":
                        need_drop = True
                        break
                if need_drop:
                    continue

            # ignore shared output case for license-plate-recognition-barrier-0007 model
            if (
                len(self.graph_info[dequantize_node_name].outputs) == 2
                and self.graph_info[self.graph_info[dequantize_node_name].outputs[0]].node.op == "Reshape"
                and self.graph_info[self.graph_info[dequantize_node_name].outputs[1]].node.op == "Shape"
            ):
                continue

            new_node = node_def_pb2.NodeDef()
            new_node.op = quantized_node.op

            if dequantize_node.op == "Dequantize":
                fused_ops = str(quantized_node.attr["fused_ops"].list.s).replace("Requantize", "Dequantize")
            new_node.name = quantized_node.name + "_dequantize"

            for _, value in enumerate(quantized_node.input):
                new_node.input.append(value)

            if "input_quant_mode" in quantized_node.attr:
                new_node.attr["input_quant_mode"].CopyFrom(quantized_node.attr["input_quant_mode"])
            if "output_quant_mode" in quantized_node.attr:
                new_node.attr["output_quant_mode"].CopyFrom(quantized_node.attr["output_quant_mode"])
            if "leakyrelu_alpha" in quantized_node.attr:
                new_node.attr["leakyrelu_alpha"].CopyFrom(quantized_node.attr["leakyrelu_alpha"])
            if "T1" in quantized_node.attr:
                new_node.attr["T1"].CopyFrom(quantized_node.attr["T1"])
            if "T2" in quantized_node.attr:
                new_node.attr["T2"].CopyFrom(quantized_node.attr["T2"])
            if "U" in quantized_node.attr:
                new_node.attr["U"].CopyFrom(quantized_node.attr["U"])
            if "is_weight_const" in quantized_node.attr:
                new_node.attr["is_weight_const"].CopyFrom(quantized_node.attr["is_weight_const"])
            if "is_bias_const" in quantized_node.attr:
                new_node.attr["is_bias_const"].CopyFrom(quantized_node.attr["is_bias_const"])
            if "transpose_a" in quantized_node.attr:
                new_node.attr["transpose_a"].CopyFrom(quantized_node.attr["transpose_a"])
            if "transpose_b" in quantized_node.attr:
                new_node.attr["transpose_b"].CopyFrom(quantized_node.attr["transpose_b"])
            if "Tdevice_inputs" in quantized_node.attr:
                new_node.attr["Tdevice_inputs"].CopyFrom(quantized_node.attr["Tdevice_inputs"])
            if "Tdevice_outputs" in quantized_node.attr:
                new_node.attr["Tdevice_outputs"].CopyFrom(quantized_node.attr["Tdevice_outputs"])
            if "Thost_inputs" in quantized_node.attr:
                new_node.attr["Thost_inputs"].CopyFrom(quantized_node.attr["Thost_inputs"])
            if "Tbias" in quantized_node.attr:
                new_node.attr["Tbias"].CopyFrom(quantized_node.attr["Tbias"])
            if "adj_x" in quantized_node.attr:
                new_node.attr["adj_x"].CopyFrom(quantized_node.attr["adj_x"])
            if "adj_y" in quantized_node.attr:
                new_node.attr["adj_y"].CopyFrom(quantized_node.attr["adj_y"])
            if "input_quant_mode" in quantized_node.attr:
                new_node.attr["input_quant_mode"].CopyFrom(quantized_node.attr["input_quant_mode"])
            if "output_quant_mode" in quantized_node.attr:
                new_node.attr["output_quant_mode"].CopyFrom(quantized_node.attr["output_quant_mode"])
            if "fused_ops" in quantized_node.attr:
                new_node.attr["fused_ops"].CopyFrom(quantized_node.attr["fused_ops"])

            # update Tbias for single MatMul without bias case, same as Tout.
            if dequantize_node.op == "Dequantize":
                Helper.set_attr_type_list(new_node, "Thost_outputs", [dequantize_node.attr["dtype"].type])
                new_node.attr["Tout"].CopyFrom(attr_value_pb2.AttrValue(type=dequantize_node.attr["dtype"].type))
                if new_node.op == "_QuantizedBatchMatMul":
                    new_node.attr["U"].CopyFrom(attr_value_pb2.AttrValue(type=dequantize_node.attr["DstT"].type))
                if str(quantized_node.attr["fused_ops"].list.s) == str([b"Requantize"]):
                    new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=dequantize_node.attr["dtype"].type))
                Helper.set_attr_string_list(new_node, "fused_ops", eval(fused_ops))
            else:
                Helper.set_attr_type_list(new_node, "Thost_outputs", [dequantize_node.attr["DstT"].type])
                new_node.attr["Tout"].CopyFrom(attr_value_pb2.AttrValue(type=dequantize_node.attr["DstT"].type))
                if new_node.op == "_QuantizedBatchMatMul":
                    new_node.attr["U"].CopyFrom(attr_value_pb2.AttrValue(type=dequantize_node.attr["DstT"].type))

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
