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
"""Fuse QuantizedMatMul with Requantize/Dequantize Graph Rewriter."""

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.adaptor.tf_utils.util import version1_gt_version2, version1_lt_version2

from ..graph_base import GraphRewriterBase


class FuseMatMulRequantizeDequantizeTransformer(GraphRewriterBase):
    """Fuse QuantizedMatMul + Requantize + Dequantize into QuantizedMatMulWithBiasAndDequantize."""

    def __init__(self, model, device="cpu"):
        """Initialization."""
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model

        self.graph_info = self.graph_analyzer.parse_graph()

        self.eps = 1e-5

    def do_transformation(self):
        """Apply the fusion of QuantizedMatMul + Requantize + Dequantize."""
        fuse_pattern = []
        if tf.version.VERSION in ("1.15.0-up2", "1.15.0-up3") or version1_gt_version2(tf.version.VERSION, "2.1.0"):
            fuse_pattern = [["QuantizedMatMulWithBias"], ["Requantize"], ["Dequantize"], ("Softmax",)]
        float32_type = dtypes.float32.as_datatype_enum
        qint32_type = dtypes.qint32.as_datatype_enum
        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(fuse_pattern)
        for i in target_nodes:
            # TODO Remove below checker once the TF's limitation removed.
            if len(i) == 5 and version1_lt_version2(tf.__version__, "2.6.0"):
                continue

            quantized_node_name = i[0]
            quantized_node = self.graph_info[quantized_node_name].node
            requantize_node_name = i[1]
            requantize_node = self.graph_info[requantize_node_name].node
            requested_output_min_name = requantize_node.input[3]
            requested_output_max_name = requantize_node.input[4]
            deq_node_name = i[2]

            quantized_node_op = i[-1][0]

            new_node = node_def_pb2.NodeDef()

            new_node.op = quantized_node_op + "AndDequantize"
            new_node.name = requantize_node_name
            for _, value in enumerate(quantized_node.input):
                new_node.input.append(value)

            new_node.input.append(requested_output_min_name)
            new_node.input.append(requested_output_max_name)
            if "T1" in quantized_node.attr:
                new_node.attr["T1"].CopyFrom(quantized_node.attr["T1"])
            if "T2" in quantized_node.attr:
                new_node.attr["T2"].CopyFrom(quantized_node.attr["T2"])

            top_node_name = Helper.node_name_from_input(quantized_node.input[0])
            max_filter_node = self.graph_info[new_node.input[6]].node
            min_filter_node = self.graph_info[new_node.input[5]].node
            last_node = self.graph_info[new_node.input[0]].node

            weight_node = self.graph_info[Helper.node_name_from_input(new_node.input[1])].node
            bias_node = self.graph_info[Helper.node_name_from_input(new_node.input[2])].node
            max_input_node = self.graph_info[last_node.input[-1]].node
            min_input_node = self.graph_info[last_node.input[-2]].node

            if max_input_node.op == "Enter":  # pragma: no cover
                min_input_parent_name = Helper.node_name_from_input(min_input_node.input[0])
                max_input_parent_name = Helper.node_name_from_input(max_input_node.input[0])
                min_input_parent_node = self.graph_info[min_input_parent_name].node
                max_input_parent_node = self.graph_info[max_input_parent_name].node
                if min_input_parent_node.op != "Const" or max_input_parent_node.op != "Const":
                    continue
                min_input_node = min_input_parent_node
                max_input_node = max_input_parent_node
            if max_filter_node.op == "Enter":  # pragma: no cover
                min_filter_parent_name = Helper.node_name_from_input(min_filter_node.input[0])
                max_filter_parent_name = Helper.node_name_from_input(max_filter_node.input[0])
                min_filter_parent_node = self.graph_info[min_filter_parent_name].node
                max_filter_parent_node = self.graph_info[max_filter_parent_name].node
                if min_filter_parent_node.op != "Const" or max_filter_parent_node.op != "Const":
                    continue
                min_filter_node = min_filter_parent_node
                max_filter_node = max_filter_parent_node
            if weight_node.op == "Enter":  # pragma: no cover
                weight_parent_name = Helper.node_name_from_input(weight_node.input[0])
                weight_parent_node = self.graph_info[weight_parent_name].node
                if weight_parent_node.op != "Const":
                    continue
                weight_node = weight_parent_node
            bias_enter_node = None
            if bias_node.op == "Enter":  # pragma: no cover
                bias_enter_node = bias_node
                bias_parent_name = Helper.node_name_from_input(bias_node.input[0])
                bias_parent_node = self.graph_info[bias_parent_name].node
                if bias_parent_node.op != "Const":
                    continue
                bias_node = bias_parent_node

            if max_filter_node.op == "Const":
                min_input_value = (min_input_node.attr["value"].tensor.float_val)[0]
                max_input_value = (max_input_node.attr["value"].tensor.float_val)[0]

                max_filter_value = (max_filter_node.attr["value"].tensor.float_val)[0]
                min_filter_value = (min_filter_node.attr["value"].tensor.float_val)[0]

                weights_tensor = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
                bias_tensor = tensor_util.MakeNdarray(bias_node.attr["value"].tensor)
                is_min_first = bool(quantized_node.attr["input_quant_mode"].s == b"MIN_FIRST")
                input_range = (
                    max_input_value - min_input_value
                    if is_min_first
                    else max(abs(max_input_value), abs(min_input_value))
                )

                if -self.eps <= input_range <= self.eps:
                    input_range += self.eps

                if -self.eps <= max_input_value - min_input_value <= self.eps:
                    max_input_value += self.eps
                int32_bias = Helper.generate_int32_bias_for_matmul(
                    bias_tensor,
                    weights_tensor,
                    input_range,
                    max_input_value,
                    min_input_value,
                    max_filter_value,
                    min_filter_value,
                )

                bias_node.attr["dtype"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                )
                bias_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            bias_tensor if self.device == "gpu" else int32_bias,
                            dtypes.float32 if self.device == "gpu" else dtypes.int32,
                            bias_tensor.shape,
                        )
                    )
                )

                bias_node.attr["value"].tensor.dtype = float32_type if self.device == "gpu" else qint32_type
                new_node.attr["Tbias"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                )
                if bias_enter_node:
                    bias_enter_node.attr["T"].CopyFrom(
                        attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                    )
            else:
                new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))
            new_node.attr["Toutput"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))

            self.graph_analyzer.remove_node(requantize_node_name)

            if self.graph_info[deq_node_name].outputs:
                self.graph_analyzer.replace_single_node(
                    new_node,
                    [top_node_name],
                    quantized_node_name,
                    self.graph_info[deq_node_name].outputs,
                    deq_node_name,
                )
                self.graph_analyzer.remove_node(deq_node_name)
            else:
                self.graph_analyzer.remove_node(deq_node_name)

                new_node.name = deq_node_name
                self.graph_analyzer.replace_single_node(
                    new_node, [top_node_name], quantized_node_name, [], deq_node_name
                )

            self.graph_analyzer.remove_node(quantized_node_name)

        return self.graph_analyzer.dump_graph()


class FuseMatMulRequantizeTransformer(GraphRewriterBase):
    """Fuse Quantized MatMul Op with the successor Requantize Op."""

    def __init__(self, model, device="cpu"):
        """Initialization."""
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model
        self.eps = 1e-05
        self.graph_info = self.graph_analyzer.parse_graph()

    def do_transformation(self):
        """Fuse the quantized op with the following requantize op.

        Returns:
            [graphdef]: the optimized graphdef object
        """
        uint8_type = dtypes.quint8.as_datatype_enum
        float32_type = dtypes.float32.as_datatype_enum
        qint32_type = dtypes.qint32.as_datatype_enum

        while True:
            target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(
                [["QuantizedMatMulWithBiasAndRelu"], ["Requantize"]]
            )
            if len(target_nodes) == 0:
                break

            i = target_nodes[0]
            quantized_node_name = i[0]
            quantized_node = self.graph_info[quantized_node_name].node
            requantize_node_name = i[1]
            requantize_node = self.graph_info[requantize_node_name].node
            requested_output_min_name = requantize_node.input[3]
            requested_output_max_name = requantize_node.input[4]

            quantized_node_op = i[-1][0]

            new_node = node_def_pb2.NodeDef()

            new_node.op = quantized_node_op + "AndRequantize"
            new_node.name = requantize_node_name
            for _, value in enumerate(quantized_node.input):
                new_node.input.append(value)
            new_node.input.append(requested_output_min_name)
            new_node.input.append(requested_output_max_name)
            if "T1" in quantized_node.attr:
                new_node.attr["T1"].CopyFrom(quantized_node.attr["T1"])
            if "T2" in quantized_node.attr:
                new_node.attr["T2"].CopyFrom(quantized_node.attr["T2"])

            parent_node_name = Helper.node_name_from_input(quantized_node.input[0])
            max_filter_node = self.graph_info[new_node.input[6]].node
            min_filter_node = self.graph_info[new_node.input[5]].node
            last_node = self.graph_info[new_node.input[0]].node

            is_min_first = bool(quantized_node.attr["input_quant_mode"].s == b"MIN_FIRST")

            weight_node = self.graph_info[new_node.input[1]].node
            bias_node = self.graph_info[new_node.input[2]].node

            max_input_node = None
            min_input_node = None
            if last_node.op.find("Requantize") != -1 or last_node.op.find("QuantizeV2") != -1:
                max_input_node = self.graph_info[last_node.input[-1]].node
                min_input_node = self.graph_info[last_node.input[-2]].node

            if max_input_node and max_input_node.op == "Enter":  # pragma: no cover
                min_input_parent_name = Helper.node_name_from_input(min_input_node.input[0])
                max_input_parent_name = Helper.node_name_from_input(max_input_node.input[0])
                min_input_parent_node = self.graph_info[min_input_parent_name].node
                max_input_parent_node = self.graph_info[max_input_parent_name].node
                if min_input_parent_node.op != "Const" or max_input_parent_node.op != "Const":
                    continue
                min_input_node = min_input_parent_node
                max_input_node = max_input_parent_node
            if max_filter_node.op == "Enter":  # pragma: no cover
                min_filter_parent_name = Helper.node_name_from_input(min_filter_node.input[0])
                max_filter_parent_name = Helper.node_name_from_input(max_filter_node.input[0])
                min_filter_parent_node = self.graph_info[min_filter_parent_name].node
                max_filter_parent_node = self.graph_info[max_filter_parent_name].node
                if min_filter_parent_node.op != "Const" or max_filter_parent_node.op != "Const":
                    continue
                min_filter_node = min_filter_parent_node
                max_filter_node = max_filter_parent_node
            if weight_node.op == "Enter":  # pragma: no cover
                weight_parent_name = Helper.node_name_from_input(weight_node.input[0])
                weight_parent_node = self.graph_info[weight_parent_name].node
                if weight_parent_node.op != "Const":
                    continue
                weight_node = weight_parent_node
            bias_enter_node = None
            if bias_node.op == "Enter":  # pragma: no cover
                bias_enter_node = bias_node
                bias_parent_name = Helper.node_name_from_input(bias_node.input[0])
                bias_parent_node = self.graph_info[bias_parent_name].node
                if bias_parent_node.op != "Const":
                    continue
                bias_node = bias_parent_node

            if last_node.op.find("Requantize") != -1 or last_node.op.find("QuantizeV2") != -1:
                min_input_value = (min_input_node.attr["value"].tensor.float_val)[0]
                max_input_value = (max_input_node.attr["value"].tensor.float_val)[0]

                max_filter_value = (max_filter_node.attr["value"].tensor.float_val)[0]
                min_filter_value = (min_filter_node.attr["value"].tensor.float_val)[0]

                weights_tensor = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
                bias_tensor = tensor_util.MakeNdarray(bias_node.attr["value"].tensor)

                input_range = (
                    max_input_value - min_input_value
                    if is_min_first
                    else max(abs(max_input_value), abs(min_input_value))
                )

                if -self.eps <= input_range <= self.eps:
                    input_range += self.eps

                if -self.eps <= max_input_value - min_input_value <= self.eps:
                    max_input_value += self.eps
                int32_bias = Helper.generate_int32_bias_for_matmul(
                    bias_tensor,
                    weights_tensor,
                    input_range,
                    max_input_value,
                    min_input_value,
                    max_filter_value,
                    min_filter_value,
                )
                bias_node.attr["dtype"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                )
                bias_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            bias_tensor if self.device == "gpu" else int32_bias,
                            dtypes.float32 if self.device == "gpu" else dtypes.int32,
                            bias_tensor.shape,
                        )
                    )
                )

                bias_node.attr["value"].tensor.dtype = float32_type if self.device == "gpu" else qint32_type
                new_node.attr["Tbias"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                )

                new_node.attr["Toutput"].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))
                # TODO enabled below commit once the graph refactor pre_optimize committed.
                if quantized_node_op.find("Relu") == -1:
                    deq_node_name = self.graph_info[requantize_node_name].outputs[0]
                    deq_node = self.graph_info[deq_node_name].node
                    deq_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))
                if bias_enter_node:
                    bias_enter_node.attr["T"].CopyFrom(
                        attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                    )
            else:
                new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))

            self.graph_analyzer.replace_single_node(
                new_node,
                [parent_node_name],
                quantized_node_name,
                [self.graph_info[requantize_node_name].outputs[0]],
                requantize_node_name,
            )
            self.graph_analyzer.remove_node(quantized_node_name)

        return self.graph_analyzer.dump_graph()


class FuseMatMulRequantizeDequantizeNewAPITransformer(GraphRewriterBase):  # pragma: no cover
    """Fuse _QuantizedMatMul + Requantize + Dequantize into _QuantizedMatMul."""

    def __init__(self, model, device="cpu"):
        """Initialization."""
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model

        self.graph_info = self.graph_analyzer.parse_graph()

        self.eps = 1e-5

    def do_transformation(self):
        """Apply the fusion of QuantizedMatMul + Requantize + Dequantize."""
        fuse_pattern = [["_QuantizedMatMul"], ["Requantize", "RequantizePerChannel"], ["Dequantize"], ("Softmax",)]

        uint8_type = dtypes.quint8.as_datatype_enum
        int8_type = dtypes.qint8.as_datatype_enum
        float32_type = dtypes.float32.as_datatype_enum
        qint32_type = dtypes.qint32.as_datatype_enum
        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(fuse_pattern)
        for i in target_nodes:
            quantized_node_name = i[0]
            quantized_node = self.graph_info[quantized_node_name].node
            requantize_node_name = i[1]
            requantize_node = self.graph_info[requantize_node_name].node
            requested_output_min_name = requantize_node.input[3]
            requested_output_max_name = requantize_node.input[4]
            deq_node_name = i[2]

            quantized_node_op = i[-1][0]

            # "BiasAdd" + "Add" only supports "Dequantize"
            attr_fused_ops = "".join(
                x
                for x in quantized_node.attr["fused_ops"].SerializeToString().decode("UTF-8", "ignore").strip()
                if x.isprintable()
            )
            if "BiasAddAdd" not in attr_fused_ops:
                continue

            new_node = node_def_pb2.NodeDef()

            new_node.op = quantized_node_op
            new_node.name = requantize_node_name
            for _, value in enumerate(quantized_node.input):
                new_node.input.append(value)

            new_node.input.append(requested_output_min_name)
            new_node.input.append(requested_output_max_name)
            if "T1" in quantized_node.attr:
                new_node.attr["T1"].CopyFrom(quantized_node.attr["T1"])
            if "T2" in quantized_node.attr:
                new_node.attr["T2"].CopyFrom(quantized_node.attr["T2"])
            if "U" in quantized_node.attr:
                new_node.attr["U"].CopyFrom(quantized_node.attr["U"])
            if "transpose_b" in quantized_node.attr:
                new_node.attr["transpose_b"].CopyFrom(quantized_node.attr["transpose_b"])
            if "transpose_a" in quantized_node.attr:
                new_node.attr["transpose_a"].CopyFrom(quantized_node.attr["transpose_a"])
            if "input_quant_mode" in quantized_node.attr:
                new_node.attr["input_quant_mode"].CopyFrom(quantized_node.attr["input_quant_mode"])
            if "output_quant_mode" in quantized_node.attr:
                new_node.attr["output_quant_mode"].CopyFrom(quantized_node.attr["output_quant_mode"])

            top_node_name = Helper.node_name_from_input(quantized_node.input[0])
            max_filter_node = None
            min_filter_node = None

            # MatMul + BiasAdd + Add
            # The Min and Max of non-const weight node are from QuantizeV2's output, not valid nodes.
            # Add check here for excluding this case.
            if ":2" not in new_node.input[7]:
                max_filter_node = self.graph_info[new_node.input[7]].node
            if ":1" not in new_node.input[6]:
                min_filter_node = self.graph_info[new_node.input[6]].node
            last_node = self.graph_info[new_node.input[0]].node
            weight_node = self.graph_info[Helper.node_name_from_input(new_node.input[1])].node
            bias_node = self.graph_info[Helper.node_name_from_input(new_node.input[2])].node
            if not last_node.op == "QuantizedConcatV2":
                max_input_node = self.graph_info[last_node.input[-1]].node
                min_input_node = self.graph_info[last_node.input[-2]].node

            type_bias = float32_type
            if not last_node.op == "QuantizedConcatV2" and max_input_node.op == "Enter":  # pragma: no cover
                min_input_parent_name = Helper.node_name_from_input(min_input_node.input[0])
                max_input_parent_name = Helper.node_name_from_input(max_input_node.input[0])
                min_input_parent_node = self.graph_info[min_input_parent_name].node
                max_input_parent_node = self.graph_info[max_input_parent_name].node
                if min_input_parent_node.op != "Const" or max_input_parent_node.op != "Const":
                    continue
                min_input_node = min_input_parent_node
                max_input_node = max_input_parent_node
            if max_filter_node and min_filter_node and max_filter_node.op == "Enter":  # pragma: no cover
                min_filter_parent_name = Helper.node_name_from_input(min_filter_node.input[0])
                max_filter_parent_name = Helper.node_name_from_input(max_filter_node.input[0])
                min_filter_parent_node = self.graph_info[min_filter_parent_name].node
                max_filter_parent_node = self.graph_info[max_filter_parent_name].node
                if min_filter_parent_node.op != "Const" or max_filter_parent_node.op != "Const":
                    continue
                min_filter_node = min_filter_parent_node
                max_filter_node = max_filter_parent_node
            if weight_node.op == "Enter":  # pragma: no cover
                weight_parent_name = Helper.node_name_from_input(weight_node.input[0])
                weight_parent_node = self.graph_info[weight_parent_name].node
                if weight_parent_node.op != "Const":
                    continue
                weight_node = weight_parent_node
            bias_enter_node = None
            if bias_node.op == "Enter":  # pragma: no cover
                bias_enter_node = bias_node
                bias_parent_name = Helper.node_name_from_input(bias_node.input[0])
                bias_parent_node = self.graph_info[bias_parent_name].node
                if bias_parent_node.op != "Const":
                    continue
                bias_node = bias_parent_node

            if (
                max_filter_node
                and min_filter_node
                and max_filter_node.op == "Const"
                and weight_node.op == "Const"
                and not last_node.op == "QuantizedConcatV2"
            ):
                min_input_value = (min_input_node.attr["value"].tensor.float_val)[0]
                max_input_value = (max_input_node.attr["value"].tensor.float_val)[0]
                if requantize_node.op.find("PerChannel") != -1:  # pragma: no cover
                    max_filter_tensor = tensor_util.MakeNdarray(max_filter_node.attr["value"].tensor)  # get tensor
                    min_filter_tensor = tensor_util.MakeNdarray(min_filter_node.attr["value"].tensor)  # get tensor
                else:
                    max_filter_value = (max_filter_node.attr["value"].tensor.float_val)[0]
                    min_filter_value = (min_filter_node.attr["value"].tensor.float_val)[0]

                weights_tensor = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
                bias_tensor = tensor_util.MakeNdarray(bias_node.attr["value"].tensor)
                is_min_first = bool(quantized_node.attr["input_quant_mode"].s == b"MIN_FIRST")
                input_range = (
                    max_input_value - min_input_value
                    if is_min_first
                    else max(abs(max_input_value), abs(min_input_value))
                )

                if -self.eps <= input_range <= self.eps:
                    input_range += self.eps

                if -self.eps <= max_input_value - min_input_value <= self.eps:
                    max_input_value += self.eps
                if requantize_node.op.find("PerChannel") != -1:  # pragma: no cover
                    int32_bias = Helper.generate_int32_bias_for_matmul_per_channel(
                        bias_tensor,
                        weights_tensor,
                        max_input_value,
                        min_input_value,
                        max_filter_tensor,
                        min_filter_tensor,
                    )
                else:
                    int32_bias = Helper.generate_int32_bias_for_matmul(
                        bias_tensor,
                        weights_tensor,
                        input_range,
                        max_input_value,
                        min_input_value,
                        max_filter_value,
                        min_filter_value,
                    )

                bias_node.attr["dtype"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                )
                bias_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            bias_tensor if self.device == "gpu" else int32_bias,
                            dtypes.float32 if self.device == "gpu" else dtypes.int32,
                            bias_tensor.shape,
                        )
                    )
                )

                bias_node.attr["value"].tensor.dtype = float32_type if self.device == "gpu" else qint32_type
                type_bias = float32_type if self.device == "gpu" else qint32_type
                if bias_enter_node:
                    bias_enter_node.attr["T"].CopyFrom(
                        attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                    )
            else:
                type_bias = float32_type

            new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=type_bias))

            Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"Add", b"Dequantize"])
            Helper.set_attr_type_list(
                new_node,
                "Thost_inputs",
                [
                    uint8_type,
                    int8_type,
                    type_bias,
                    float32_type,
                    float32_type,
                    float32_type,
                    float32_type,
                    float32_type,
                    float32_type,
                    float32_type,
                ],
            )

            Helper.set_attr_type_list(new_node, "Thost_outputs", [float32_type])
            new_node.attr["Tout"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))

            self.graph_analyzer.remove_node(requantize_node_name)

            if self.graph_info[deq_node_name].outputs:
                self.graph_analyzer.replace_single_node(
                    new_node,
                    [top_node_name],
                    quantized_node_name,
                    self.graph_info[deq_node_name].outputs,
                    deq_node_name,
                )
                self.graph_analyzer.remove_node(deq_node_name)
            else:
                self.graph_analyzer.remove_node(deq_node_name)

                new_node.name = deq_node_name
                self.graph_analyzer.replace_single_node(
                    new_node, [top_node_name], quantized_node_name, [], deq_node_name
                )

            self.graph_analyzer.remove_node(quantized_node_name)

        return self.graph_analyzer.dump_graph()


class FuseMatMulRequantizeNewAPITransformer(GraphRewriterBase):
    """Fuse newAPI Quantized MatMul Op with the successor Requantize Op."""

    def __init__(self, model, device="cpu"):
        """Initialization."""
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model
        self.eps = 1e-05
        self.graph_info = self.graph_analyzer.parse_graph()

    def do_transformation(self):
        """Fuse the quantized op with the following requantize op.

        Returns:
            [graphdef]: the optimized graphdef object
        """
        uint8_type = dtypes.quint8.as_datatype_enum
        int8_type = dtypes.qint8.as_datatype_enum
        float32_type = dtypes.float32.as_datatype_enum
        qint32_type = dtypes.qint32.as_datatype_enum

        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(
            [["_QuantizedMatMul"], ["Requantize", "RequantizePerChannel"]]
        )
        for i in target_nodes:
            quantized_node_name = i[0]
            quantized_node = self.graph_info[quantized_node_name].node
            requantize_node_name = i[1]
            requantize_node = self.graph_info[requantize_node_name].node
            requested_output_min_name = requantize_node.input[3]
            requested_output_max_name = requantize_node.input[4]

            quantized_node_op = i[-1][0]
            attr_fused_ops = "".join(
                x
                for x in quantized_node.attr["fused_ops"].SerializeToString().decode("UTF-8", "ignore").strip()
                if x.isprintable()
            )
            # "Requantize"
            # "BiasAdd", "Requantize"
            # "BiasAdd", "Activation", "Requantize"
            if "BiasAddAdd" in attr_fused_ops:
                continue
            new_node = node_def_pb2.NodeDef()

            new_node.op = quantized_node_op
            new_node.name = requantize_node_name
            for _, value in enumerate(quantized_node.input):
                new_node.input.append(value)
            new_node.input.append(requested_output_min_name)
            new_node.input.append(requested_output_max_name)

            if "transpose_b" in quantized_node.attr:
                new_node.attr["transpose_b"].CopyFrom(quantized_node.attr["transpose_b"])
            if "transpose_a" in quantized_node.attr:
                new_node.attr["transpose_a"].CopyFrom(quantized_node.attr["transpose_a"])
            if "T1" in quantized_node.attr:
                new_node.attr["T1"].CopyFrom(quantized_node.attr["T1"])
            if "T2" in quantized_node.attr:
                new_node.attr["T2"].CopyFrom(quantized_node.attr["T2"])
            if "U" in quantized_node.attr:
                new_node.attr["U"].CopyFrom(quantized_node.attr["U"])
            if "input_quant_mode" in quantized_node.attr:
                new_node.attr["input_quant_mode"].CopyFrom(quantized_node.attr["input_quant_mode"])
            if "output_quant_mode" in quantized_node.attr:
                new_node.attr["output_quant_mode"].CopyFrom(quantized_node.attr["output_quant_mode"])

            parent_node_name = Helper.node_name_from_input(quantized_node.input[0])
            max_filter_node = None
            min_filter_node = None
            # The Min and Max of non-const weight node are from QuantizeV2's output, not valid nodes.
            # Add check here for excluding this case.
            if len(attr_fused_ops) == 0:  # single matmul case
                if ":2" not in new_node.input[5]:
                    max_filter_node = self.graph_info[new_node.input[5]].node
                if ":1" not in new_node.input[4]:
                    min_filter_node = self.graph_info[new_node.input[4]].node
            else:
                if ":2" not in new_node.input[6]:
                    max_filter_node = self.graph_info[new_node.input[6]].node
                if ":1" not in new_node.input[5]:
                    min_filter_node = self.graph_info[new_node.input[5]].node
            last_node = self.graph_info[new_node.input[0]].node
            is_min_first = bool(quantized_node.attr["input_quant_mode"].s == b"MIN_FIRST")
            weight_node = self.graph_info[new_node.input[1]].node
            bias_node = None
            if "BiasAdd" in attr_fused_ops:
                bias_node = self.graph_info[new_node.input[2]].node
            if not last_node.op == "QuantizedConcatV2":
                max_input_node = self.graph_info[last_node.input[-1]].node
                min_input_node = self.graph_info[last_node.input[-2]].node

            if not last_node.op == "QuantizedConcatV2" and max_input_node.op == "Enter":  # pragma: no cover
                min_input_parent_name = Helper.node_name_from_input(min_input_node.input[0])
                max_input_parent_name = Helper.node_name_from_input(max_input_node.input[0])
                min_input_parent_node = self.graph_info[min_input_parent_name].node
                max_input_parent_node = self.graph_info[max_input_parent_name].node
                if min_input_parent_node.op != "Const" or max_input_parent_node.op != "Const":
                    continue
                min_input_node = min_input_parent_node
                max_input_node = max_input_parent_node
            if (
                max_filter_node
                and min_filter_node
                and min_filter_node.input
                and max_filter_node.input
                and max_filter_node.op == "Enter"
            ):  # pragma: no cover
                min_filter_parent_name = Helper.node_name_from_input(min_filter_node.input[0])
                max_filter_parent_name = Helper.node_name_from_input(max_filter_node.input[0])
                min_filter_parent_node = self.graph_info[min_filter_parent_name].node
                max_filter_parent_node = self.graph_info[max_filter_parent_name].node
                if min_filter_parent_node.op != "Const" or max_filter_parent_node.op != "Const":
                    continue
                min_filter_node = min_filter_parent_node
                max_filter_node = max_filter_parent_node
            if weight_node.op == "Enter":  # pragma: no cover
                weight_parent_name = Helper.node_name_from_input(weight_node.input[0])
                weight_parent_node = self.graph_info[weight_parent_name].node
                if weight_parent_node.op != "Const":
                    continue
                weight_node = weight_parent_node
            bias_enter_node = None
            if bias_node and bias_node.op == "Enter":  # pragma: no cover
                bias_enter_node = bias_node
                bias_parent_name = Helper.node_name_from_input(bias_node.input[0])
                bias_parent_node = self.graph_info[bias_parent_name].node
                if bias_parent_node.op != "Const":
                    continue
                bias_node = bias_parent_node

            if bias_node and (
                last_node.op.find("_QuantizedMatMul") != -1
                or last_node.op.find("QuantizeV2") != -1
                and max_filter_node
                and min_filter_node
            ):
                min_input_value = (min_input_node.attr["value"].tensor.float_val)[0]
                max_input_value = (max_input_node.attr["value"].tensor.float_val)[0]
                if requantize_node.op.find("PerChannel") != -1:  # pragma: no cover
                    max_filter_tensor = tensor_util.MakeNdarray(max_filter_node.attr["value"].tensor)  # get tensor
                    min_filter_tensor = tensor_util.MakeNdarray(min_filter_node.attr["value"].tensor)  # get tensor
                else:
                    max_filter_value = (max_filter_node.attr["value"].tensor.float_val)[0]
                    min_filter_value = (min_filter_node.attr["value"].tensor.float_val)[0]

                weights_tensor = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
                bias_tensor = tensor_util.MakeNdarray(bias_node.attr["value"].tensor)

                input_range = (
                    max_input_value - min_input_value
                    if is_min_first
                    else max(abs(max_input_value), abs(min_input_value))
                )
                if -self.eps <= input_range <= self.eps:
                    input_range += self.eps

                if -self.eps <= max_input_value - min_input_value <= self.eps:
                    max_input_value += self.eps
                if requantize_node.op.find("PerChannel") != -1:  # pragma: no cover
                    int32_bias = Helper.generate_int32_bias_for_matmul_per_channel(
                        bias_tensor,
                        weights_tensor,
                        max_input_value,
                        min_input_value,
                        max_filter_tensor,
                        min_filter_tensor,
                    )
                else:
                    int32_bias = Helper.generate_int32_bias_for_matmul(
                        bias_tensor,
                        weights_tensor,
                        input_range,
                        max_input_value,
                        min_input_value,
                        max_filter_value,
                        min_filter_value,
                    )
                bias_node.attr["dtype"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                )
                bias_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            bias_tensor if self.device == "gpu" else int32_bias,
                            dtypes.float32 if self.device == "gpu" else dtypes.int32,
                            bias_tensor.shape,
                        )
                    )
                )

                bias_node.attr["value"].tensor.dtype = float32_type if self.device == "gpu" else qint32_type
                if bias_enter_node:
                    bias_enter_node.attr["T"].CopyFrom(
                        attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                    )
                new_node.attr["Tbias"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type if self.device == "gpu" else qint32_type)
                )

                deq_node_name = self.graph_info[requantize_node_name].outputs[0]
                deq_node = self.graph_info[deq_node_name].node
                deq_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))

                Helper.set_attr_type_list(
                    new_node,
                    "Thost_inputs",
                    [
                        uint8_type,
                        int8_type,
                        float32_type if self.device == "gpu" else qint32_type,
                        float32_type,
                        float32_type,
                        float32_type,
                        float32_type,
                        float32_type,
                        float32_type,
                    ],
                )
            else:
                new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))
                deq_node_name = self.graph_info[requantize_node_name].outputs[0]
                deq_node = self.graph_info[deq_node_name].node
                deq_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))
                if bias_node:
                    Helper.set_attr_type_list(
                        new_node,
                        "Thost_inputs",
                        [
                            uint8_type,
                            int8_type,
                            float32_type,
                            float32_type,
                            float32_type,
                            float32_type,
                            float32_type,
                            float32_type,
                            float32_type,
                        ],
                    )
                else:
                    Helper.set_attr_type_list(
                        new_node,
                        "Thost_inputs",
                        [
                            uint8_type,
                            int8_type,
                            float32_type,
                            float32_type,
                            float32_type,
                            float32_type,
                            float32_type,
                            float32_type,
                        ],
                    )

            Helper.set_attr_type_list(new_node, "Thost_outputs", [uint8_type, float32_type, float32_type])

            if "GeluApproximate" in attr_fused_ops:
                Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"GeluApproximate", b"Requantize"])
            elif "GeluExact" in attr_fused_ops:
                Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"GeluExact", b"Requantize"])
            elif "Elu" in attr_fused_ops:
                Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"Elu", b"Requantize"])
            elif "LeakyRelu" in attr_fused_ops:
                Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"LeakyRelu", b"Requantize"])
            elif "Relu6" in attr_fused_ops:
                Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"Relu6", b"Requantize"])
            elif "Tanh" in attr_fused_ops:
                Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"Tanh", b"Requantize"])
            elif "Sigmoid" in attr_fused_ops:
                Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"Sigmoid", b"Requantize"])
            elif "Relu" in attr_fused_ops:
                Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"Relu", b"Requantize"])
            elif "BiasAdd" in attr_fused_ops:
                Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"Requantize"])
            else:
                Helper.set_attr_string_list(new_node, "fused_ops", [b"Requantize"])
            new_node.attr["Tout"].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))

            self.graph_analyzer.replace_single_node(
                new_node,
                [parent_node_name],
                quantized_node_name,
                [self.graph_info[requantize_node_name].outputs[0]],
                requantize_node_name,
            )
            self.graph_analyzer.remove_node(quantized_node_name)

        return self.graph_analyzer.dump_graph()
