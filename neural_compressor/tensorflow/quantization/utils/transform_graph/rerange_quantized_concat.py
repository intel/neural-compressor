#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2021 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""ConcatV2 rerange transform."""

from __future__ import absolute_import, division, print_function

import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper

from .graph_transform_base import GraphTransformBase


class RerangeQuantizedConcat(GraphTransformBase):
    """This class implements the rerange_quantize concat graph transform."""

    fused_requantized_bias_op = (
        "QuantizedConv2DWithBiasAndRequantize",
        "QuantizedConv2DWithBiasAndReluAndRequantize",
        "QuantizedConv2DWithBiasSumAndReluAndRequantize",
        "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
    )
    fuse_requantized_bias_op_new_api = (
        [b"BiasAdd", b"Requantize"],
        [b"BiasAdd", b"Relu", b"Requantize"],
        [b"BiasAdd", b"LeakyRelu", b"Requantize"],
        [b"BiasAdd", b"Sum", b"Relu", b"Requantize"],
        [b"BiasAdd", b"Sum", b"Requantize"],
        [b"BiasAdd", b"Elu", b"Requantize"],
    )
    fuse_requantized_relu_op_new_api = (
        [b"BiasAdd", b"Relu", b"Requantize"],
        [b"BiasAdd", b"LeakyRelu", b"Requantize"],
        # [b'BiasAdd', b'Sum',  b'Relu',  b'Requantize']
        [b"BiasAdd", b"Elu", b"Requantize"],
    )
    offset_map = {
        "QuantizedConv2DAndRequantize": 6,
        "QuantizedConv2DAndReluAndRequantize": 6,
        "QuantizedConv2DWithBiasAndRequantize": 7,
        "QuantizedConv2DWithBiasAndReluAndRequantize": 7,
        "QuantizedConv2DWithBiasSumAndReluAndRequantize": 7,
        "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize": 7,
    }
    offset_map_new_api = {
        str([b"Requantize"]): 6,
        str([b"Relu", b"Requantize"]): 6,
        str([b"LeakyRelu", b"Requantize"]): 6,
        str([b"BiasAdd", b"Requantize"]): 7,
        str([b"BiasAdd", b"Relu", b"Requantize"]): 7,
        str([b"BiasAdd", b"LeakyRelu", b"Requantize"]): 7,
        str([b"BiasAdd", b"Elu", b"Requantize"]): 7,
        str([b"BiasAdd", b"Sum", b"Relu", b"Requantize"]): 10,
        str([b"BiasAdd", b"Sum", b"Requantize"]): 10,
    }

    def __init__(self, input_pb, device, performance_only=False):
        """Initialization."""
        super().__init__(input_pb)

        self.parse_input_pb()
        self.concat_node_input_mapping = {}
        self.rerange_concat_node = []
        self.device = device
        self.performance_only = performance_only

    def _analyze_concat_node_recursively(self, quantized_conv_nodes, input_node):
        """Analyze all the ConcatV2 nodes recursively."""
        op_type = input_node.op
        if op_type == "QuantizedConcatV2":
            can_rerange = True
            concat_input_num = input_node.attr["N"].i
            for index in range(concat_input_num):
                concat_input_node = self.node_mapping[self.get_node_name_from_input(input_node.input[index])]
                concat_input_node_op_type = concat_input_node.op
                if concat_input_node_op_type in self.offset_map:
                    quantized_conv_nodes.append(concat_input_node)
                elif (
                    concat_input_node.op == "_FusedQuantizedConv2D"
                    and "fused_ops" in concat_input_node.attr
                    and str(concat_input_node.attr["fused_ops"].list.s) in self.offset_map_new_api
                ):
                    quantized_conv_nodes.append(concat_input_node)
                elif concat_input_node_op_type in ("QuantizedMaxPool", "QuantizedAvgPool"):
                    another_concat_node = self.node_mapping[self.get_node_name_from_input(concat_input_node.input[0])]
                    if self.performance_only:
                        if (
                            another_concat_node.op == "_FusedQuantizedConv2D"
                            and "fused_ops" in another_concat_node.attr
                            and str(another_concat_node.attr["fused_ops"].list.s) in self.offset_map_new_api
                        ):
                            quantized_conv_nodes.append(another_concat_node)
                    else:
                        if not self._analyze_concat_node_recursively(quantized_conv_nodes, another_concat_node):
                            can_rerange = False
                            break
                elif concat_input_node_op_type == "QuantizedConcatV2":
                    if not self._analyze_concat_node_recursively(quantized_conv_nodes, concat_input_node):
                        can_rerange = False
                        break
                elif self.performance_only and concat_input_node_op_type == "QuantizeV2":
                    quantized_conv_nodes.append(concat_input_node)
                else:
                    can_rerange = False
                    break

            return can_rerange
        elif op_type == "QuantizedConv2DWithBiasAndReluAndRequantize" or (
            input_node.op == "_FusedQuantizedConv2D"
            and "fused_ops" in input_node.attr
            and input_node.attr["fused_ops"].list.s in self.fuse_requantized_relu_op_new_api
        ):
            can_rerange = True
            quantized_conv_nodes.append(input_node)
            return can_rerange
        else:
            return False

    def do_transformation(self):
        """Apply the rerange quantized ConcatV2 transform."""
        for _, node in enumerate(self.input_graph.node):
            if node.op != "QuantizedConcatV2":
                continue
            quantized_conv_nodes = []
            can_rerange = self._analyze_concat_node_recursively(quantized_conv_nodes, node)
            if not can_rerange:
                continue

            self.rerange_concat_node.append(node.name)
            combined_min = np.finfo(np.float64).max
            combined_max = -combined_min

            for node in quantized_conv_nodes:
                offset_value = 6
                if (
                    node.op == "_FusedQuantizedConv2D"
                    and "fused_ops" in node.attr
                    and str(node.attr["fused_ops"].list.s) in self.offset_map_new_api
                ):
                    offset_value = self.offset_map_new_api[str(node.attr["fused_ops"].list.s)]
                elif self.performance_only and node.op == "QuantizeV2":
                    offset_value = 1
                else:
                    offset_value = self.offset_map[node.op]
                min_value_node = self.node_mapping[node.input[offset_value]]
                max_value_node = self.node_mapping[node.input[offset_value + 1]]
                min_value = min_value_node.attr["value"].tensor.float_val[0]
                max_value = max_value_node.attr["value"].tensor.float_val[0]
                if min_value < combined_min:
                    combined_min = min_value
                if max_value > combined_max:
                    combined_max = max_value

            if self.performance_only:
                combined_value = max(abs(combined_min), abs(combined_max))
                combined_min = -combined_value
                combined_max = combined_value

            for node in quantized_conv_nodes:
                offset_value = 6
                if (
                    node.op == "_FusedQuantizedConv2D"
                    and "fused_ops" in node.attr
                    and str(node.attr["fused_ops"].list.s) in self.offset_map_new_api
                ):
                    offset_value = self.offset_map_new_api[str(node.attr["fused_ops"].list.s)]
                elif self.performance_only and node.op == "QuantizeV2":
                    offset_value = 1
                else:
                    offset_value = self.offset_map[node.op]

                min_value_node = self.node_mapping[node.input[offset_value]]
                max_value_node = self.node_mapping[node.input[offset_value + 1]]
                min_value_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(float(combined_min), dtypes.float32, [])
                    )
                )

                max_value_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(float(combined_max), dtypes.float32, [])
                    )
                )
        if self.device == "cpu":
            self._update_bias()
        return self.input_graph

    def _update_bias(self):
        """Convert the bias from float to int."""
        for node_name in self.node_mapping:
            current_node = self.node_mapping[node_name]
            current_node_op = current_node.op
            if (current_node_op in self.fused_requantized_bias_op) or (
                current_node_op == "_FusedQuantizedConv2D"
                and "fused_ops" in current_node.attr
                and current_node.attr["fused_ops"].list.s in self.fuse_requantized_bias_op_new_api
            ):
                done = False
                another_conv_node = None
                original_conv_node = current_node
                while not done:
                    current_node = self.node_mapping[self.get_node_name_from_input(current_node.input[0])]
                    if current_node.op in self.offset_map:
                        another_conv_node = current_node
                        done = True
                    elif (
                        current_node.op == "_FusedQuantizedConv2D"
                        and "fused_ops" in current_node.attr
                        and str(current_node.attr["fused_ops"].list.s) in self.offset_map_new_api
                    ):
                        another_conv_node = current_node
                        done = True
                    elif current_node.op == "QuantizedConcatV2":
                        if current_node.name not in self.rerange_concat_node:
                            done = True
                    elif current_node.op not in ("QuantizedMaxPool", "QuantizedAvgPool"):
                        done = True

                if not another_conv_node:
                    continue

                bias_node = self.node_mapping[self.get_node_name_from_input(original_conv_node.input[2])]
                bias_node_type = original_conv_node.attr["Tbias"]

                if bias_node_type.type != dtypes.float32 or bias_node_type.type == dtypes.qint32:
                    continue
                sum_off_set = 0
                if original_conv_node.op == "_FusedQuantizedConv2D":
                    if str(original_conv_node.attr["fused_ops"].list.s) == str(
                        [b"BiasAdd", b"Sum", b"Relu", b"Requantize"]
                    ) or str(original_conv_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Sum", b"Requantize"]):
                        sum_off_set = 1
                    # else:
                    #    print(str(original_conv_node.attr['fused_ops'].list.s))
                min_filter_node = self.node_mapping[original_conv_node.input[5 + sum_off_set]]
                max_filter_node = self.node_mapping[original_conv_node.input[6 + sum_off_set]]

                channel_size = (
                    1
                    if not min_filter_node.attr["value"].tensor.tensor_shape.dim
                    else min_filter_node.attr["value"].tensor.tensor_shape.dim[0].size
                )

                if channel_size == 1:
                    max_filter_tensor = []
                    min_filter_tensor = []
                    max_filter_tensor.append((max_filter_node.attr["value"].tensor.float_val)[0])
                    min_filter_tensor.append((min_filter_node.attr["value"].tensor.float_val)[0])
                else:
                    max_filter_tensor = tensor_util.MakeNdarray(max_filter_node.attr["value"].tensor)
                    min_filter_tensor = tensor_util.MakeNdarray(min_filter_node.attr["value"].tensor)

                offset_value = 6
                if (
                    another_conv_node.op == "_FusedQuantizedConv2D"
                    and "fused_ops" in another_conv_node.attr
                    and str(another_conv_node.attr["fused_ops"].list.s) in self.offset_map_new_api
                ):
                    offset_value = self.offset_map_new_api[str(another_conv_node.attr["fused_ops"].list.s)]
                else:
                    offset_value = self.offset_map[another_conv_node.op]

                min_freezed_output_node = self.node_mapping[another_conv_node.input[offset_value]]
                max_freezed_output_node = self.node_mapping[another_conv_node.input[offset_value + 1]]
                min_input = min_freezed_output_node.attr["value"].tensor.float_val[0]
                max_input = max_freezed_output_node.attr["value"].tensor.float_val[0]
                # To avoid generating int32 bias exception for corner case
                if min_input == 0 and max_input == 0:
                    continue

                bias_tensor = tensor_util.MakeNdarray(bias_node.attr["value"].tensor)

                activation_range = 127.0 if current_node.attr["out_type"].type == dtypes.qint8 else 255.0

                int32_bias = Helper.generate_int32_bias_for_conv(
                    bias_tensor,
                    channel_size,
                    max_input,
                    min_input,
                    max_filter_tensor,
                    min_filter_tensor,
                    activation_range,
                )

                original_conv_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.qint32.as_datatype_enum))
                bias_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.qint32.as_datatype_enum))

                bias_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(int32_bias, dtypes.int32, bias_tensor.shape)
                    )
                )
                bias_node.attr["value"].tensor.dtype = dtypes.qint32.as_datatype_enum
                if "Thost_inputs" in original_conv_node.attr:
                    original_conv_node.attr["Thost_inputs"].list.type[2] = original_conv_node.attr["Tbias"].type
