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
"""Fuse QuantizedConv Requantize/Dequantize Graph Rewriter."""

import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper

from ..graph_base import GraphRewriterBase


class FuseConvRequantizeTransformer(GraphRewriterBase):
    """Fuse Quantized Conv Op with the successor Requantize Op."""

    fuse_patterns = [
        [
            "QuantizedConv2DWithBiasAndRelu",
            "QuantizedDepthwiseConv2DWithBiasAndRelu",
            "QuantizedConv2DWithBias",
            "QuantizedDepthwiseConv2DWithBias",
            "_FusedQuantizedConv2D",
            "_FusedQuantizedDepthwiseConv2D",
            "_FusedQuantizedConv3D",
            "_FusedQuantizedDeconv2D",
            "_FusedQuantizedDeconv3D",
        ],
        ["RequantizePerChannel", "Requantize"],
        ("Dequantize",),
    ]

    fuse_sum_op_types = (
        [b"BiasAdd", b"Sum"],
        [b"BiasAdd", b"Sum", b"Relu"],
        [b"BiasAdd", b"Sum", b"LeakyRelu"],
        [b"BiasAdd", b"Relu", b"Sum"],
        [b"BiasAdd", b"LeakyRelu", b"Sum"],
    )

    sum_pattern = [
        [
            "QuantizedConv2DWithBiasSumAndRelu",
            "QuantizedConv2DWithBiasReluAndSum",
            "_FusedQuantizedDepthwiseConv2D",
            "_FusedQuantizedConv2D",
            "_FusedQuantizedConv3D",
        ],
        ["RequantizePerChannel", "Requantize"],
    ]

    def __init__(self, model, device="cpu", new_api=False):
        """Initialization."""
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model
        self.graph_info = self.graph_analyzer.parse_graph()
        self.fused_ops = []
        self.output_types = []
        self.new_api = new_api

    def do_transformation(self):
        """Fuse the quantized op with the following requantize op.

        The transformation has two stages, the first step is to fuse the patterns
        defined in self.fuse_patterns and the last step is to fuse the self.sum_patterns.

        Returns:
            [graphdef]: the optimized graphdef object
        """
        int8_type = dtypes.qint8.as_datatype_enum
        uint8_type = dtypes.quint8.as_datatype_enum
        float32_type = dtypes.float32.as_datatype_enum
        qint32_type = dtypes.qint32.as_datatype_enum
        dtype_map_dict = {
            dtypes.qint8.as_datatype_enum: dtypes.qint8,
            dtypes.quint8.as_datatype_enum: dtypes.quint8,
            dtypes.float32.as_datatype_enum: dtypes.float32,
            dtypes.qint32.as_datatype_enum: dtypes.qint32,
        }
        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(self.fuse_patterns)

        for i in target_nodes:
            quantized_node_name = i[0]
            quantized_node = self.graph_info[quantized_node_name].node
            if not self.new_api and quantized_node.op == "QuantizedDepthwiseConv2DWithBias":
                continue
            if i[-1][0] in (
                "_FusedQuantizedDepthwiseConv2D",
                "_FusedQuantizedConv2D",
                "_FusedQuantizedConv3D",
                "_FusedQuantizedDeconv2D",
                "_FusedQuantizedDeconv3D",
            ):
                if str(quantized_node.attr["fused_ops"].list.s).find("Sum") != -1:
                    continue
                # else:
                #   print(quantized_node.attr['fused_ops'].list.s)
            requantize_node_name = i[1]
            requantize_node = self.graph_info[requantize_node_name].node
            if (
                i[-1][-1] == "Dequantize"
                and self.new_api
                and i[0]
                in (
                    "_FusedQuantizedDepthwiseConv2D",
                    "_FusedQuantizedConv2D",
                    "_FusedQuantizedConv3D",
                    "_FusedQuantizedDeconv2D",
                    "_FusedQuantizedDeconv3D",
                )
            ):
                dequantize_node_name = i[2]
            else:
                dequantize_node_name = None
            requested_output_min_name = requantize_node.input[3]
            requested_output_max_name = requantize_node.input[4]

            quantized_node_op = i[-1][0]

            new_node = node_def_pb2.NodeDef()
            if self.new_api:
                if i[-1][0] == "QuantizedConv2DWithBiasAndRelu":
                    new_node.op = "_FusedQuantizedConv2D"
                    self.fused_ops = [b"BiasAdd", b"Relu", b"Requantize"]
                    self.output_types = [
                        requantize_node.attr["out_type"].type,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ]
                elif i[-1][0] == "QuantizedConv2DWithBias":
                    new_node.op = "_FusedQuantizedConv2D"
                    self.fused_ops = [b"BiasAdd", b"Requantize"]
                    self.output_types = [
                        requantize_node.attr["out_type"].type,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ]
                elif i[-1][0] == "QuantizedDepthwiseConv2DWithBias":
                    new_node.op = "_FusedQuantizedDepthwiseConv2D"
                    self.fused_ops = [b"BiasAdd", b"Requantize"]
                    self.output_types = [
                        requantize_node.attr["out_type"].type,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ]
                elif i[-1][0] == "QuantizedDepthwiseConv2DWithBiasAndRelu":
                    new_node.op = "_FusedQuantizedDepthwiseConv2D"
                    self.fused_ops = [b"BiasAdd", b"Relu", b"Requantize"]
                    self.output_types = [
                        requantize_node.attr["out_type"].type,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ]
                elif quantized_node_op == "_FusedQuantizedConv2D":
                    new_node.op = "_FusedQuantizedConv2D"
                elif quantized_node_op == "_FusedQuantizedDepthwiseConv2D":
                    new_node.op = "_FusedQuantizedDepthwiseConv2D"
                elif quantized_node_op == "_FusedQuantizedConv3D":
                    new_node.op = "_FusedQuantizedConv3D"
                elif quantized_node_op == "_FusedQuantizedDeconv2D":
                    new_node.op = "_FusedQuantizedDeconv2D"
                elif quantized_node_op == "_FusedQuantizedDeconv3D":
                    new_node.op = "_FusedQuantizedDeconv3D"
            else:
                new_node.op = quantized_node_op + "AndRequantize"
            new_node.name = requantize_node_name
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
            if "Tsummand" in quantized_node.attr:
                new_node.attr["Tsummand"].CopyFrom(quantized_node.attr["Tsummand"])
            if "data_format" in quantized_node.attr:
                new_node.attr["data_format"].CopyFrom(quantized_node.attr["data_format"])

            parent_node_name = Helper.node_name_from_input(quantized_node.input[0])
            if new_node.op in ("_FusedQuantizedDeconv2D", "_FusedQuantizedDeconv3D"):
                max_filter_node = self.graph_info[new_node.input[-3]].node
                min_filter_node = self.graph_info[new_node.input[-4]].node
            else:
                max_filter_node = self.graph_info[new_node.input[-1]].node
                min_filter_node = self.graph_info[new_node.input[-2]].node
            last_node = self.graph_info[new_node.input[0]].node
            new_node.input.append(requested_output_min_name)
            new_node.input.append(requested_output_max_name)

            if (
                last_node.op.find("Requantize") != -1
                or ((last_node.op.find("QuantizeV2") != -1 or last_node.op.find("QuantizedConv2D") != -1))
            ) and len(quantized_node.attr["fused_ops"].list.s) > 0:
                bias_node = self.graph_info[new_node.input[2]].node
                max_input_node = self.graph_info[last_node.input[-1]].node
                min_input_node = self.graph_info[last_node.input[-2]].node
                min_input = (min_input_node.attr["value"].tensor.float_val)[0]
                max_input = (max_input_node.attr["value"].tensor.float_val)[0]
                if "Depthwise" in quantized_node_op or requantize_node.op.find("PerChannel") != -1:
                    channel_size = max_filter_node.attr["value"].tensor.tensor_shape.dim[0].size
                    max_filter_tensor = tensor_util.MakeNdarray(min_filter_node.attr["value"].tensor)
                    min_filter_tensor = tensor_util.MakeNdarray(min_filter_node.attr["value"].tensor)
                else:
                    channel_size = 1
                    max_filter_tensor = []
                    min_filter_tensor = []
                    max_filter_tensor.append((max_filter_node.attr["value"].tensor.float_val)[0])
                    min_filter_tensor.append((min_filter_node.attr["value"].tensor.float_val)[0])
                bias_tensor = tensor_util.MakeNdarray(self.graph_info[new_node.input[2]].node.attr["value"].tensor)

                activation_range = 127.0 if new_node.attr["Tinput"].type == dtypes.qint8 else 255.0

                int32_bias = Helper.generate_int32_bias_for_conv(
                    bias_tensor,
                    channel_size,
                    max_input,
                    min_input,
                    max_filter_tensor,
                    min_filter_tensor,
                    activation_range,
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
            else:
                new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))
            # in tf 2.10, the "padding_list" attr name changes to explicit_paddings
            if "padding_list" in quantized_node.attr:
                if not self.new_api:
                    new_node.attr["padding_list"].CopyFrom(quantized_node.attr["padding_list"])
                elif quantized_node.attr["padding"].s == b"EXPLICIT":
                    new_node.attr["explicit_paddings"].CopyFrom(quantized_node.attr["padding_list"])
            elif "explicit_paddings" in quantized_node.attr:
                new_node.attr["explicit_paddings"].CopyFrom(quantized_node.attr["explicit_paddings"])
            if "dilations" in quantized_node.attr:
                new_node.attr["dilations"].CopyFrom(quantized_node.attr["dilations"])

            if self.new_api and new_node.op in (
                "_FusedQuantizedConv2D",
                "_FusedQuantizedDepthwiseConv2D",
                "_FusedQuantizedDeconv2D",
                "_FusedQuantizedDeconv3D",
            ):
                input_data_type = dtypes.qint8 if new_node.attr["Tinput"].type == dtypes.qint8 else dtypes.quint8
                if new_node.op in ("_FusedQuantizedDeconv2D", "_FusedQuantizedDeconv3D"):
                    Helper.set_attr_type_list(
                        new_node,
                        "Thost_inputs",
                        [
                            dtypes.int32.as_datatype_enum,
                            dtypes.qint8.as_datatype_enum,
                            input_data_type.as_datatype_enum,
                            (
                                dtypes.float32.as_datatype_enum
                                if new_node.attr["Tbias"].type == dtypes.float32
                                else dtypes.qint32.as_datatype_enum
                            ),
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                        ],
                    )
                else:
                    Helper.set_attr_type_list(
                        new_node,
                        "Thost_inputs",
                        [
                            input_data_type.as_datatype_enum,
                            dtypes.qint8.as_datatype_enum,
                            (
                                dtypes.float32.as_datatype_enum
                                if new_node.attr["Tbias"].type == dtypes.float32
                                else dtypes.qint32.as_datatype_enum
                            ),
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                        ],
                    )

                if quantized_node_op not in (
                    "_FusedQuantizedConv2D",
                    "_FusedQuantizedDepthwiseConv2D",
                    "_FusedQuantizedDeconv2D",
                    "_FusedQuantizedDeconv3D",
                ):
                    Helper.set_attr_type_list(new_node, "Thost_outputs", self.output_types)
                    new_node.attr["Tsummand"].CopyFrom(attr_value_pb2.AttrValue(type=self.output_types[0]))
                else:
                    if str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"_FusedHardSwish"]):
                        self.fused_ops = [b"BiasAdd", b"_FusedHardSwish", b"Requantize"]
                        Helper.set_attr_type_list(
                            new_node,
                            "Thost_outputs",
                            [
                                requantize_node.attr["out_type"].type,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                            ],
                        )
                        Helper.set_attr_dtype(
                            new_node, "out_type", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                        Helper.set_attr_dtype(
                            new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                    elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"_FusedSwish"]):
                        self.fused_ops = [b"BiasAdd", b"_FusedSwish", b"Requantize"]
                        Helper.set_attr_type_list(
                            new_node,
                            "Thost_outputs",
                            [
                                requantize_node.attr["out_type"].type,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                            ],
                        )
                        Helper.set_attr_dtype(
                            new_node, "out_type", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                        Helper.set_attr_dtype(
                            new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                    elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Relu"]):
                        self.fused_ops = [b"BiasAdd", b"Relu", b"Requantize"]
                        Helper.set_attr_type_list(
                            new_node,
                            "Thost_outputs",
                            [
                                requantize_node.attr["out_type"].type,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                            ],
                        )
                        Helper.set_attr_dtype(
                            new_node, "out_type", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                        Helper.set_attr_dtype(
                            new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                    elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"LeakyRelu"]):
                        self.fused_ops = [b"BiasAdd", b"LeakyRelu", b"Requantize"]
                        Helper.set_attr_type_list(
                            new_node,
                            "Thost_outputs",
                            [
                                requantize_node.attr["out_type"].type,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                            ],
                        )
                        Helper.set_attr_dtype(
                            new_node, "out_type", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                        Helper.set_attr_dtype(
                            new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                    elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Elu"]):
                        self.fused_ops = [b"BiasAdd", b"Elu", b"Requantize"]
                        Helper.set_attr_type_list(
                            new_node,
                            "Thost_outputs",
                            [
                                requantize_node.attr["out_type"].type,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                            ],
                        )
                        Helper.set_attr_dtype(
                            new_node, "out_type", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                        Helper.set_attr_dtype(
                            new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                    elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Sigmoid"]):
                        self.fused_ops = [b"BiasAdd", b"Sigmoid", b"Requantize"]
                        Helper.set_attr_type_list(
                            new_node,
                            "Thost_outputs",
                            [
                                requantize_node.attr["out_type"].type,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                            ],
                        )
                        Helper.set_attr_dtype(
                            new_node, "out_type", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                        Helper.set_attr_dtype(
                            new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                    elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd"]):
                        self.fused_ops = [b"BiasAdd", b"Requantize"]
                        Helper.set_attr_type_list(
                            new_node,
                            "Thost_outputs",
                            [
                                requantize_node.attr["out_type"].type,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                            ],
                        )
                        Helper.set_attr_dtype(
                            new_node, "out_type", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                        if new_node.op not in ("_FusedQuantizedDeconv2D", "_FusedQuantizedDeconv3D"):
                            Helper.set_attr_dtype(
                                new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type]
                            )
                    elif len(quantized_node.attr["fused_ops"].list.s) == 0:
                        if new_node.op in ("_FusedQuantizedDeconv2D", "_FusedQuantizedDeconv3D"):
                            Helper.set_attr_type_list(
                                new_node,
                                "Thost_inputs",
                                [
                                    dtypes.int32.as_datatype_enum,
                                    dtypes.qint8.as_datatype_enum,
                                    input_data_type.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                ],
                            )
                        else:
                            Helper.set_attr_type_list(
                                new_node,
                                "Thost_inputs",
                                [
                                    input_data_type.as_datatype_enum,
                                    dtypes.qint8.as_datatype_enum,
                                    # dtypes.float32.as_datatype_enum if new_node.attr["Tbias"].type == dtypes.float32 \
                                    # else dtypes.qint32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                    dtypes.float32.as_datatype_enum,
                                ],
                            )
                        self.fused_ops = [b"Requantize"]
                        Helper.set_attr_type_list(
                            new_node,
                            "Thost_outputs",
                            [
                                requantize_node.attr["out_type"].type,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                            ],
                        )
                        Helper.set_attr_dtype(
                            new_node, "out_type", dtype_map_dict[requantize_node.attr["out_type"].type]
                        )
                        if new_node.op not in ("_FusedQuantizedDeconv2D", "_FusedQuantizedDeconv3D"):
                            Helper.set_attr_dtype(
                                new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type]
                            )
                Helper.set_attr_string_list(new_node, "fused_ops", self.fused_ops)

            if "_kernel" in quantized_node.attr:
                new_node.attr["_kernel"].CopyFrom(quantized_node.attr["_kernel"])

            if new_node.op in ("_FusedQuantizedConv3D"):
                input_data_type = dtypes.qint8 if new_node.attr["Tinput"].type == dtypes.qint8 else dtypes.quint8
                if len(quantized_node.attr["fused_ops"].list.s) == 0:
                    Helper.set_attr_string_list(new_node, "fused_ops", [b"Requantize"])
                    Helper.set_attr_type_list(
                        new_node,
                        "Thost_inputs",
                        [
                            input_data_type.as_datatype_enum,
                            dtypes.qint8.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                        ],
                    )
                elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd"]):
                    Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"Requantize"])
                    Helper.set_attr_type_list(
                        new_node,
                        "Thost_inputs",
                        [
                            input_data_type.as_datatype_enum,
                            dtypes.qint8.as_datatype_enum,
                            (
                                dtypes.float32.as_datatype_enum
                                if new_node.attr["Tbias"].type == dtypes.float32
                                else dtypes.qint32.as_datatype_enum
                            ),
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                        ],
                    )
                elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Relu"]):
                    Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"Relu", b"Requantize"])
                    Helper.set_attr_type_list(
                        new_node,
                        "Thost_inputs",
                        [
                            input_data_type.as_datatype_enum,
                            dtypes.qint8.as_datatype_enum,
                            (
                                dtypes.float32.as_datatype_enum
                                if new_node.attr["Tbias"].type == dtypes.float32
                                else dtypes.qint32.as_datatype_enum
                            ),
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                        ],
                    )
                elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"LeakyRelu"]):
                    Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"LeakyRelu", b"Requantize"])
                    Helper.set_attr_type_list(
                        new_node,
                        "Thost_inputs",
                        [
                            input_data_type.as_datatype_enum,
                            dtypes.qint8.as_datatype_enum,
                            (
                                dtypes.float32.as_datatype_enum
                                if new_node.attr["Tbias"].type == dtypes.float32
                                else dtypes.qint32.as_datatype_enum
                            ),
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                        ],
                    )
                elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Elu"]):
                    Helper.set_attr_string_list(new_node, "fused_ops", [b"BiasAdd", b"Elu", b"Requantize"])
                    Helper.set_attr_type_list(
                        new_node,
                        "Thost_inputs",
                        [
                            input_data_type.as_datatype_enum,
                            dtypes.qint8.as_datatype_enum,
                            (
                                dtypes.float32.as_datatype_enum
                                if new_node.attr["Tbias"].type == dtypes.float32
                                else dtypes.qint32.as_datatype_enum
                            ),
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                        ],
                    )

                Helper.set_attr_type_list(
                    new_node,
                    "Thost_outputs",
                    [
                        requantize_node.attr["out_type"].type,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                Helper.set_attr_dtype(new_node, "out_type", dtype_map_dict[requantize_node.attr["out_type"].type])
                Helper.set_attr_dtype(new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type])

            if (
                quantized_node.op == "QuantizedConv2D"
                or quantized_node.op == "QuantizedConv2DWithBias"
                or quantized_node.op == "QuantizedDepthwiseConv2D"
                or quantized_node.op == "QuantizedDepthwiseConv2DWithBias"
                or ("alpha" in quantized_node.attr and quantized_node.attr["alpha"].f > 0)
            ):
                new_node.attr["out_type"].CopyFrom(attr_value_pb2.AttrValue(type=int8_type))
            elif (
                quantized_node.op == "QuantizedConv2DWithBiasAndRelu"
                or quantized_node.op == "QuantizedDepthwiseConv2DWithBiasAndRelu"
            ):
                new_node.attr["out_type"].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))
            elif new_node.op not in (
                "_FusedQuantizedConv2D",
                "_FusedQuantizedDepthwiseConv2D",
                "_FusedQuantizedConv3D",
                "_FusedQuantizedDeconv2D",
                "_FusedQuantizedDeconv3D",
            ):
                new_node.attr["out_type"].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))
            elif new_node.op in ("_FusedQuantizedDeconv2D", "_FusedQuantizedDeconv3D"):
                new_node.attr["out_type"].CopyFrom(attr_value_pb2.AttrValue(type=int8_type))
            old_input_name = dequantize_node_name if dequantize_node_name else requantize_node_name
            self.graph_analyzer.replace_single_node(
                new_node,
                [parent_node_name],
                quantized_node_name,
                self.graph_info[old_input_name].outputs,
                old_input_name,
            )
            self.graph_analyzer.remove_node(quantized_node_name)

        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(self.sum_pattern)
        for i in target_nodes:
            quantized_node_name = i[0]
            quantized_node = self.graph_info[quantized_node_name].node
            if i[-1][0] in ("_FusedQuantizedDepthwiseConv2D", "_FusedQuantizedConv2D", "_FusedQuantizedConv3D"):
                if quantized_node.attr["fused_ops"].list.s not in self.fuse_sum_op_types:
                    continue
                # else:
                #   print(quantized_node.attr['fused_ops'].list.s)

            requantize_node_name = i[1]
            requantize_node = self.graph_info[requantize_node_name].node
            requested_output_min_name = requantize_node.input[3]
            requested_output_max_name = requantize_node.input[4]

            quantized_node_op = i[-1][0]

            new_node = node_def_pb2.NodeDef()

            if self.new_api:
                if i[-1][0] in ("QuantizedConv2DWithBiasSumAndRelu",):
                    new_node.op = "_FusedQuantizedConv2D"
                    self.fused_ops = [b"BiasAdd", b"Sum", b"Relu", b"Requantize"]
                elif i[-1][0] == "_FusedQuantizedConv2D":
                    new_node.op = "_FusedQuantizedConv2D"
                elif quantized_node_op == "_FusedQuantizedDepthwiseConv2D":
                    new_node.op = "_FusedQuantizedDepthwiseConv2D"
                elif i[-1][0] == "_FusedQuantizedConv3D":
                    new_node.op = "_FusedQuantizedConv3D"
            else:
                new_node.op = quantized_node_op + "AndRequantize"

            new_node.name = requantize_node_name

            for _, value in enumerate(quantized_node.input[:-1]):
                new_node.input.append(value)

            new_node.attr["Tinput"].CopyFrom(quantized_node.attr["Tinput"])
            new_node.attr["Tfilter"].CopyFrom(quantized_node.attr["Tfilter"])
            new_node.attr["strides"].CopyFrom(quantized_node.attr["strides"])
            new_node.attr["padding"].CopyFrom(quantized_node.attr["padding"])
            # new_node.attr["Tsummand"].CopyFrom(quantized_node.attr['Tsummand'])

            new_node.input.append(requested_output_min_name)
            new_node.input.append(requested_output_max_name)
            deq_node = self.graph_info[Helper.node_name_from_input(quantized_node.input[-1])].node
            if deq_node.op != "Dequantize" or deq_node.op.find("Quantize") != -1:
                continue

            if deq_node.op == "Dequantize":
                original_summand_node = self.graph_info[Helper.node_name_from_input(deq_node.input[0])].node
            else:
                original_summand_node = deq_node
            summand_op_type = uint8_type if dtypes.as_dtype(deq_node.attr["T"].type) == uint8_type else int8_type

            for j in range(3):
                new_node.input.append(original_summand_node.name + ":{}".format(j))
            # in tf 2.10, the "padding_list" attr name changes to explicit_paddings
            if "padding_list" in quantized_node.attr:
                if not self.new_api:
                    new_node.attr["padding_list"].CopyFrom(quantized_node.attr["padding_list"])
                elif quantized_node.attr["padding"].s == b"EXPLICIT":
                    new_node.attr["explicit_paddings"].CopyFrom(quantized_node.attr["padding_list"])
            elif "explicit_paddings" in quantized_node.attr:
                new_node.attr["explicit_paddings"].CopyFrom(quantized_node.attr["explicit_paddings"])

            if "dilations" in quantized_node.attr:
                new_node.attr["dilations"].CopyFrom(quantized_node.attr["dilations"])

            if "alpha" in quantized_node.attr and quantized_node.attr["alpha"].f > 0:
                new_node.attr["out_type"].CopyFrom(attr_value_pb2.AttrValue(type=int8_type))
            else:
                new_node.attr["out_type"].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))

            new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))
            new_node.attr["Tsummand"].CopyFrom(attr_value_pb2.AttrValue(type=summand_op_type))

            if new_node.op in ("_FusedQuantizedConv2D", "_FusedQuantizedDepthwiseConv2D", "_FusedQuantizedConv3D"):
                original_input = list(new_node.input)
                new_input = []
                new_input.extend(original_input[:3])
                new_input.append(original_input[-3])
                new_input.extend(original_input[3:7])
                new_input.extend(original_input[-2:])
                new_input.extend(original_input[-5:-3])
                new_node.ClearField("input")
                new_node.input.extend(new_input)
                input_data_type = dtypes.qint8 if new_node.attr["Tinput"].type == dtypes.qint8 else dtypes.quint8
                Helper.set_attr_type_list(
                    new_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        (
                            dtypes.quint8.as_datatype_enum
                            if summand_op_type != int8_type
                            else dtypes.qint8.as_datatype_enum
                        ),
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                Helper.set_attr_dtype(
                    new_node, "Tsummand", dtypes.quint8 if summand_op_type != int8_type else dtypes.qint8
                )
                if str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Sum", b"Relu"]):
                    self.fused_ops = [b"BiasAdd", b"Sum", b"Relu", b"Requantize"]
                elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Sum", b"LeakyRelu"]):
                    self.fused_ops = [b"BiasAdd", b"Sum", b"LeakyRelu", b"Requantize"]
                elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"LeakyRelu", b"Sum"]):
                    self.fused_ops = [b"BiasAdd", b"LeakyRelu", b"Sum", b"Requantize"]
                elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Relu", b"Sum"]):
                    self.fused_ops = [b"BiasAdd", b"Relu", b"Sum", b"Requantize"]
                    # Current fusion requires summand has same dtype as output if output is qint8
                    Helper.set_attr_dtype(new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type])
                elif str(quantized_node.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Sum"]):
                    self.fused_ops = [b"BiasAdd", b"Sum", b"Requantize"]
                    # Current fusion requires summand has same dtype as output if output is qint8
                    Helper.set_attr_dtype(new_node, "Tsummand", dtype_map_dict[requantize_node.attr["out_type"].type])
                Helper.set_attr_type_list(
                    new_node,
                    "Thost_outputs",
                    [
                        requantize_node.attr["out_type"].type,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                Helper.set_attr_dtype(new_node, "out_type", dtype_map_dict[requantize_node.attr["out_type"].type])
                Helper.set_attr_string_list(new_node, "fused_ops", self.fused_ops)

            if not self.new_api:
                if quantized_node_op == "QuantizedConv2DWithBiasReluAndSum":
                    new_node.op = "QuantizedConv2DWithBiasReluAndSumAndRequantize"
                    if "alpha" in quantized_node.attr:
                        new_node.attr["alpha"].CopyFrom(quantized_node.attr["alpha"])

                elif summand_op_type == int8_type:
                    new_node.op = "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"

            self.graph_analyzer.replace_single_node(
                new_node,
                [quantized_node.input[0], original_summand_node.name],
                quantized_node.name,
                self.graph_info[requantize_node_name].outputs,
                requantize_node_name,
            )
            self.graph_analyzer.remove_node(quantized_node_name)

            if deq_node.op == "Dequantize":
                self.graph_analyzer.remove_node_with_single_input_output(deq_node.name)

        return self.graph_analyzer.dump_graph()
