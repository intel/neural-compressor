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

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.tensorflow.utils import version1_gt_version2, version1_lt_version2

from ..graph_base import GraphRewriterBase


class ConvertUniformQDQOptimizer(GraphRewriterBase):
    """Fuse newAPI Quantized MatMul Op with the successor Requantize Op."""

    def __init__(self, model, min_max_dict, device="cpu"):
        """Initialization."""
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model
        self.eps = 1e-05
        self.graph_info = self.graph_analyzer.parse_graph()

        self.uint8_type = dtypes.quint8.as_datatype_enum
        self.int8_type = dtypes.qint8.as_datatype_enum
        self.float32_type = dtypes.float32.as_datatype_enum
        self.qint32_type = dtypes.qint32.as_datatype_enum
        self.min_max_dict = min_max_dict
        self.quantization_min_val = None
        self.quantization_max_val = None

    def _calculate_zp_and_scale(self, min_value, max_value, dtype, quantize_pre_node_op):
        if isinstance(min_value, list):
            assert quantize_pre_node_op == Const, "Scales and zero-points for activations must always be a scalar"

        if dtype == attr_value_pb2.AttrValue(type=self.int8_type):
            zp = 0
            scale_range = 127
            self.quantization_min_val = -128
            self.quantization_max_val = 127
        elif dtype == attr_value_pb2.AttrValue(type=self.uint8_type):
            assert quantize_pre_node_op != "Const", "Zero-point must be always 0 for weights"
            zp = 128
            scale_range = 255
            self.quantization_min_val = 0
            self.quantization_max_val = 255
        else:
            raise ValueError("Unexpected data type for Quantize Op.")

        if isinstance(max_value, float):
            if dtype == attr_value_pb2.AttrValue(type=self.int8_type):
                scale_factor = max(abs(max_value), abs(min_value))/scale_range
            else : # uint8
                scale_factor = (max_value - min_value) / scale_range
            return (zp, scale_factor) if scale_range == 127 else (-round(min_value/scale_factor), scale_factor)

        scales = []
        zero_points = []
        for i in range(len(max_value)):
            scale_factor = max(abs(max_value[i]), abs(min_value[i]))/scale_range
            scales.append(scale_factor)
            zero_points.append(zp)

        return zero_points, scales

    def do_transformation(self):
        """Fuse the quantized op with the following requantize op.

        Returns:
            [graphdef]: the optimized graphdef object
        """
        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(
            [["QuantizeV2"], ["Dequantize"]]
        )
        for i in target_nodes:
            shared_quantize_node = False
            quantize_node_name = i[0]
            dequantize_node_name = i[1]
            dequantize_node = self.graph_info[dequantize_node_name].node
            dequantize_down_node = self.graph_info[self.graph_info[dequantize_node_name].outputs[0]].node

            quantize_node = self.graph_info[quantize_node_name].node
            quantize_pre_node_op = self.graph_info[quantize_node.input[0]].node.op
            quantize_min_name = quantize_node.input[1]
            quantize_max_name = quantize_node.input[2]

            dtype = quantize_node.attr["T"]
            try:
                min_value = self.graph_info[quantize_min_name].node.attr["value"].tensor.float_val[0]
                max_value = self.graph_info[quantize_max_name].node.attr["value"].tensor.float_val[0]
            except:
                min_value = self.min_max_dict[quantize_min_name]
                max_value = self.min_max_dict[quantize_max_name]
            zero_point_value, scale_value = self._calculate_zp_and_scale(min_value, max_value, dtype, quantize_pre_node_op)
            zero_point_name = quantize_min_name[:-4] + "zero_point"
            scale_name = quantize_min_name[:-4] + "scale"
            print("zero_point_value:", zero_point_value)
            zero_point_node = Helper.create_constant_node(zero_point_name, zero_point_value, dtypes.int32, device="cpu")
            scale_node = Helper.create_constant_node(scale_name, scale_value, dtypes.float32, device="cpu")

            uniform_quantize_node = node_def_pb2.NodeDef()
            uniform_quantize_node.op = "UniformQuantize"
            uniform_quantize_node.name = quantize_node_name+"_UniformQuantize"
            uniform_quantize_node.input.extend([quantize_node.input[0], scale_name, zero_point_name])
            Helper.set_attr_int(uniform_quantize_node, "quantization_min_val", self.quantization_min_val)
            Helper.set_attr_int(uniform_quantize_node, "quantization_max_val", self.quantization_max_val)
            Helper.set_attr_dtype(uniform_quantize_node, "Tin", dtypes.float32)

            # per-channel weights
            if isinstance(zero_point_value, list):
                # const_weight->q->dq->conv2d
                if dequantize_down_node.op == "Conv2D":
                    Helper.set_attr_int(uniform_quantize_node, "quantization_axis", 3)
                elif dequantize_down_node.op == "DepthwiseConv2dNative":
                    Helper.set_attr_int(uniform_quantize_node, "quantization_axis", 2)
                # const_weight->q->dq->matmul
                elif dequantize_down_node.op == "MatMul":
                    if str(dequantize_down_node.attr["transpose_b"])=='b: true\n':
                        Helper.set_attr_int(uniform_quantize_node, "quantization_axis", 0)
                    else:
                        Helper.set_attr_int(uniform_quantize_node, "quantization_axis", 1)
                else:
                    raise ValueError("Unsupported op type for per-channel quantization.")
            # per-tensor weights and activations
            else:
                Helper.set_attr_int(uniform_quantize_node, "quantization_axis", -1)

            uniform_quantize_node.attr["Tout"].CopyFrom(quantize_node.attr["T"])
            uniform_dequantize_node = node_def_pb2.NodeDef()
            uniform_dequantize_node.op = "UniformDequantize"
            uniform_dequantize_node.name = dequantize_node_name+"_UniformDequantize"
            uniform_dequantize_node.input.extend([uniform_quantize_node.name,
                                            scale_name,
                                            zero_point_name,
                                            ])
            Helper.set_attr_int(uniform_dequantize_node, "quantization_min_val", self.quantization_min_val)
            Helper.set_attr_int(uniform_dequantize_node, "quantization_max_val", self.quantization_max_val)
            Helper.set_attr_dtype(uniform_dequantize_node, "Tout", dtypes.float32)

            if "quantization_axis" in uniform_quantize_node.attr:
                uniform_dequantize_node.attr["quantization_axis"].CopyFrom(uniform_quantize_node.attr["quantization_axis"])
            if "Tin" in uniform_quantize_node.attr:
                uniform_dequantize_node.attr["Tin"].CopyFrom(uniform_quantize_node.attr["Tout"])

            parent_node_name = Helper.node_name_from_input(quantize_node.input[0])

            self.graph_analyzer.add_node(zero_point_node, None, [uniform_quantize_node.name])
            self.graph_analyzer.add_node(scale_node, None, [uniform_quantize_node.name])

            quantize_output_node_name = set()
            for node_name in self.graph_info[quantize_node_name].outputs:
                quantize_output_node_name.add(node_name)
            self.graph_analyzer.replace_single_node(
                uniform_quantize_node,
                [parent_node_name],
                quantize_node_name,
                [i for i in quantize_output_node_name],
                quantize_node_name,
            )

            dequantize_output_node_name = set()
            for node_name in self.graph_info[dequantize_node_name].outputs:
                dequantize_output_node_name.add(node_name)
            self.graph_analyzer.replace_single_node(
                uniform_dequantize_node,
                [uniform_quantize_node.name],
                dequantize_node_name,
                [i for i in dequantize_output_node_name],
                dequantize_node_name,
            )

            self.graph_analyzer.remove_node(quantize_node_name)
            self.graph_analyzer.remove_node(dequantize_node_name)


        return self.graph_analyzer.dump_graph()
