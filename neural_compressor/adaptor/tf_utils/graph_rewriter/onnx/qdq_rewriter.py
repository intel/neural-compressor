#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2022 Intel Corporation
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

import numpy as np
from onnx import TensorProto, helper
from . import tf2onnx_utils as utils

# pylint: disable=missing-docstring

def extract_numpy_array(node):
    return np.frombuffer(node.attr["value"].t.raw_data, dtype="float32")

def convert_qdq_nodes(onnx_graph, match_results):

    for match in match_results:
        qdq_node = match.get_op('output')
        qdq_node_output_dtype = onnx_graph.get_dtype(qdq_node.output[0])
        qdq_node_output_shape = onnx_graph.get_shape(qdq_node.output[0])

        # Get the attributes of qdq node
        narrow_range = qdq_node.attr['narrow_range'].i
        signed_input = qdq_node.attr['signed_input'].i
        range_given = qdq_node.get_attr_value("range_given", qdq_node.type != "QuantizeAndDequantizeV2" or \
                                                             qdq_node.type != "QuantizeAndDequantizeV4")

        min_quantized, max_quantized = [-127, 127]
        if not narrow_range and signed_input:
            min_quantized = -128

        if not signed_input:
            min_quantized, max_quantized = [0, 255]

        # Get axis attribute for per channel implementation.
        axis = qdq_node.get_attr_value('axis', -1)
        q_attrs = {}

        quantized_np_dtype = np.int8 if signed_input else np.uint8
        quantized_dtype = TensorProto.INT8 if signed_input else TensorProto.UINT8

        if axis != -1:
            utils.assert_error(onnx_graph.opset >= 13, "Opset >= 13 is required for per channel quantization")
            q_attrs['axis'] = axis

        if not range_given:
            min_np = np.array(min_quantized, np.float32)
            max_np = np.array(max_quantized, np.float32)
            max_quantized_const = onnx_graph.make_const(utils.set_name("max_const"), max_np).output[0]
            if signed_input:
                min_quantized_const = onnx_graph.make_const(utils.set_name("min_const"), min_np).output[0]
            reduce_attr = {'keepdims': 0}
            if axis != -1:
                inp_rank = onnx_graph.get_rank(qdq_node.input[0])
                utils.assert_error(inp_rank is not None, "Input rank cannot be unknown for qdq op %s", qdq_node.name)
                reduce_axes = [i for i in range(inp_rank) if i != axis]
                reduce_attr['axes'] = reduce_axes

            max_value = onnx_graph.make_node("ReduceMax", [qdq_node.input[0]], attr=reduce_attr).output[0]
            if signed_input:
                min_value = onnx_graph.make_node("ReduceMin", [qdq_node.input[0]], attr=reduce_attr).output[0]

            scale_from_max_side = onnx_graph.make_node("Div", [max_value, max_quantized_const]).output[0]
            if signed_input:
                scale_from_min_side = onnx_graph.make_node("Div", [min_value, min_quantized_const]).output[0]
                scale = onnx_graph.make_node("Max", [scale_from_min_side, scale_from_max_side]).output[0]
            else:
                scale = scale_from_max_side

            if axis == -1:
                zero_point_np = np.zeros([], dtype=quantized_np_dtype)
                zero_point = onnx_graph.make_const(utils.set_name("zero_point"), zero_point_np).output[0]
            else:
                zero_tensor = helper.make_tensor("value", quantized_dtype, dims=[1], vals=[0])
                scale_shape = onnx_graph.make_node("Shape", [scale]).output[0]
                zero_point = onnx_graph.make_node("ConstantOfShape", inputs=[scale_shape], attr={"value": zero_tensor}).output[0]
        else:
            # Get the min and max value of the inputs to QDQ op
            min_value = extract_numpy_array(qdq_node.inputs[1])
            max_value = extract_numpy_array(qdq_node.inputs[2])

            num_channels = min_value.shape[0]
            scales = np.zeros(num_channels, dtype=np.float32)

            for i in range(num_channels):
                # Calculate scales from the min and max values
                scale_from_min_side = min_value[i] / min_quantized if min_quantized < 0 else 0
                scale_from_max_side = max_value[i] / max_quantized if max_quantized > 0 else 0

                if scale_from_min_side > scale_from_max_side:
                    scale = scale_from_min_side
                else:
                    scale = scale_from_max_side

                utils.assert_error(scale > 0, "Quantize/Dequantize scale must be greater than zero")
                scales[i] = np.float32(scale)

            # Set scalars for scale and zero point for per layer quantization
            if num_channels == 1:
                scales = scales[0]
                zero_point_np = np.zeros([], dtype=quantized_np_dtype)
            else:
                utils.assert_error(axis != -1, "Axis must be specified for per channel quantization")
                zero_point_np = np.zeros([num_channels], dtype=quantized_np_dtype)

            # Split it into QuantizeLinear and DequantizeLinear and remove the QDQ node reference
            cast_scale = scales.astype(np.float32)
            scale = onnx_graph.make_const(name=utils.set_name("quant_scale"), np_val=cast_scale).output[0]
            zero_point = onnx_graph.make_const(utils.set_name("zero_point"), zero_point_np).output[0]

        quant_node = onnx_graph.make_node(op_type="QuantizeLinear",
                                 inputs=[qdq_node.input[0], scale, zero_point],
                                 shapes=[qdq_node_output_shape],
                                 attr=q_attrs,
                                 dtypes=[quantized_dtype],
                                 name=utils.set_name("QuantLinearNode"))

        onnx_graph.set_shape(quant_node.output[0], qdq_node_output_shape)

        onnx_graph.remove_node(qdq_node.name)

        dequant_node = onnx_graph.make_node(op_type="DequantizeLinear",
                                   inputs=[quant_node.output[0], scale, zero_point],
                                   outputs=[qdq_node.output[0]],
                                   shapes=[qdq_node_output_shape],
                                   attr=q_attrs,
                                   dtypes=[qdq_node_output_dtype],
                                   name=utils.set_name("DequantLinearNode"))
        onnx_graph.set_shape(dequant_node.output[0], qdq_node_output_shape)

    return onnx_graph.get_nodes()

def rewrite_quantize_and_dequantize(g, ops):

    pattern_for_qdq_v2 = \
        OpTypePattern('QuantizeAndDequantizeV2', name='output', inputs=[
            OpTypePattern("*"),
            OpTypePattern(None),
            OpTypePattern(None),
        ])
    pattern_for_qdq_v3 = \
        OpTypePattern('QuantizeAndDequantizeV3', name='output', inputs=[
            OpTypePattern("*"),
            OpTypePattern(None),
            OpTypePattern(None),
            OpTypePattern(None),
        ])
    pattern_for_qdq_v4 = \
        OpTypePattern('QuantizeAndDequantizeV4', name='output', inputs=[
            OpTypePattern("*"),
            OpTypePattern(None),
            OpTypePattern(None),
        ])

    # Match all the patterns for QDQ ops
    patterns = [pattern_for_qdq_v2, pattern_for_qdq_v3, pattern_for_qdq_v4]
    match_results = []
    for pattern in patterns:
        matcher = GraphMatcher(pattern)
        results = list(matcher.match_ops(ops))
        match_results.extend(results)

    return create_qdq_nodes(g, match_results)