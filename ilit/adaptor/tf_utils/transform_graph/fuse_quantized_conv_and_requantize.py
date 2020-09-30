#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from ..quantize_graph.quantize_graph_common import QuantizeGraphHelper as helper


def get_fuse_index(input_node_map, input_name_list, device):
    conv_op_list = ("QuantizedConv2DWithBiasAndRelu",
                    "QuantizedDepthwiseConv2DWithBiasAndRelu",
                    "QuantizedConv2DWithBias",
                    "QuantizedConv2DWithBiasSumAndRelu")
    fuse_op_list = []
    fuse_op_with_sum_list = []
    fuse_op_with_sum_deq_list = []
    const_node_type = "HostConst" if device == "gpu" else "Const"
    for node_index, node_name in enumerate(input_name_list):
        node = input_node_map[node_name]
        node_op = node.op

        if node_op in conv_op_list and \
                input_node_map[input_name_list[node_index + 1]].op == const_node_type and \
                input_node_map[input_name_list[node_index + 2]].op == const_node_type and \
                (input_node_map[input_name_list[node_index + 3]].op == "Requantize" or
                 input_node_map[input_name_list[node_index + 3]].op == "RequantizePerChannel"):
            fuse_op_list.append(node_index)
            has_dequantize = False
            for node_input in node.input:
                if node_input in input_node_map and input_node_map[
                        node_input].op == "Dequantize":
                    has_dequantize = True
                    break

            if has_dequantize:
                fuse_op_with_sum_deq_list.append(node_index)
            else:
                fuse_op_with_sum_list.append(node_index)

    return fuse_op_with_sum_list, fuse_op_with_sum_deq_list


def generate_output_graph(
        input_graph_def,
        input_node_map,
        output_node_map,
        fuse_op_list,
        fuse_op_deq_list,
        device):
    output_graph_def = graph_pb2.GraphDef()
    skip_list = []
    skip_node_name = []
    int8_type = dtypes.qint8.as_datatype_enum
    uint8_type = dtypes.quint8.as_datatype_enum
    float32_type = dtypes.float32.as_datatype_enum
    qint32_type = dtypes.qint32.as_datatype_enum
    for index, node in enumerate(input_graph_def.node):
        if index in fuse_op_list:
            const_node_1 = input_graph_def.node[index + 1]
            const_node_2 = input_graph_def.node[index + 2]
            requantize_node = input_graph_def.node[index + 3]
            new_node = node_def_pb2.NodeDef()

            new_node.op = node.op + "AndRequantize"
            new_node.name = requantize_node.name
            for _, value in enumerate(node.input):
                new_node.input.append(value)

            new_node.input.append(const_node_1.name)
            new_node.input.append(const_node_2.name)

            new_node.attr["Tinput"].CopyFrom(node.attr['Tinput'])
            new_node.attr["Tfilter"].CopyFrom(node.attr['Tfilter'])
            new_node.attr["strides"].CopyFrom(node.attr['strides'])
            new_node.attr["padding"].CopyFrom(node.attr['padding'])
            if input_node_map[new_node.input[0]].op.find("Requantize") != -1:
                bias_node = input_node_map[new_node.input[2]]
                last_node = input_node_map[new_node.input[0]]
                max_input_node = (input_node_map[last_node.input[4][:-2]])
                min_input_node = (input_node_map[last_node.input[3][:-2]])
                max_filter = input_node_map[new_node.input[6]]
                min_filter = input_node_map[new_node.input[5]]

                min_input = (min_input_node.attr['value'].tensor.float_val)[0]
                max_input = (max_input_node.attr['value'].tensor.float_val)[0]
                if 'Depthwise' in node.op or "RequantizePerChannel" in [
                        node.op for node in output_node_map[node.name]
                ]:

                    channel_size = max_filter.attr[
                        'value'].tensor.tensor_shape.dim[0].size
                    max_filter_tensor = tensor_util.MakeNdarray(
                        max_filter.attr['value'].tensor)
                    min_filter_tensor = tensor_util.MakeNdarray(
                        min_filter.attr['value'].tensor)
                else:

                    channel_size = 1
                    max_filter_tensor = []
                    min_filter_tensor = []
                    max_filter_tensor.append(
                        (max_filter.attr['value'].tensor.float_val)[0])
                    min_filter_tensor.append(
                        (min_filter.attr['value'].tensor.float_val)[0])

                bias_tensor = tensor_util.MakeNdarray(
                    input_node_map[new_node.input[2]].attr['value'].tensor)
                bias_length = bias_tensor.shape[0]
                scales = []
                for i in range(channel_size):
                    scales.append(255.0 * 127.0 /
                                  (max(abs(max_input), abs(min_input)) *
                                   max(abs(max_filter_tensor[i]),
                                       abs(min_filter_tensor[i]))))

                int32_bias = []
                if channel_size > 1:
                    for i in range(bias_length):
                        int32_bias.append((int)(bias_tensor[i] * scales[i]))
                else:
                    for i in range(bias_length):
                        int32_bias.append((int)(bias_tensor[i] * scales[0]))
                #(TODO) GPU not support qint32 bias tensor
                # float32 type should be removed after GPU support qint32 bias
                bias_node.attr['dtype'].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type \
                                             if device =='gpu' else qint32_type))
                bias_node.attr['value'].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            bias_tensor if device == 'gpu' else int32_bias ,
                            dtypes.float32 if device == 'gpu' else dtypes.int32,
                            bias_tensor.shape)))

                bias_node.attr['value'].tensor.dtype = float32_type \
                                        if device == 'gpu' else qint32_type
                skip_node_name.append(bias_node.name)
                output_graph_def.node.extend([bias_node])
                new_node.attr["Tbias"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type \
                                             if device == 'gpu' else qint32_type))

            else:
                new_node.attr["Tbias"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type))

            if "padding_list" in node.attr:
                new_node.attr["padding_list"].CopyFrom(
                    node.attr['padding_list'])
            if "dilations" in node.attr:
                new_node.attr["dilations"].CopyFrom(node.attr['dilations'])

            if node.op == "QuantizedConv2D" or node.op == "QuantizedConv2DWithBias":
                new_node.attr["out_type"].CopyFrom(
                    attr_value_pb2.AttrValue(type=int8_type))
            else:
                new_node.attr["out_type"].CopyFrom(
                    attr_value_pb2.AttrValue(type=uint8_type))

            skip_list.append(index + 1)
            skip_list.append(index + 2)
            skip_list.append(index + 3)
            output_graph_def.node.extend(
                [new_node, const_node_1, const_node_2])
        elif index in skip_list or node.name in skip_node_name:
            continue
        elif node.op == "Dequantize":
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(node)
            new_node.attr["mode"].s = b"SCALED"
            p_node = input_node_map[new_node.input[0]]
            pp_node = input_node_map[p_node.name].input[0]
            if input_node_map[pp_node].op.find("Relu") != -1 or p_node.op in (
                    "QuantizedAvgPool", "QuantizedMaxPool",
                    "QuantizedConcatV2"):
                new_node.attr["T"].CopyFrom(
                    attr_value_pb2.AttrValue(type=uint8_type))
            elif input_node_map[pp_node].op.find("QuantizedMatMulWithBias"
                                                 ) != -1 and p_node.op.find("Requantize") != -1:
                new_node.attr["mode"].s = node.attr["mode"].s
                new_node.attr["T"].CopyFrom(
                    attr_value_pb2.AttrValue(type=node.attr["T"].type))
            else:
                new_node.attr["T"].CopyFrom(
                    attr_value_pb2.AttrValue(type=int8_type))
            output_graph_def.node.extend([new_node])
        elif index in fuse_op_deq_list:
            original_summand_node = input_node_map[
                input_graph_def.node[index].input[-1]]
            sum_const_node_1 = input_graph_def.node[index + 1]
            sum_const_node_2 = input_graph_def.node[index + 2]
            sum_requantize_node = input_graph_def.node[index + 3]

            new_node = node_def_pb2.NodeDef()

            new_node.op = node.op + "AndRequantize"
            new_node.name = sum_requantize_node.name
            for _, value in enumerate(node.input[:-1]):
                new_node.input.append(value)
            new_node.input.append(sum_const_node_1.name)
            new_node.input.append(sum_const_node_2.name)
            new_node.input.append(
                input_node_map[original_summand_node.name].input[0])
            new_node.input.append(
                input_node_map[original_summand_node.name].input[0] + ":1")
            new_node.input.append(
                input_node_map[original_summand_node.name].input[0] + ":2")

            # skip_list.append(index + 1)
            # skip_list.append(index + 2)
            skip_list.append(index + 3)

            new_node.attr["Tinput"].CopyFrom(node.attr['Tinput'])
            new_node.attr["Tfilter"].CopyFrom(node.attr['Tfilter'])
            new_node.attr["strides"].CopyFrom(node.attr['strides'])
            new_node.attr["padding"].CopyFrom(node.attr['padding'])
            if input_node_map[new_node.input[0]].op.find("Requantize") != -1:

                bias_node = input_node_map[new_node.input[2]]
                last_node = input_node_map[new_node.input[0]]
                max_input_node = (input_node_map[last_node.input[4][:-2]])
                min_input_node = (input_node_map[last_node.input[3][:-2]])
                max_filter = input_node_map[new_node.input[6]]
                min_filter = input_node_map[new_node.input[5]]

                min_input = (min_input_node.attr['value'].tensor.float_val)[0]
                max_input = (max_input_node.attr['value'].tensor.float_val)[0]

                if "RequantizePerChannel" in [
                        node.op for node in output_node_map[node.name]
                ]:
                    channel_size = max_filter.attr[
                        'value'].tensor.tensor_shape.dim[0].size
                    max_filter_tensor = tensor_util.MakeNdarray(
                        max_filter.attr['value'].tensor)
                    min_filter_tensor = tensor_util.MakeNdarray(
                        min_filter.attr['value'].tensor)
                else:
                    channel_size = 1
                    max_filter_tensor = []
                    min_filter_tensor = []
                    max_filter_tensor.append(
                        (max_filter.attr['value'].tensor.float_val)[0])
                    min_filter_tensor.append(
                        (min_filter.attr['value'].tensor.float_val)[0])

                bias_tensor = (tensor_util.MakeNdarray(
                    input_node_map[new_node.input[2]].attr['value'].tensor))
                bias_length = bias_tensor.shape[0]
                scales = []
                for i in range(channel_size):
                    scales.append(255.0 * 127.0 /
                                  (max(abs(max_input), abs(min_input)) *
                                   max(abs(max_filter_tensor[i]),
                                       abs(min_filter_tensor[i]))))

                int32_bias = []
                if channel_size > 1:
                    for i in range(bias_length):
                        int32_bias.append(int(bias_tensor[i] * scales[i]))
                else:
                    for i in range(bias_length):
                        int32_bias.append(int(bias_tensor[i] * scales[0]))

                #(TODO) GPU not support qint32 bias tensor
                # float32 type should be removed after GPU support qint32 bias
                bias_node.attr['dtype'].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type \
                                             if device =='gpu' else qint32_type))
                bias_node.attr['value'].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            bias_tensor if device == 'gpu' else int32_bias ,
                            dtypes.float32 if device == 'gpu' else dtypes.int32,
                            bias_tensor.shape)))

                bias_node.attr['value'].tensor.dtype = float32_type \
                                        if device == 'gpu' else qint32_type
                new_node.attr["Tbias"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type \
                                             if device == 'gpu' else qint32_type))

                skip_node_name.append(bias_node.name)
                output_graph_def.node.extend([bias_node])

            else:
                new_node.attr["Tbias"].CopyFrom(
                    attr_value_pb2.AttrValue(type=float32_type))

            if "padding_list" in node.attr:
                new_node.attr["padding_list"].CopyFrom(
                    node.attr['padding_list'])
            if "dilations" in node.attr:
                new_node.attr["dilations"].CopyFrom(node.attr['dilations'])

            new_node.attr["out_type"].CopyFrom(
                attr_value_pb2.AttrValue(type=uint8_type))

            summand_op_type = uint8_type if dtypes.as_dtype(
                original_summand_node.attr["T"].type
            ) == uint8_type else int8_type

            if summand_op_type == int8_type:
                new_node.op = "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"

            new_node.attr["Tsummand"].CopyFrom(
                attr_value_pb2.AttrValue(type=summand_op_type))
            output_graph_def.node.extend([new_node])
        else:
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(node)
            output_graph_def.node.extend([new_node])
    return output_graph_def


def parse_input_graph(input_graph_def):
    node_name_list = []
    input_node_map = {}
    output_node_map = {}
    for node in input_graph_def.node:
        node_name_list.append(node.name)
        if node.input:
            for _, sub_input in enumerate(node.input):
                input_node_name = helper.node_name_from_input(sub_input)
                if input_node_name not in output_node_map:
                    output_node_map[input_node_name] = []

                if node not in output_node_map[input_node_name]:
                    output_node_map[input_node_name].append(node)

        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            logging.getLogger().info('Duplicate node name {}'.format(node.name))

    return input_node_map, output_node_map, node_name_list


def fuse_quantized_conv_and_requantize(input_graph, device):
    input_node_map, output_node_map, node_name_list = parse_input_graph(
        input_graph)
    fuse_op_list, fuse_op_deq_list = get_fuse_index(input_node_map,
                                                    node_name_list, device)
    return generate_output_graph(
        input_graph,
        input_node_map,
        output_node_map,
        fuse_op_list,
        fuse_op_deq_list,
        device)
