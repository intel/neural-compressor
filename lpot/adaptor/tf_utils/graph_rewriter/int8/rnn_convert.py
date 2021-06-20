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


from lpot.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops

import numpy as np
import tensorflow as tf


class QuantizedRNNConverter(GraphRewriterBase):
    def __init__(self, model, calibration_data, rnn_details):
        super().__init__(model)
        self.calibration_data = calibration_data
        self.rnn_details = rnn_details

    @dump_elapsed_time("Pass QuantizedRNNConverter")
    def do_transformation(self):
        g = GraphAnalyzer()
        g.graph = self.model
        graph_info = g.parse_graph()

        for i in self.rnn_details.keys():         # pragma: no cover
            start_node_name = graph_info[i[0]].node.input[0]
            min_str = i[0] + '_eightbit_min_' + \
                start_node_name + '__print__;__min:'
            input_min_values = []
            input_max_values = []
            output_min_values = []
            output_max_values = []
            max_str = i[0] + '_eightbit_max_' + \
                start_node_name + '__print__;__max:'
            output_str = i[0] + \
                '_eightbit_requant_range__print__;__requant_min_max:'
            for j in self.calibration_data:
                if j.find(min_str) != -1:
                    input_min_values.append(
                        float(j.split('[')[-1].split(']')[0]))
                if j.find(max_str) != -1:
                    input_max_values.append(
                        float(j.split('[')[-1].split(']')[0]))

                if j.find(output_str) != -1:
                    output_min_values.append(
                        float(j.split(':')[-1][1:].split(']')[0]))
                    output_max_values.append(float(j.split('][')[-1][:-1]))
            min_input = min(input_min_values)
            max_input = max(input_max_values)
            min_output = min(output_min_values)
            max_output = max(output_max_values)
            q_max_in_node = Helper.create_constant_node(
                i[0] + '_quant_max', max_input, dtypes.float32)

            q_min_in_node = Helper.create_constant_node(
                i[0] + '_quant_min', min_input, dtypes.float32)
            q_enter_min_node = Helper.create_node(
                'Enter', q_min_in_node.name+'_enter', [q_min_in_node.name])
            Helper.set_attr_string(
                q_enter_min_node, 'frame_name', self.rnn_details[i].encode())
            Helper.set_attr_dtype(q_enter_min_node, 'T', dtypes.float32)
            Helper.set_attr_bool(q_enter_min_node, 'is_constant', True)
            Helper.set_attr_int(q_enter_min_node, 'parallel_iterations', 32)
            q_enter_max_node = Helper.create_node(
                'Enter', q_max_in_node.name+'_enter', [q_max_in_node.name])
            Helper.set_attr_dtype(q_enter_max_node, 'T', dtypes.float32)
            Helper.set_attr_string(
                q_enter_max_node, 'frame_name', self.rnn_details[i].encode())
            Helper.set_attr_bool(q_enter_max_node, 'is_constant', True)
            Helper.set_attr_int(q_enter_max_node, 'parallel_iterations', 32)

            split_node_name = graph_info[i[0]].node.input[1]
            enter_node_name = graph_info[Helper.node_name_from_input(
                split_node_name)].node.input[1]
            weight_node_name = graph_info[Helper.node_name_from_input(
                enter_node_name)].node.input[0]
            weight_node = graph_info[Helper.node_name_from_input(
                weight_node_name)].node
            if weight_node.attr['dtype'].type == dtypes.qint8:
                qint8_const_name = weight_node_name
            else:
                base_name = weight_node_name + "_"
                qint8_const_name = base_name + "qint8_const"
                min_name = base_name + "min"
                max_name = base_name + "max"

            need_to_create_const_node = bool(
                qint8_const_name not in graph_info)
            if need_to_create_const_node:
                float_tensor = tensor_util.MakeNdarray(
                    weight_node.attr["value"].tensor)

                min_value = np.min(float_tensor.flatten())
                max_value = np.max(float_tensor.flatten())
                # Same processing of min-max as in quantize_weight_eightbit
                # function.
                if min_value > 0.0:
                    min_value = 0.0
                if min_value == max_value:
                    if abs(min_value) < 0.000001:
                        max_value = min_value + 1.0
                    elif min_value > 0:
                        max_value = 2 * min_value
                    else:
                        max_value = min_value / 2.0

                sess = tf.compat.v1.Session()
                with sess.as_default():
                    quantize_op = array_ops.quantize_v2(
                        float_tensor,
                        min_value,
                        max_value,
                        dtypes.qint8,
                        mode='SCALED',
                        round_mode="HALF_TO_EVEN")
                    qint8_tensor = quantize_op[0].numpy(
                    ) if tf.executing_eagerly() else quantize_op[0].eval()
                    # Updated min-max values should be passed to the next
                    # feeding node.
                    min_value = quantize_op[1].numpy(
                    ) if tf.executing_eagerly() else quantize_op[1].eval()
                    max_value = quantize_op[2].numpy(
                    ) if tf.executing_eagerly() else quantize_op[2].eval()
                sess.close()

                shape = tensor_util.TensorShapeProtoToList(
                    weight_node.attr["value"].tensor.tensor_shape)
                qint8_const_node = Helper.create_constant_node(qint8_const_name,
                                                               qint8_tensor,
                                                               dtypes.qint8,
                                                               shape=shape)

                min_node = Helper.create_constant_node(
                    min_name, min_value, dtypes.float32)

                max_node = Helper.create_constant_node(
                    max_name, max_value, dtypes.float32)
                enter_min_node = Helper.create_node(
                    'Enter', min_name+'_enter', [min_name])
                Helper.set_attr_string(
                    enter_min_node, 'frame_name', self.rnn_details[i].encode())
                Helper.set_attr_dtype(enter_min_node, 'T', dtypes.float32)
                Helper.set_attr_bool(enter_min_node, 'is_constant', True)
                Helper.set_attr_int(enter_min_node, 'parallel_iterations', 32)
                enter_max_node = Helper.create_node(
                    'Enter', max_name+'_enter', [max_name])
                Helper.set_attr_dtype(enter_max_node, 'T', dtypes.float32)
                Helper.set_attr_string(
                    enter_max_node, 'frame_name', self.rnn_details[i].encode())
                Helper.set_attr_bool(enter_max_node, 'is_constant', True)
                Helper.set_attr_int(enter_max_node, 'parallel_iterations', 32)
            else:
                qint8_const_node = graph_info[qint8_const_name].node
                min_node = graph_info[min_name].node
                max_node = graph_info[max_name].node
            quant_input = [start_node_name,
                           q_enter_min_node.name, q_enter_max_node.name]
            quantize_node = Helper.create_node(
                'QuantizeV2', i[0] + '_quantize', quant_input)
            Helper.set_attr_dtype(quantize_node, "T", dtypes.quint8)
            Helper.set_attr_string(quantize_node, "mode", b"MIN_FIRST")
            g.add_node(quantize_node, start_node_name, [i[0]])
            g.add_node(q_enter_max_node, None, [quantize_node.name])
            g.add_node(q_enter_min_node, None, [quantize_node.name])
            g.add_node(q_max_in_node, None, [q_enter_max_node.name])
            g.add_node(q_min_in_node, None, [q_enter_min_node.name])

            bias_node = graph_info[graph_info[i[0]].outputs[0]].node
            if graph_info[bias_node.name].outputs:
                last_node_name = [
                    graph_info[graph_info[bias_node.name].outputs[0]].node.name]
            else:
                last_node_name = []
            quantized_matmul_input = [quantize_node.name,
                                      Helper.node_name_from_input(
                                          graph_info[i[0]].node.input[1]),
                                      bias_node.input[1]]
            quantized_matmul_input.append(quantize_node.name + ':1')
            quantized_matmul_input.append(quantize_node.name + ':2')

            quantized_matmul_input.append(enter_min_node.name)
            quantized_matmul_input.append(enter_max_node.name)
            quantized_matmul_with_bias_node = Helper.create_node(
                'QuantizedMatMulWithBias', i[0] + '_quantized_mat_mul', quantized_matmul_input)
            Helper.set_attr_dtype(
                quantized_matmul_with_bias_node, 'T1', dtypes.quint8)
            Helper.set_attr_dtype(
                quantized_matmul_with_bias_node, 'T2', dtypes.qint8)
            Helper.set_attr_dtype(
                quantized_matmul_with_bias_node, 'Tbias', dtypes.float32)
            Helper.set_attr_dtype(
                quantized_matmul_with_bias_node, 'Toutput', dtypes.qint32)
            Helper.set_attr_bool(
                quantized_matmul_with_bias_node, 'transpose_a', False)
            Helper.set_attr_bool(
                quantized_matmul_with_bias_node, 'transpose_b', False)
            Helper.set_attr_string(
                quantized_matmul_with_bias_node, 'input_quant_mode', b"MIN_FIRST")
            g.add_node(quantized_matmul_with_bias_node,
                       quantize_node.name, [bias_node.name])

            if qint8_const_node.name not in graph_info:
                g.add_node(qint8_const_node, None, [enter_node_name])
                enter_node = graph_info[enter_node_name].node
                split_node = graph_info[Helper.node_name_from_input(
                    split_node_name)].node
                Helper.set_attr_dtype(enter_node, 'T', dtypes.qint8)
                Helper.set_attr_dtype(split_node, 'T', dtypes.qint8)
                graph_info[enter_node.name].node.input[0] = qint8_const_node.name
            elif qint8_const_node.name in graph_info:
                pass
            else:
                g.add_node(qint8_const_node, None, [
                           quantized_matmul_with_bias_node.name])

            if need_to_create_const_node:
                g.add_node(enter_min_node, None, [
                           quantized_matmul_with_bias_node.name])
                g.add_node(enter_max_node, None, [
                           quantized_matmul_with_bias_node.name])
                g.add_node(min_node, None,  [enter_min_node.name])
                g.add_node(max_node, None,  [enter_max_node.name])

            # create requantize node
            requantize_min_node = Helper.create_constant_node(
                i[0] + 'requant_w_min', min_output, dtypes.float32)
            requantize_max_node = Helper.create_constant_node(
                i[0] + 'requant_w_max', max_output, dtypes.float32)

            enter_req_min_node = Helper.create_node(
                'Enter', requantize_min_node.name+'_enter', [requantize_min_node.name])
            Helper.set_attr_string(
                enter_req_min_node, 'frame_name', self.rnn_details[i].encode())
            Helper.set_attr_dtype(enter_req_min_node, 'T', dtypes.float32)
            Helper.set_attr_bool(enter_req_min_node, 'is_constant', True)
            Helper.set_attr_int(enter_req_min_node, 'parallel_iterations', 32)

            enter_req_max_node = Helper.create_node(
                'Enter', requantize_max_node.name+'_enter', [requantize_max_node.name])
            Helper.set_attr_dtype(enter_req_max_node, 'T', dtypes.float32)
            Helper.set_attr_string(
                enter_req_max_node, 'frame_name', self.rnn_details[i].encode())
            Helper.set_attr_bool(enter_req_max_node, 'is_constant', True)
            Helper.set_attr_int(enter_req_max_node, 'parallel_iterations', 32)
            requantize_input = [quantized_matmul_with_bias_node.name,
                                quantized_matmul_with_bias_node.name +
                                ':1', quantized_matmul_with_bias_node.name + ':2',
                                enter_req_min_node.name, enter_req_max_node.name]
            requantize_node = Helper.create_node(
                'Requantize', i[0] + '_requantize', requantize_input)
            Helper.set_attr_dtype(requantize_node, 'out_type', dtypes.qint8)
            Helper.set_attr_dtype(requantize_node, 'Tinput', dtypes.qint32)

            g.add_node(requantize_node,
                       quantized_matmul_with_bias_node.name, [bias_node.name])
            dequantize_input = [
                requantize_node.name, requantize_node.name + ':1', requantize_node.name + ':2']
            dequantize_node = Helper.create_node(
                'Dequantize', i[0] + '_dequantize', dequantize_input)
            Helper.set_attr_dtype(dequantize_node, "T", dtypes.qint8)
            Helper.set_attr_dtype(dequantize_node, "dtype", dtypes.float32)
            Helper.set_attr_string(dequantize_node, "mode", b"MIN_FIRST")

            g.add_node(enter_req_min_node, None, [requantize_node.name])
            g.add_node(enter_req_max_node, None, [requantize_node.name])
            g.add_node(requantize_min_node, None, [enter_req_min_node.name])
            g.add_node(requantize_max_node, None, [enter_req_max_node.name])
            g.add_node(dequantize_node, requantize_node.name, last_node_name)
            if last_node_name:
                graph_info[last_node_name[0]
                           ].node.input[0] = dequantize_node.name
            g.remove_node(bias_node.name)
            g.remove_node(i[0])

            # g.remove_node(weight_node_name)

        return g.dump_graph()
