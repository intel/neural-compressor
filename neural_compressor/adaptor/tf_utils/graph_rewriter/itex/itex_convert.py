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

import copy
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from neural_compressor.utils.utility import dump_elapsed_time
from neural_compressor.adaptor.tf_utils.graph_rewriter.graph_util import GraphAnalyzer
from ..graph_base import GraphRewriterBase
from ..graph_util import GraphRewriterHelper as Helper


class GenerateITEXModel(GraphRewriterBase):
    """ Insert Q/DQ pairs before quantizable ops.

    Args: model: input model.
          data: sampling data.

    Return: converted model
    """

    def __init__(self, model, calibration_data):
        super().__init__(model)
        self.data = calibration_data

    @dump_elapsed_time("Pass GenerateITEXGraph")
    def do_transformation(self):
        itex_min_max_values = {}
        for i in self.data:
            if i.find('_requant') == -1:
                key, value = i.rsplit(':')[0], i.rsplit(':')[1]
                key = key.split('_eightbit_')[0][1:] + key[-5:]
                if key not in itex_min_max_values:
                    itex_min_max_values[key] = [float(value[1:-1])]
                else:
                    itex_min_max_values[key].append(float(value[1:-1]))
        quantizable_op_names = []
        for i in itex_min_max_values:
            if i.split('__')[0] not in quantizable_op_names:
                quantizable_op_names.append(i.split('__')[0])

        g = GraphAnalyzer()
        g.graph = copy.deepcopy(self.model.graph_def)
        graph_info = g.parse_graph()
        for op_name in quantizable_op_names:
            min_node = Helper.create_constant_node(
                op_name + '_min', np.min(itex_min_max_values[op_name+'__min']), dtypes.float32)
            max_node = Helper.create_constant_node(
                op_name + '_max', np.max(itex_min_max_values[op_name+'__max']), dtypes.float32)
            quantizable_node_input = graph_info[op_name].node.input[0]
            quant_v2_node = Helper.create_node(
                "QuantizeV2", op_name + '_quantize',
                [quantizable_node_input, min_node.name, max_node.name])
            is_asymmetric = bool(graph_info[op_name].node.op == 'MatMul')
            quant_deq_dt = dtypes.quint8 if is_asymmetric else dtypes.qint8
            Helper.set_attr_dtype(quant_v2_node, "T", quant_deq_dt)
            if not is_asymmetric:
                Helper.set_attr_string(quant_v2_node, "round_mode", b"HALF_TO_EVEN")
            Helper.set_attr_bool(quant_v2_node, "narrow_range", False if is_asymmetric else True)

            Helper.set_attr_string(
                quant_v2_node, "mode", b"MIN_FIRST" if is_asymmetric else b"SCALED")

            dequantize_node = Helper.create_node("Dequantize", op_name + '_dequantize',
                                                 [quant_v2_node.name,
                                                 quant_v2_node.name + ':1',
                                                 quant_v2_node.name + ':2'])
            Helper.set_attr_dtype(dequantize_node, "T", quant_deq_dt)
            Helper.set_attr_string(
                dequantize_node, "mode", b"MIN_FIRST" if is_asymmetric else b"SCALED")
            g.add_node(quant_v2_node,
                        graph_info[op_name].node.input[0],
                        [dequantize_node.name])
            g.add_node(dequantize_node, quant_v2_node.name, [op_name])
            g.add_node(min_node, None, [quant_v2_node.name])
            g.add_node(max_node, None, [quant_v2_node.name])
            graph_info[op_name].node.input[0] = dequantize_node.name

        g_weight = GraphAnalyzer()
        g_weight.graph = g.dump_graph()
        graph_info = g_weight.parse_graph()
        target_nodes = g_weight.query_fusion_pattern_nodes(
            [["Conv2D", "MatMul"], ["BiasAdd"], ('Relu',)])
        for i in target_nodes:
            computational_node_name = i[0]
            computational_node = graph_info[computational_node_name].node
            weight_name = computational_node.input[1]
            weight_node = graph_info[weight_name].node
            weight_tensor = tensor_util.MakeNdarray(weight_node.attr['value'].tensor)
            min_value = np.min(weight_tensor)
            max_value = np.max(weight_tensor)
            min_const_node = Helper.create_constant_node(
                weight_name + '_min', min_value, dtypes.float32)
            max_const_node = Helper.create_constant_node(
                weight_name + '_max', max_value, dtypes.float32)
            quant_node = Helper.create_node(
                "QuantizeV2", weight_name + '_quant',
                [weight_name, weight_name + '_min', weight_name + '_max'])
            dequant_node=Helper.create_node(
                "Dequantize", weight_name + '_dequant',
                [quant_node.name, quant_node.name + ':1', quant_node.name + ':2'])
            Helper.set_attr_dtype(quant_node, "T", dtypes.qint8)
            Helper.set_attr_string(quant_node, "mode", b"SCALED")
            Helper.set_attr_string(quant_node, "round_mode", b"HALF_TO_EVEN")

            Helper.set_attr_dtype(dequant_node, "T", dtypes.qint8)
            Helper.set_attr_string(dequant_node, "mode", b"SCALED")

            g_weight.add_node(quant_node, weight_name, [])
            g_weight.add_node(min_const_node, None, [quant_node.name])
            g_weight.add_node(max_const_node, None, [quant_node.name])
            g_weight.add_node(dequant_node, quant_node.name, [computational_node_name])
            computational_node.input[1] = dequant_node.name

        return g_weight.dump_graph()
