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

import copy
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from neural_compressor.utils.utility import dump_elapsed_time
from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from ..graph_base import GraphRewriterBase
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
import re

class GenerateGraphWithQDQPattern(GraphRewriterBase):
    """ Insert Q/DQ pairs before quantizable ops.

    Args: model: input model.
          data: sampling data.
          device: cpu or gpu

    Return: converted model with QDQ pattern
    """

    def __init__(self, model, calibration_data, op_wise_config, fake_quant, fp32_ops,
                 bf16_ops, quantized_nodes, device):
        super().__init__(model)
        self.data = calibration_data
        self.op_wise_config = op_wise_config
        self.fake_quant = fake_quant
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops
        self.quantized_nodes = quantized_nodes
        self.device = device

        self.node_name_mapping = {}
        for node in self.model.graph_def.node:
            if node.name not in self.node_name_mapping:
                self.node_name_mapping[node.name] = node
            else:
                raise ValueError("Duplicate node names detected for ", node.name)

    @dump_elapsed_time("Pass GenerateGraphWithQDQPattern")
    def do_transformation(self):
        min_max_values = {}
        for i in self.data:
            if i.find('_requant') == -1:
                key, value = i.rsplit(':', 1)[0], i.rsplit(':', 1)[1]
                key = key.split('_eightbit_')[0][1:] + key[-5:]
                if key not in min_max_values:
                    min_max_values[key] = [float(value[1:-1])]
                else:
                    min_max_values[key].append(float(value[1:-1]))
        quantizable_op_names = []
        for i in min_max_values:
            if i.split('__')[0] not in quantizable_op_names:
                quantizable_op_names.append(i.split('__')[0])

        self.g = GraphAnalyzer()
        self.g.graph = copy.deepcopy(self.model.graph_def)
        self.graph_info = self.g.parse_graph()
        
        # insert QDQ pattern for op's input
        for op_name in quantizable_op_names:
            if self._ignore_insert_qdq_pattern(op_name):
                continue

            if op_name not in self.op_wise_config.keys():
                is_asymmetric = False
            else:
                op_wise_cfg = self.op_wise_config[op_name]
                is_asymmetric = op_wise_cfg[2]
            if self.graph_info[op_name].node.op == "ConcatV2":
                self._insert_qdq_pattern_for_concatv2(self.graph_info[op_name].node,
                                                      is_asymmetric)
            else:
                self._insert_qdq_pattern_for_common_ops(self.graph_info[op_name].node,
                                                        is_asymmetric)

        # insert QDQ pattern for op's weight
        self.g_weight = GraphAnalyzer()
        self.g_weight.graph = self.g.dump_graph()
        self.graph_info = self.g_weight.parse_graph()
        target_nodes = self.g_weight.query_fusion_pattern_nodes(
               [["Conv2D", "Conv3D", "DepthwiseConv2dNative", "MatMul", "BatchMatMulV2"]])
        for i in target_nodes:
            if i[0] not in quantizable_op_names:
                continue
            computational_node_name = i[0]
            if self._ignore_insert_qdq_pattern(computational_node_name):
                continue

            computational_node = self.graph_info[computational_node_name].node
            weight_name = computational_node.input[1]
            weight_node = self.graph_info[weight_name].node
            if computational_node_name in self.op_wise_config.keys():
                op_wise_cfg = self.op_wise_config[computational_node_name]
                per_channel = op_wise_cfg[0]
                weight_bit = op_wise_cfg[3]
            else:
                per_channel = False
                weight_bit = 7
            
            self._insert_qdq_pattern_for_weight_node(computational_node,
                                                     weight_node,
                                                     min_max_values,
                                                     per_channel,
                                                     weight_bit,
                                                     self.device)

        return self.g_weight.dump_graph()

    def _check_op_list(self, node_type):
        op_list = ("ConcatV2", "Conv2D", "Conv3D", "DepthwiseConv2D", "QuantizeV2", "DepthwiseConv2dNative",
                   "MaxPool", "MaxPool3D", "FusedBatchNormV3", "Requantize", "RequantizePerChannel", "AvgPool", "Pad",
                   "CropAndResize", "Dequantize", "Mean", "MatMul", "BatchMatMulV2", "FakeQuantWithMinMaxVars")
        return any([node_type.find(i) != -1 for i in op_list])

    def _find_relu_node(self, node):
        if node.op in ("Relu", "Relu6", "Elu") or \
            (node.op.find("AndRelu") != -1 and \
            ('alpha' not in node.attr or ('alpha' in node.attr and node.attr['alpha'].f == 0))):
            return True
        elif 'T' in node.attr and node.attr['T'].type in (dtypes.quint8, dtypes.uint8):
            return True
        elif (node.op.find("QuantizedConv") != -1
              or node.op.find("QuantizedDepthwiseConv") != -1 or
              node.op.find("QuantizedMatMul") != -1
              ) and ((node.op.find("Relu") == -1 and node.op.find("Elu") == -1) or \
              ('alpha' in node.attr and node.attr['alpha'].f > 0)):
            return False
        elif self._check_op_list(node.op):
            if "split:" in node.input[0]:
                input_node = self.node_name_mapping[node.input[0].rsplit(':', 1)[0]]
            else:
                input_node = self.node_name_mapping[node.input[0]]
            return self._find_relu_node(input_node)
        else:
            return False

    def _insert_qdq_pattern_for_common_ops(self, original_node, is_asymmetric):
        namespace_prefix = original_node.name + "_eightbit"
        for each_input_name in self.node_name_mapping[original_node.name].input[:1]:
            if each_input_name[0] == '^':
                continue

            if self.node_name_mapping[original_node.name].op == "MatMul" or \
               self.node_name_mapping[original_node.name].op == "BatchMatMulV2":
                dtype = dtypes.quint8
            else:
                input_node_name = Helper.node_name_from_input(each_input_name)
                if input_node_name in self.graph_info:
                    if self.graph_info[input_node_name].node.op == "Dequantize":
                        dtype = dtypes.DType(
                            self.graph_info[input_node_name].node.attr["T"].type)
                    elif self._find_relu_node(self.node_name_mapping[original_node.name]):
                        dtype = dtypes.quint8
                    else:
                        dtype = dtypes.qint8
                else:
                    dtype = dtypes.quint8 if self._find_relu_node(
                        self.node_name_mapping[original_node.name]
                    ) else dtypes.qint8
            self._insert_qdq_pattern_for_each_input(original_node.name,
                                                    namespace_prefix,
                                                    each_input_name,
                                                    is_asymmetric,
                                                    dtype)


    def _insert_qdq_pattern_for_concatv2(self, original_node, is_asymmetric):
        namespace_prefix = original_node.name + "_eightbit"
        num_input = len(original_node.input)
        original_inputs = original_node.input[0:num_input - 1]
        input_idx = 0
        for original_input_name in original_inputs:
            self._insert_qdq_pattern_for_each_input(original_node.name,
                                                    namespace_prefix,
                                                    original_input_name,
                                                    is_asymmetric,
                                                    dtypes.quint8,
                                                    input_idx)
            input_idx += 1


    def _insert_qdq_pattern_for_each_input(self, op_name, namespace_prefix,
                                           input_name, is_asymmetric,
                                           dtype=dtypes.quint8, input_index=0):
        """Takes one float input to an op, and converts it to quantized form."""
        unique_input_name = input_name.replace(":", "__port__").replace("^", "__hat__")
        min_input_name = namespace_prefix + "_min_" + unique_input_name
        max_input_name = namespace_prefix + "_max_" + unique_input_name
        quantize_input_name = namespace_prefix + "_quantize_" + unique_input_name

        reshape_dims_name = namespace_prefix + "_reshape_dims"
        reduction_dims_name = namespace_prefix + "_reduction_dims"

        if self.fake_quant: # pragma: no cover
            min_node = Helper.create_constant_node(
                min_input_name, -1., dtypes.float32)
            max_node = Helper.create_constant_node(
                max_input_name, 1., dtypes.float32)
            quant_v2_node = Helper.create_node(
                "QuantizeV2", quantize_input_name,
                [input_name, min_input_name, max_input_name])
            Helper.set_attr_dtype(quant_v2_node, "T", dtype)
            if not is_asymmetric:
                Helper.set_attr_string(quant_v2_node, "round_mode", b"HALF_TO_EVEN")
            #Helper.set_attr_bool(quant_v2_node, "narrow_range", False if is_asymmetric else True)
            Helper.set_attr_string(
                quant_v2_node, "mode", b"MIN_FIRST" if is_asymmetric else b"SCALED")
            if "Concat" in self.graph_info[op_name].node.op:
                dequantize_node = Helper.create_node(
                    "Dequantize", op_name + '_dequantize_' + str(input_index),
                    [quant_v2_node.name, quant_v2_node.name + ':1', quant_v2_node.name + ':2'])
            else:
                dequantize_node = Helper.create_node(
                    "Dequantize", op_name + '_dequantize',
                    [quant_v2_node.name, quant_v2_node.name + ':1', quant_v2_node.name + ':2'])
            Helper.set_attr_dtype(dequantize_node, "T", dtype)
            Helper.set_attr_string(
                dequantize_node, "mode", b"MIN_FIRST" if is_asymmetric else b"SCALED")
            self.g.add_node(quant_v2_node,
                            self.graph_info[op_name].node.input[0],
                            [dequantize_node.name])
            self.g.add_node(dequantize_node, quant_v2_node.name, [op_name])
            self.g.add_node(min_node, None, [quant_v2_node.name])
            self.g.add_node(max_node, None, [quant_v2_node.name])
            self.graph_info[op_name].node.input[input_index] = dequantize_node.name
        else:
            reshape_dims_node = Helper.create_constant_node(
                reshape_dims_name, -1, dtypes.int32, [1])
            reduction_dims_node = Helper.create_constant_node(
                reduction_dims_name, 0, dtypes.int32, [1])
            reshape_input_name = namespace_prefix + "_reshape_" + unique_input_name
            min_input_name = namespace_prefix + "_min_" + unique_input_name
            max_input_name = namespace_prefix + "_max_" + unique_input_name
            quantize_input_name = namespace_prefix + "_quantize_" + unique_input_name

            reshape_input_node = Helper.create_node(
                "Reshape", reshape_input_name,
                [input_name, reshape_dims_name])
            Helper.set_attr_dtype(reshape_input_node, "T", dtypes.float32)

            min_input_node = Helper.create_node(
                "Min", min_input_name, [reshape_input_name, reduction_dims_name])
            Helper.set_attr_dtype(min_input_node, "T", dtypes.float32)
            Helper.set_attr_dtype(min_input_node, "Tidx", dtypes.int32)
            Helper.set_attr_bool(min_input_node, "keep_dims", False)

            max_input_node = Helper.create_node(
                "Max", max_input_name, [reshape_input_name, reduction_dims_name])
            Helper.set_attr_dtype(max_input_node, "T", dtypes.float32)
            Helper.set_attr_dtype(max_input_node, "Tidx", dtypes.int32)
            Helper.set_attr_bool(max_input_node, "keep_dims", False)

            quant_v2_node = Helper.create_node("QuantizeV2", quantize_input_name,
                [input_name, min_input_name, max_input_name])
            Helper.set_attr_dtype(quant_v2_node, "T", dtype)
            if not is_asymmetric:
                Helper.set_attr_string(quant_v2_node, "round_mode", b"HALF_TO_EVEN")
            #Helper.set_attr_bool(quant_v2_node, "narrow_range", False if is_asymmetric else True)
            Helper.set_attr_string(
                quant_v2_node, "mode", b"MIN_FIRST" if is_asymmetric else b"SCALED")

            if "Concat" in self.graph_info[op_name].node.op:
                dequantize_node = Helper.create_node(
                    "Dequantize", op_name + '_dequantize_' + str(input_index),
                    [quant_v2_node.name, quant_v2_node.name + ':1', quant_v2_node.name + ':2'])
            else:
                dequantize_node = Helper.create_node(
                    "Dequantize", op_name + '_dequantize',
                    [quant_v2_node.name, quant_v2_node.name + ':1', quant_v2_node.name + ':2'])
            Helper.set_attr_dtype(dequantize_node, "T", dtype)
            Helper.set_attr_string(
                dequantize_node, "mode", b"MIN_FIRST" if is_asymmetric else b"SCALED")

            self.g.add_node(quant_v2_node,
                            self.graph_info[op_name].node.input[0],
                            [dequantize_node.name])
            self.g.add_node(dequantize_node, quant_v2_node.name, [op_name])
            self.g.add_node(reshape_dims_node, None, [reshape_input_name])
            self.g.add_node(reduction_dims_node, None, [min_input_name, max_input_name])
            self.g.add_node(reshape_input_node, reshape_dims_name, [min_input_name, max_input_name])
            self.g.add_node(min_input_node, reshape_input_name, [quant_v2_node.name])
            self.g.add_node(max_input_node, reshape_input_name, [quant_v2_node.name])
            self.graph_info[op_name].node.input[input_index] = dequantize_node.name

    def _insert_qdq_pattern_for_weight_node(self,
                                            computational_node,
                                            weight_node,
                                            min_max_values,
                                            per_channel,
                                            weight_bit=7.0,
                                            device='cpu'):
        host_op_type = computational_node.op
        base_name = weight_node.name + "_"
        qint8_const_name = base_name + "qint8_const"
        min_name = base_name + "min"
        max_name = base_name + "max"
        epsilon = 1e-4  # Needs to be set empirically if accuracy is not satisfactory
        range_coefficent = 127 / (2 ** weight_bit - 1)
        min_value = 0
        max_value = 0
       
        # The weight node of BatchMatMul may have no value
        if 'value' in weight_node.attr and \
           host_op_type in ("Conv2D", "MatMul", "BatchMatMulV2", "Conv3D"):
            float_tensor = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
            if per_channel:
                if host_op_type == 'Conv3D':
                    ranges = np.abs(float_tensor).max(axis=(0, 1, 2, 3))
                elif host_op_type == 'Conv2D':
                    ranges = np.abs(float_tensor).max(axis=(0, 1, 2))
                else:
                    ranges = np.abs(float_tensor).max(axis=(0, 1))

                ranges *= range_coefficent
                min_value = -ranges
                max_value = ranges
                # nudging min-max values outside epsilon radius around zero
                ranges[ranges < epsilon] = epsilon
                min_value[np.abs(min_value) < epsilon] = -epsilon
                max_value[np.abs(max_value) < epsilon] = epsilon
                qint8_tensor = (np.around(float_tensor *127.0/ranges)).astype(np.int8)
            else:
                min_value = np.min(float_tensor)
                max_value = np.max(float_tensor)
                min_value *= range_coefficent
                max_value *= range_coefficent
                min_value = min(min_value, 0.0)
                if min_value == max_value:
                    if abs(min_value) < 0.000001:
                        max_value = min_value + 1.0
                    elif min_value > 0:
                        max_value = 2 * min_value
                    else:
                        max_value = min_value / 2.0
                range_value = np.max(np.abs([min_value, max_value]))
                qint8_tensor = (np.around(float_tensor * 127.0 / range_value)).astype(np.int8)
                qint8_tensor = np.clip(qint8_tensor, -127, 127).astype(np.int8)
                min_value = -range_value
                max_value = range_value
        elif host_op_type == "DepthwiseConv2dNative":
            float_tensor = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
            # get the max values based on dim 0 and 1 for depthwise conv
            # since, the output channel will be dim 2 * dim 3
            ranges = np.abs(float_tensor).max(axis=(0, 1))
            ranges = ranges.flatten()
            min_value = -ranges
            max_value = ranges
            # nudging min-max values outside epsilon radius around zero
            ranges[ranges < epsilon] = epsilon
            min_value[np.abs(min_value) < epsilon] = -epsilon
            max_value[np.abs(max_value) < epsilon] = epsilon
            # Since output channel will be 1 dim which is dim 2 * dim 3
            # When divide by range, qint8_tensor needs to be 3 dim
            # where, 3rd dim should be same dim of ranges
            a, b, c, d = float_tensor.shape
            qint8_tensor = (np.around(float_tensor.reshape(a, b, c * d) * 127.0 /
                            ranges)).astype(np.int8)
            # get the shape back to 4 dim
            qint8_tensor = qint8_tensor.reshape(a, b, c, d)
        else:
            min_value = np.min(min_max_values[computational_node.name+'__min'])
            max_value = np.max(min_max_values[computational_node.name+'__max'])

        min_node = Helper.create_constant_node(min_name, min_value,
                                                            dtypes.float32, device=device)
        max_node = Helper.create_constant_node(max_name, max_value,
                                                            dtypes.float32, device=device)
        quant_node = Helper.create_node(
                "QuantizeV2", qint8_const_name + '_quant',
                [weight_node.name, min_name, max_name])
        dequant_node = Helper.create_node(
            "Dequantize", base_name + '_dequant',
            [quant_node.name, quant_node.name + ':1', quant_node.name + ':2'])
        Helper.set_attr_dtype(quant_node, "T", dtypes.qint8)
        Helper.set_attr_string(quant_node, "mode", b"SCALED")
        Helper.set_attr_string(quant_node, "round_mode", b"HALF_TO_EVEN")
        Helper.set_attr_dtype(dequant_node, "T", dtypes.qint8)
        Helper.set_attr_string(dequant_node, "mode", b"SCALED")
        if per_channel:
            if host_op_type == 'Conv2D':
                Helper.set_attr_int(quant_node, 'axis', 3)
                Helper.set_attr_int(dequant_node, 'axis', 3)
            elif host_op_type == 'Conv3D':
                Helper.set_attr_int(quant_node, 'axis', 4)
                Helper.set_attr_int(dequant_node, 'axis', 4)
            elif host_op_type == 'MatMul':
                Helper.set_attr_int(quant_node, 'axis', 1)
                Helper.set_attr_int(dequant_node, 'axis', 1)
            else:
                Helper.set_attr_int(quant_node, 'axis', -1)
                Helper.set_attr_int(dequant_node, 'axis', -1)

        self.g_weight.add_node(quant_node, weight_node.name, [])
        self.g_weight.add_node(min_node, None, [quant_node.name])
        self.g_weight.add_node(max_node, None, [quant_node.name])
        self.g_weight.add_node(dequant_node, quant_node.name, [computational_node.name])
        computational_node.input[1] = dequant_node.name

    def _ignore_insert_qdq_pattern(self, matched_node_name):
        if matched_node_name in self.fp32_ops or \
           matched_node_name in self.bf16_ops:
            return True

        if matched_node_name not in self.op_wise_config and (matched_node_name, ) not in self.quantized_nodes:
            return True

        #TODO Remove below two lines once the TF enabled the QuantizedMatMul while
        # transpose_a/transpose_a could be set to True.
        if self.graph_info[matched_node_name].node.op == "MatMul":
            if self.graph_info[matched_node_name].node.attr["transpose_a"].b == True or \
               self.graph_info[matched_node_name].node.attr["transpose_b"].b == True:
                return True

        return False


