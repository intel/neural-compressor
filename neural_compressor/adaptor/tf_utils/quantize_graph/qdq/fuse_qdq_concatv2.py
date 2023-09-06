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
"""Quantize ConcatV2 to int8 op."""

import os
import re

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper as helper

from ..quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithConcatV2(QuantizeNodeBase):
    """Quantize ConcatV2 to int8 op QuantizedConcatV2."""

    def __init__(self, **kwargs):
        """Initilizaiton."""
        super().__init__(**kwargs)
        self.sorted_patterns = sorted(self.patterns, key=lambda i: len(i), reverse=True)
        self.dtype = dtypes.quint8
        self.exclude_concat_nodes = []

    def _get_node_from_name(self, name):
        """Get node struct from node name."""
        if name.startswith("^"):
            name = name[1:]
        if re.search(r"\w+:\d+", name):
            node = self.node_name_mapping[name.rsplit(":", 1)[0]].node
        else:
            node = self.node_name_mapping[name].node
        return node

    def _get_first_input_from_name(self, name):
        """Get the first input of the node."""
        if len(name) == 0:
            return name
        node = self._get_node_from_name(name)
        if len(node.input) == 0:
            return ""
        return node.input[0]

    def _quantizable_concat(self, node):
        """Check if the ConcatV2 is quantizable."""
        deq_type = []
        is_quantizable = True
        if self.performance_only or os.getenv("TF_FORCE_CONCAT_OPTS") == "1":
            _, normal_inputs = self._get_node_input(node.name)
            original_inputs = normal_inputs[: node.attr["N"].i]

            # the input chain of concatv2 is QuantizedOp -> (req) -> q -> dq -> concat
            for each_input in original_inputs:
                dq_input = self._get_first_input_from_name(each_input)
                q_input = self._get_first_input_from_name(dq_input)
                pre_input = self._get_first_input_from_name(q_input)
                if pre_input == "":
                    continue
                req_input = self._get_node_from_name(pre_input)

                # the concatv2 with these Ops as inputs can't be reranged
                if req_input.op in ["_QuantizedFusedBatchNorm", "_QuantizedFusedInstanceNorm"]:
                    is_quantizable = False
                    break
                if (
                    req_input.op == "Requantize"
                    or req_input.op == "RequantizePerChannel"
                    or req_input.op.startswith("Quantized")
                ):
                    is_quantizable = True
                    if str(self.node_name_mapping[pre_input].node.attr["out_type"]) != "":
                        deq_type.append(self.node_name_mapping[pre_input].node.attr["out_type"].type)
                    else:
                        deq_type.append(self.node_name_mapping[pre_input].node.attr["T"].type)
        else:
            for input_node_name in node.input[: node.attr["N"].i]:
                node_name = helper.node_name_from_input(input_node_name)
                if self.node_name_mapping[node_name].node.op != "Dequantize":
                    self.exclude_concat_nodes.append(node.name)
                    return False

                deq_type.append(self.node_name_mapping[node_name].node.attr["T"].type)

        if len(set(deq_type)) != 1:
            is_quantizable = False
        else:
            if self.performance_only or os.getenv("TF_FORCE_CONCAT_OPTS") == "1":
                self.dtype = dtypes.DType(deq_type[0])

        if not is_quantizable:
            self.exclude_concat_nodes.append(node.name)

        return is_quantizable

    def _apply_concatv2_quantization(self, match_node_name):
        """Quantize ConcatV2."""
        # Dequantize + ConcatV2 + QuantizeV2
        skip_node_names = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        _, normal_inputs = self._get_node_input(matched_node.node.name)
        num_input = len(normal_inputs)
        shape_input_name = normal_inputs[num_input - 1]
        original_inputs = normal_inputs[0 : num_input - 1]

        input_names = []
        min_names = []
        max_names = []
        for each_input in original_inputs:
            _, q_each_input = self._get_node_input(each_input)
            input_names.append(q_each_input[0])
            min_names.append(q_each_input[1])
            max_names.append(q_each_input[2])
            skip_node_names.append(each_input)

        if self.dtype == dtypes.qint8:
            for each_name in input_names:
                input_node = self._get_node_from_name(each_name)
                if input_node.op == "QuantizeV2":
                    helper.set_attr_dtype(input_node, "T", self.dtype)
                pre_input = self._get_first_input_from_name(each_name)
                pre_input_node = self._get_node_from_name(pre_input)
                if pre_input_node.op == "Dequantize":
                    helper.set_attr_dtype(pre_input_node, "T", self.dtype)

        all_input_names = input_names
        all_input_names.append(shape_input_name)
        all_input_names.extend(min_names)
        all_input_names.extend(max_names)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_names:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))
                quantized_op_name = node.name + "_eightbit_quantized_concatv2"
                quantized_op_type = "Quantized" + node.op
                quantized_concat_node = helper.create_node(quantized_op_type, quantized_op_name, all_input_names)

                helper.set_attr_int(quantized_concat_node, "N", len(original_inputs))
                helper.set_attr_dtype(quantized_concat_node, "T", self.dtype)
                self.add_output_graph_node(quantized_concat_node)
                self._intel_cpu_add_dequantize_result_node(quantized_op_name, node.name, dtype=self.dtype)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self, do_transform=False):
        """Get longest fusion pattern."""
        self._get_op_list()
        matched_node_name = []

        for k, v in enumerate(self.op_list):
            if v in set(fusion[1] for fusion in self.sorted_patterns):
                cur_node = self.node_name_mapping[list(self.node_name_mapping.keys())[k]].node

                if cur_node.name != self.start_node_name:
                    continue

                # possible attributes that decide output data type
                output_attr_list = ["out_type", "T", "dtype", "Taxis", "Tindices", "Tparams"]
                if not do_transform:
                    _, normal_inputs = self._get_node_input(cur_node.name)
                    original_inputs = normal_inputs[: cur_node.attr["N"].i]
                    unsupported_input_type = False
                    for each_input in original_inputs:
                        each_input_name = helper.node_name_from_input(each_input)
                        input_dtype = None
                        for output_attr in output_attr_list:
                            input_dtype_attr = self.node_name_mapping[each_input_name].node.attr[output_attr]
                            if str(input_dtype_attr) != "":
                                input_dtype = dtypes.DType(input_dtype_attr.type)
                                break
                        if input_dtype != dtypes.bfloat16 and input_dtype != dtypes.float32:
                            unsupported_input_type = True
                            break
                    if unsupported_input_type:
                        continue

                for sub_rule in self.sorted_patterns:
                    if sub_rule[0] != "Dequantize" or sub_rule[-1] != "QuantizeV2":
                        continue
                    if v != sub_rule[1]:
                        continue

                if (self.performance_only or os.getenv("TF_FORCE_CONCAT_OPTS") == "1") and not do_transform:
                    matched_node_name.clear()
                    matched_node_name.append(cur_node.name)
                    return sub_rule, matched_node_name

                if self._quantizable_concat(cur_node):
                    if dtypes.as_dtype(cur_node.attr["T"].type) == dtypes.float32 and not re.search(
                        r"map(_\d+)?/while", cur_node.name
                    ):
                        matched_node_name.clear()
                        matched_node_name.append(sub_rule[0])
                        matched_node_name.append(cur_node.name)
                        matched_node_name.append(sub_rule[-1])
                        return sub_rule, matched_node_name
                else:
                    if self.performance_only or os.getenv("TF_FORCE_CONCAT_OPTS") == "1":
                        new_inputs = []
                        control_inputs, normal_inputs = self._get_node_input(cur_node.name)
                        original_inputs = normal_inputs[: cur_node.attr["N"].i]
                        for each_input in original_inputs:
                            each_node = self._get_node_from_name(each_input)
                            if each_node.op == "Dequantize":
                                q_input = self._get_first_input_from_name(each_input)
                                if q_input == "":
                                    continue
                                if self._get_node_from_name(q_input).op == "QuantizeV2":
                                    pre_input = self._get_first_input_from_name(q_input)
                                    new_inputs.append(pre_input)
                                elif self._get_node_from_name(q_input).op == "Requantize":
                                    new_inputs.append(each_input)
                            else:
                                new_inputs.append(each_input)
                        new_inputs.append(normal_inputs[-1])
                        cur_node.ClearField("input")
                        cur_node.input.extend(new_inputs + control_inputs)

        return None, None

    def apply_the_transform(self):
        """Apply the quantization of ConcatV2."""
        self._get_op_list()
        matched_rule, matched_node_name = self.get_longest_fuse(do_transform=True)
        if matched_node_name:
            fusion_name = "".join(matched_rule)
            if fusion_name == "DequantizeConcatV2QuantizeV2":
                self._apply_concatv2_quantization(matched_node_name)
            else:  # pragma: no cover
                self.logger.info("Unknown fusion pattern {}.".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, self.exclude_concat_nodes

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)
            return self.output_graph, self.exclude_concat_nodes

        if self.remove_redundant_quant_flag:
            self.input_graph = self.remove_redundant_quantization(self.input_graph)
        return self.input_graph, self.exclude_concat_nodes
