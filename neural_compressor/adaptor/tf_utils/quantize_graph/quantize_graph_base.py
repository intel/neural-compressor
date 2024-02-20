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
"""QuantizeGraph Base Class."""

import logging
from collections import namedtuple

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper as helper
from neural_compressor.adaptor.tf_utils.util import version1_eq_version2, version1_gt_version2, version1_lt_version2


class QuantizeGraphBase:
    """This is the base class for quantize graph."""

    def __init__(self, output_node_names):
        """Initilizaiton."""
        self.output_node_names = output_node_names
        self.transformers = {}

    def register_transformer(self, node_name, entry):
        """Register transformers API."""
        if node_name not in self.transformers:
            self.transformers[node_name] = {}

        self.transformers[node_name] = entry

    def do_transform(self):
        """This is the virtual interface need to be implemented by derived class.

        :return:
        """
        pass

    def remove_dead_nodes(self, input_graph, output_names):
        """Removes nodes that are no longer needed for inference from the graph."""
        return tf.compat.v1.graph_util.extract_sub_graph(input_graph, output_names)


class QuantizeNodeBase:
    """This is the base class for nodes fusion.

    Arguments:
        object {[type]} -- [description]
    """

    node_details = namedtuple("node_details", ["node", "output"])

    def __init__(self, **kwargs):
        """Initilizaiton."""
        self.logger = logging.getLogger("neural_compressor")
        input_graph = kwargs["input_graph"]

        assert isinstance(input_graph, graph_pb2.GraphDef)

        self.input_graph = input_graph

        self._parse_graph()
        self.output_node_maps = {}
        self.output_graph = graph_pb2.GraphDef()
        self.quantized_node_dict = {}
        self.patterns = kwargs["patterns"]
        self.remove_redundant_quant_flag = kwargs["remove_redundant_quant_flag"]
        self.fake_quant = kwargs["fake_quant"] if "fake_quant" in kwargs else False
        self.frame_info = kwargs["frame_info"] if "frame_info" in kwargs else None
        self.per_channel, self.is_asymmetric = kwargs["op_wise_cfg"][0], kwargs["op_wise_cfg"][2]
        self.op_wise_config_name_list = kwargs["op_wise_config_name_list"]
        self.weight_bit = kwargs["op_wise_cfg"][3]
        self.start_node_name = kwargs["start_node_name"]
        self.device = kwargs["device"]
        self.new_api = kwargs["new_api"]
        self.performance_only = kwargs["performance_only"]
        self.itex_mode = kwargs["itex_mode"]
        self.enable_s8 = bool(
            version1_gt_version2(tf.version.VERSION, "2.1.0")
            or version1_eq_version2(tf.version.VERSION, "2.1.0")
            or tf.version.VERSION.find("1.15.0-up") != -1
        )

    def apply_the_transform(self):
        """This is the virtual interface to be implemented by derived class.

        :return: transformed graphdef
        """
        pass

    def get_longest_fuse(self):
        """This is the virtual interface to be implemented by derived class."""
        pass

    def _insert_dummy_biasadd(self, match_node_name, matched_node):  # pragma: no cover
        """Insert dummy BiasAdd op."""
        target_node_name = matched_node.node.name
        matmul_a_node_name = helper.node_name_from_input(matched_node.node.input[0])
        matmul_a_node = self.node_name_mapping[matmul_a_node_name].node
        matmul_b_node_name = helper.node_name_from_input(matched_node.node.input[1])
        matmul_b_node = self.node_name_mapping[matmul_b_node_name].node

        if matmul_a_node.op == "Const" and matmul_b_node.op != "Const":
            pass
        else:
            from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer

            g = GraphAnalyzer()
            g.graph = self.input_graph
            graph_info = g.parse_graph()
            next_node_names = graph_info[matched_node.node.name].outputs
            bias_node_name = target_node_name + "_dummy_biasadd"
            bias_const_node_name = target_node_name + "_fake_const"

            if matched_node.node.op == "MatMul":
                t_b_index = 0 if matched_node.node.attr["transpose_b"].b else 1
            if (
                matched_node.node.op in ("Conv2D" or "DepthwiseConv2dNative")
                and matched_node.node.attr["data_format"].s == b"NHWC"
            ):
                t_b_index = 3
            elif (
                matched_node.node.op in ("Conv2D" or "DepthwiseConv2dNative")
                and matched_node.node.op.attr["data_format"].s == b"NCHW"
            ):
                t_b_index = 1
            elif matched_node.node.op == "Conv3D" and matched_node.node.attr["data_format"].s == b"NDHWC":
                t_b_index = 4
            elif matched_node.node.op == "Conv3D" and matched_node.node.attr["data_format"].s == b"NCDHW":
                t_b_index = 1
            bias_add_length = matmul_b_node.attr["value"].tensor.tensor_shape.dim[t_b_index].size

            bias_add_content = [0.0] * bias_add_length

            bias_const_node = helper.create_constant_node(
                bias_const_node_name, bias_add_content, dtypes.float32, shape=[bias_add_length]
            )
            bias_node = helper.create_node("BiasAdd", bias_node_name, [target_node_name, bias_const_node_name])
            helper.set_attr_dtype(bias_node, "T", dtypes.float32)

            g.add_node(bias_node, target_node_name, next_node_names)
            g.add_node(bias_const_node, None, [bias_node_name])
            self.input_graph = g.dump_graph()
            self._parse_graph(self.input_graph)
            new_match_node_name = match_node_name[:1] + [bias_node_name] + match_node_name[1:]
            new_match_node_name = match_node_name[:1] + [bias_node_name] + match_node_name[1:]

            return new_match_node_name

    def _is_match(self, patterns):
        """Detect the rule matched nodes collections.

        Returns:
            [List] -- [the matched rule]
            [String] -- [the list contains the matched node name]
        """
        matched_node_name = []

        for k, v in enumerate(self.op_list):
            if v in set(fusion[0] for fusion in patterns):
                cur_node = self.node_name_mapping[list(self.node_name_mapping.keys())[k]].node
                if cur_node.name != self.start_node_name:
                    continue

                if ((v in ("Conv2D", "DepthwiseConv2dNative") and not self.enable_s8)) and not self._find_relu_node(
                    cur_node
                ):
                    continue

                for sub_rule in patterns:
                    if v != sub_rule[0]:
                        continue

                    sub_rule_len = len(sub_rule)
                    check_hardswish = True if sub_rule_len > 4 else False
                    self.logger.debug("Try to apply rule: {}".format(sub_rule))

                    cur_node_name = list(self.node_name_mapping.keys())[k]

                    matched_node_name.clear()

                    matched_node_name.append(cur_node_name)
                    count = 0
                    while sub_rule_len > 1:
                        if not self.node_name_mapping[cur_node_name].output:
                            self.logger.debug("Fail to match {}".format(sub_rule))
                            break

                        next_node_name = self.node_name_mapping[cur_node_name].output[0]

                        is_shared_output = True if len(self.node_name_mapping[cur_node_name].output) > 1 else False

                        add_op_quantizable = True
                        is_hardswish = False
                        if is_shared_output:  # pragma: no cover
                            if next_node_name.find("hard_swish") != -1:
                                self.logger.debug("Find Hard Swish pattern ......")
                                is_hardswish = True
                                count = count + 1
                                if next_node_name.find("add") == -1:
                                    next_node_name = self.node_name_mapping[cur_node_name].output[1]
                            else:
                                add_op_quantizable = False
                        next_node_op = self.node_name_mapping[next_node_name].node.op

                        if next_node_op in ("Add", "AddV2", "AddN"):
                            next_node = self.node_name_mapping[next_node_name].node
                            next_node_inputs = list(next_node.input)
                            cur_node_index = next_node_inputs.index(cur_node_name)

                            for index, input_name in enumerate(next_node_inputs):
                                node_type = self.node_name_mapping[helper.node_name_from_input(input_name)].node.op
                                if input_name != cur_node_name and index < cur_node_index and node_type != "Dequantize":
                                    add_op_quantizable = False
                                    break

                        if add_op_quantizable and next_node_op == sub_rule[1 - sub_rule_len]:
                            if not is_shared_output:
                                matched_node_name.append(next_node_name)
                                sub_rule_len -= 1
                                cur_node_name = next_node_name
                            elif is_hardswish and self.new_api:  # pragma: no cover
                                matched_node_name.append(next_node_name)
                                sub_rule_len -= 1
                                cur_node_name = next_node_name
                            else:
                                matched_node_name.clear()
                                self.logger.debug("Fail to match {}.".format(sub_rule))
                                break
                        else:
                            matched_node_name.clear()
                            self.logger.debug("Fail to match {}.".format(sub_rule))
                            break

                    if sub_rule_len == 1:
                        if (
                            check_hardswish
                            and sub_rule[-1] == "Mul"
                            and sub_rule[-2] == "Mul"
                            and sub_rule[-3] == "Relu6"
                            and sub_rule[-4] == "Add"
                            and count != 1
                        ):
                            matched_node_name.clear()
                            self.logger.debug("Fail to match {}.".format(sub_rule))
                            break
                        self.logger.debug("Match {} on nodes {}.".format(sub_rule, matched_node_name))
                        return sub_rule, matched_node_name

        return None, None

    def _need_to_check(self, node_type):
        """Check op list."""
        op_list = (
            "ConcatV2",
            "Conv2D",
            "Conv3D",
            "DepthwiseConv2D",
            "QuantizeV2",
            "DepthwiseConv2dNative",
            "MaxPool",
            "MaxPool3D",
            "FusedBatchNormV3",
            "Requantize",
            "RequantizePerChannel",
            "AvgPool",
            "Pad",
            "CropAndResize",
            "Dequantize",
            "Mean",
            "MatMul",
            "BatchMatMulV2",
            "FakeQuantWithMinMaxVars",
            "_MklFusedInstanceNorm",
        )
        return any([node_type.find(i) != -1 for i in op_list])

    def _find_relu_node(self, node):
        """Find relu node algorithm to identify the positive input."""
        # if node.op.find("HardSwish") != -1:
        #    return False
        if (
            node.op in ("Relu", "Relu6")
            or (
                node.op.find("AndRelu") != -1
                and ("alpha" not in node.attr or ("alpha" in node.attr and node.attr["alpha"].f == 0))
            )
        ) and (
            node.op != "Relu"
            or not self.new_api
            or not self.performance_only
            or self.node_name_mapping[helper.node_name_from_input(node.input[0])].node.op.find("FusedBatchNorm") == -1
            or self.node_name_mapping[helper.node_name_from_input(node.input[0])].node.attr["is_training"].b
            or len(self.node_name_mapping[helper.node_name_from_input(node.input[0])].output) > 1
        ):
            return True
        elif "T" in node.attr and node.attr["T"].type in (dtypes.quint8, dtypes.uint8):
            return True
        elif (
            node.op.find("QuantizedConv") != -1
            or node.op.find("QuantizedDepthwiseConv") != -1
            or node.op.find("QuantizedMatMul") != -1
        ) and (node.op.find("Relu") == -1 or ("alpha" in node.attr and node.attr["alpha"].f > 0)):
            return False
        elif self._need_to_check(node.op):
            input_node = self.node_name_mapping[helper.node_name_from_input(node.input[0])]
            return self._find_relu_node(input_node.node)
        else:
            return False

    def _reset_output_node_maps(self):
        """Reset output node maps."""
        self.output_node_maps = {}

    def _get_op_list(self):
        """Get the op list."""
        self.op_list = []
        for _, v in enumerate(self.node_name_mapping):
            self.op_list.append(self.node_name_mapping[v].node.op)

    def _get_node_input(self, node_name):
        """Return control_input name, non-control_input node name."""
        return [i for i in self.node_name_mapping[node_name].node.input if i[0] == "^"], [
            i for i in self.node_name_mapping[node_name].node.input if i[0] != "^"
        ]

    def _intel_cpu_add_dequantize_result_node(
        self, quantized_output_name, original_node_name, dtype=dtypes.quint8, min_tensor_index=1, performance_only=False
    ):
        """Add Dequantize node after the quantized node."""
        min_max_inputs = [
            "%s:%s" % (quantized_output_name, min_tensor_index),
            "%s:%s" % (quantized_output_name, min_tensor_index + 1),
        ]
        dequantize_name = original_node_name

        dequantize_node = helper.create_node(
            "Dequantize", dequantize_name, [quantized_output_name, min_max_inputs[0], min_max_inputs[1]]
        )
        helper.set_attr_dtype(dequantize_node, "T", dtype)
        if performance_only:
            helper.set_attr_string(dequantize_node, "mode", b"SCALED")
        else:
            helper.set_attr_string(dequantize_node, "mode", b"MIN_FIRST" if self.is_asymmetric else b"SCALED")
        self.add_output_graph_node(dequantize_node)

    def eightbitize_single_input_tensor_node(self, original_node, add_op_function):
        """Quantize the single input tensor node."""
        quantized_op_name = original_node.name + "_eightbit_quantized"
        quantized_op_type = "Quantized" + original_node.op
        if version1_gt_version2(tf.version.VERSION, "2.7.0") and original_node.op == "MaxPool3D":
            quantized_op_type = "_Quantized" + original_node.op
        all_input_names = self._add_eightbit_prologue_nodes(original_node.name)
        quantized_op_node = helper.create_node(quantized_op_type, quantized_op_name, all_input_names)
        add_op_function(original_node, quantized_op_node)
        self.add_output_graph_node(quantized_op_node)
        deq_type = dtypes.quint8 if self._find_relu_node(original_node) else dtypes.qint8
        self._intel_cpu_add_dequantize_result_node(quantized_op_name, original_node.name, dtype=deq_type)

    def _add_eightbit_prologue_nodes(self, original_node):
        """Add quantized prologue nodes."""
        namespace_prefix = original_node + "_eightbit"
        reshape_dims_name, reduction_dims_name = self._add_common_quantization_nodes(
            namespace_prefix, helper.node_name_from_input(self.node_name_mapping[original_node].node.input[0])
        )
        input_names = []
        min_max_names = []
        for each_input_name in self.node_name_mapping[original_node].node.input[:1]:
            if each_input_name[0] == "^":
                continue

            if self.node_name_mapping[original_node].node.op == "MatMul":
                # mkl ops _MklQuantizedMatMulWithBiasAndRelu|AndRequantize
                # requires the T1 data type as quint8
                dtype = dtypes.quint8
            else:
                input_node_name = helper.node_name_from_input(each_input_name)
                if input_node_name in self.output_node_maps:
                    if self.output_node_maps[input_node_name].op == "Dequantize":
                        dtype = dtypes.DType(self.output_node_maps[input_node_name].attr["T"].type)
                    elif self._find_relu_node(self.node_name_mapping[original_node].node):
                        dtype = dtypes.quint8
                    else:
                        dtype = dtypes.qint8
                else:
                    dtype = (
                        dtypes.quint8
                        if self._find_relu_node(self.node_name_mapping[original_node].node)
                        else dtypes.qint8
                    )
                    if "FusedBatchNorm" in self.node_name_mapping[original_node].node.op:
                        dtype = dtypes.qint8
            quantize_input_name, min_input_name, max_input_name = self._eightbitize_input_to_node(
                namespace_prefix, each_input_name, reshape_dims_name, reduction_dims_name, dtype=dtype
            )
            input_names.append(quantize_input_name)
            min_max_names.append(min_input_name)
            min_max_names.append(max_input_name)
        all_input_names = []
        all_input_names.extend(input_names)
        if min_max_names:
            all_input_names.extend(min_max_names)

        for original_input_name in self.node_name_mapping[original_node].node.input:
            if original_input_name[0] == "^":
                all_input_names.append(original_input_name)
        return all_input_names

    def _add_eightbit_prologue_nodes_for_enter(self, original_node, enter_node=None):
        """Add quantized prologue nodes for control edge case."""
        namespace_prefix = original_node + "_eightbit"
        reshape_dims_name, reduction_dims_name = self._add_common_quantization_nodes(
            namespace_prefix,
            helper.node_name_from_input(self.node_name_mapping[original_node].node.input[0]),
            enter_node=enter_node,
        )
        input_names = []
        min_max_names = []
        enter_input_name = self.node_name_mapping[original_node].node.input[1]
        if enter_input_name[0] != "^":
            if self.node_name_mapping[original_node].node.op == "MatMul":
                # mkl ops _MklQuantizedMatMulWithBiasAndRelu|AndRequantize
                # requires the T1 data type as quint8
                dtype = dtypes.quint8
            else:
                input_node_name = helper.node_name_from_input(enter_input_name)
                if input_node_name in self.output_node_maps:
                    if self.output_node_maps[input_node_name].op == "Dequantize":
                        dtype = dtypes.DType(self.output_node_maps[input_node_name].attr["T"].type)
                    elif self._find_relu_node(self.node_name_mapping[original_node].node):
                        dtype = dtypes.quint8
                    else:
                        dtype = dtypes.qint8
                else:
                    dtype = (
                        dtypes.quint8
                        if self._find_relu_node(self.node_name_mapping[original_node].node)
                        else dtypes.qint8
                    )
                    if "FusedBatchNorm" in self.node_name_mapping[original_node].node.op:
                        dtype = dtypes.qint8
            quantize_input_name, min_input_name, max_input_name = self._eightbitize_input_to_node(
                namespace_prefix, enter_input_name, reshape_dims_name, reduction_dims_name, dtype=dtype
            )
            input_names.append(quantize_input_name)
            min_max_names.append(min_input_name)
            min_max_names.append(max_input_name)

        all_input_names = []
        all_input_names.extend(input_names)
        if min_max_names:
            all_input_names.extend(min_max_names)

        for original_input_name in self.node_name_mapping[original_node].node.input:
            if original_input_name[0] == "^":
                all_input_names.append(original_input_name)
        return all_input_names

    def _add_common_quantization_nodes(self, namespace_prefix, control_input_names=None, enter_node=None):
        """Builds constant nodes needed for quantization of inputs."""
        reshape_dims_name = namespace_prefix + "_reshape_dims"
        reduction_dims_name = namespace_prefix + "_reduction_dims"

        reshape_dims_node = helper.create_constant_node(reshape_dims_name, -1, dtypes.int32, [1])

        if enter_node:  # pragma: no cover
            reshape_dims_enter_node = helper.create_node("Enter", reshape_dims_name + "_enter", [reshape_dims_name])
            helper.set_attr_string(reshape_dims_enter_node, "frame_name", enter_node.attr["frame_name"].s)
            helper.set_attr_dtype(reshape_dims_enter_node, "T", dtypes.int32)
            helper.set_attr_bool(reshape_dims_enter_node, "is_constant", True)
            helper.set_attr_int(
                reshape_dims_enter_node, "parallel_iterations", enter_node.attr["parallel_iterations"].i
            )
            self.add_output_graph_node(reshape_dims_enter_node)

        self.add_output_graph_node(reshape_dims_node)
        reduction_dims_node = helper.create_constant_node(reduction_dims_name, 0, dtypes.int32, [1])

        if enter_node:  # pragma: no cover
            reduction_dims_enter_node = helper.create_node(
                "Enter", reduction_dims_name + "_enter", [reduction_dims_name]
            )
            helper.set_attr_string(reduction_dims_enter_node, "frame_name", enter_node.attr["frame_name"].s)
            helper.set_attr_dtype(reduction_dims_enter_node, "T", dtypes.int32)
            helper.set_attr_bool(reduction_dims_enter_node, "is_constant", True)
            helper.set_attr_int(
                reduction_dims_enter_node, "parallel_iterations", enter_node.attr["parallel_iterations"].i
            )
            self.add_output_graph_node(reduction_dims_enter_node)

        self.add_output_graph_node(reduction_dims_node)
        return (
            reshape_dims_enter_node.name if enter_node else reshape_dims_name,
            reduction_dims_enter_node.name if enter_node else reduction_dims_name,
        )

    def add_output_graph_node(self, output_node):
        """Inserts one node into the new graph."""
        assert output_node.name not in self.output_node_maps
        self.output_node_maps[output_node.name] = output_node
        self.output_graph.node.extend([output_node])

    def _parse_graph(self, input_graph=None):
        """Parse the graph and get the input node and output node name details."""
        self.logger.debug("Start to parse graph.")

        graph = self.input_graph if input_graph is None else input_graph
        self.node_name_mapping = {}

        for node in graph.node:
            # each_node = self.node_details(node=node,  output=[])
            if node.name in self.node_name_mapping:
                raise ValueError("Duplicate Node Found when _parse_graph, the node name is {}".format(node.name))
            self.node_name_mapping[node.name] = self.node_details(node=node, output=[])
        for node_name in self.node_name_mapping:
            for each_input in self.node_name_mapping[node_name].node.input:
                self.node_name_mapping[helper.node_name_from_input(each_input)].output.append(node_name)

    def remove_redundant_quantization(self, old_graph):
        """Remove the redundant Quantize/QuantizeV2 nodes."""
        old_nodes_map = self.create_nodes_map(old_graph)
        self.output_graph = graph_pb2.GraphDef()
        inputs_to_rename = {}
        # We go through all the nodes, looking for any that match the patterns we
        # know how to optimize away.
        for node in old_graph.node:
            # We always start with a Quantize node, and examine its inputs to see if
            # they are in a form that can be removed.
            if node.op not in ["Quantize", "QuantizeV2"]:
                continue

            dequantize_node_name = helper.node_name_from_input(node.input[0])

            assert dequantize_node_name in old_nodes_map, (
                "Input node name '" + dequantize_node_name + "' not found in node '" + node.name + "'"
            )

            dequantize_node = old_nodes_map[dequantize_node_name]
            # Do we have a Dequantize feeding in, with the same type as the
            # Quantize?
            if dequantize_node.op != "Dequantize":
                continue

            if node.attr["T"] != dequantize_node.attr["T"]:
                continue

            # Now look at the other inputs, and ensure they're Min/Max nodes.
            min_node_name = helper.node_name_from_input(node.input[1])
            max_node_name = helper.node_name_from_input(node.input[2])
            min_node = old_nodes_map[min_node_name]
            max_node = old_nodes_map[max_node_name]
            is_min_right_type = min_node.op in ["Min", "Dequantize"]
            is_max_right_type = max_node.op in ["Max", "Dequantize"]
            if not is_min_right_type or not is_max_right_type:
                self.logger.info("Not find expected types on inputs {}, {}.".format(min_node.op, max_node.op))
                continue
            min_node_input_name = helper.node_name_from_input(min_node.input[0])
            max_node_input_name = helper.node_name_from_input(max_node.input[0])
            # There are two different patterns for Min nodes we can recognize, one
            # where the input comes directly from the same one as the Max, and
            # another where we run it through another Min first, so check for
            # both.
            is_same_input = False
            if min_node_input_name == max_node_input_name:
                is_same_input = True
            else:
                first_min_node_input = old_nodes_map[min_node_input_name]
                if first_min_node_input.op == "Concat":
                    second_min_node_name = helper.node_name_from_input(first_min_node_input.input[1])
                    second_min_node = old_nodes_map[second_min_node_name]
                    if second_min_node.op == "Min":
                        second_min_node_input_name = helper.node_name_from_input(second_min_node.input[0])
                        is_same_input = second_min_node_input_name == max_node_input_name
            if not is_same_input:
                self.logger.info("Different min/max inputs {}.".format(min_node_input_name))
                continue
            # We recognize this pattern, so mark the graph edges to be rewired to
            # route around it entirely, since we know it's a no-op.
            dequantize_source_name = helper.node_name_from_input(dequantize_node.input[0])
            node_tensor_name = helper.ensure_tensor_name_has_port(node.name)
            min_tensor_name = node.name + ":1"
            max_tensor_name = node.name + ":2"

            inputs_to_rename[node_tensor_name] = dequantize_source_name
            inputs_to_rename[min_tensor_name] = dequantize_node.input[1]
            inputs_to_rename[max_tensor_name] = dequantize_node.input[2]
        # Finally we apply all the rewiring we've marked to the graph.
        for node in old_graph.node:
            for index, input_full_name in enumerate(node.input):
                input_name = helper.ensure_tensor_name_has_port(input_full_name)
                if input_name in inputs_to_rename:
                    node.input[index] = inputs_to_rename[input_name]
            self.add_output_graph_node(node)
        return self.output_graph

    def create_nodes_map(self, graph):
        """Builds a mapping of node names to their defs from the graph."""
        nodes_map = {}
        for node in graph.node:
            assert node.name not in nodes_map, "Duplicate node names detected."
            nodes_map[node.name] = node

        return nodes_map

    def _add_quantize_down_nodes(
        self, original_node, quantized_output_name, requantize_type=dtypes.quint8, is_relu6=False
    ):
        """Add quantize down nodes."""
        quantized_outputs = [quantized_output_name, quantized_output_name + ":1", quantized_output_name + ":2"]
        if not self.fake_quant:
            # Add a RequantizationRange node for finding the min and max values.
            requant_range_node = helper.create_node(
                (
                    "RequantizationRangePerChannel"
                    if self.per_channel or original_node.op == "DepthwiseConv2dNative"
                    else "RequantizationRange"
                ),
                original_node.name + "_eightbit_requant_range",
                quantized_outputs,
            )

            if self.per_channel or original_node.op == "DepthwiseConv2dNative":
                helper.set_attr_dtype(requant_range_node, "T", dtypes.qint32)
                if is_relu6:
                    helper.set_attr_float(requant_range_node, "clip_value_max", 6.0)
                else:
                    helper.set_attr_float(requant_range_node, "clip_value_max", 1e30)
            else:
                helper.set_attr_dtype(requant_range_node, "Tinput", dtypes.qint32)

            self.add_output_graph_node(requant_range_node)
            min_max_inputs = [requant_range_node.name + ":0", requant_range_node.name + ":1"]
        else:
            max_input_name = original_node.name + "_max"
            max_node = helper.create_constant_node(max_input_name, 1.0, dtypes.float32)
            self.add_output_graph_node(max_node)

            min_input_name = original_node.name + "_min"
            min_node = helper.create_constant_node(min_input_name, -1.0, dtypes.float32)
            self.add_output_graph_node(min_node)

            min_max_inputs = [min_input_name, max_input_name]

        requantize_node = helper.create_node(
            "RequantizePerChannel" if self.per_channel or original_node.op == "DepthwiseConv2dNative" else "Requantize",
            original_node.name + "_eightbit_requantize",
            quantized_outputs + min_max_inputs,
        )
        if self.per_channel or original_node.op == "DepthwiseConv2dNative":
            helper.set_attr_dtype(requantize_node, "T", dtypes.qint32)
        else:
            helper.set_attr_dtype(requantize_node, "Tinput", dtypes.qint32)

        helper.set_attr_dtype(requantize_node, "out_type", requantize_type)
        self.add_output_graph_node(requantize_node)
        return requantize_node.name

    def _eightbitize_input_to_node(
        self, namespace_prefix, original_input_name, reshape_dims_name, reduction_dims_name, dtype=dtypes.quint8
    ):
        """Takes one float input to an op, and converts it to quantized form."""
        unique_input_name = helper.unique_node_name_from_input(original_input_name)
        if unique_input_name in self.quantized_node_dict:
            quantized_tuple = self.quantized_node_dict[unique_input_name]
            return quantized_tuple[0], quantized_tuple[1], quantized_tuple[2]

        if self.fake_quant:
            min_input_name = namespace_prefix + "_min_" + unique_input_name
            min_node = helper.create_constant_node(min_input_name, -1.0, dtypes.float32)
            self.add_output_graph_node(min_node)
            max_input_name = namespace_prefix + "_max_" + unique_input_name
            max_node = helper.create_constant_node(max_input_name, 1.0, dtypes.float32)
            self.add_output_graph_node(max_node)
            quantize_input_name = namespace_prefix + "_quantize_" + unique_input_name
            quantize_input_node = helper.create_node(
                "QuantizeV2", quantize_input_name, [original_input_name, min_input_name, max_input_name]
            )
        else:
            reshape_input_name = namespace_prefix + "_reshape_" + unique_input_name
            min_input_name = namespace_prefix + "_min_" + unique_input_name
            max_input_name = namespace_prefix + "_max_" + unique_input_name
            quantize_input_name = namespace_prefix + "_quantize_" + unique_input_name
            reshape_input_node = helper.create_node(
                "Reshape", reshape_input_name, [original_input_name, reshape_dims_name]
            )
            helper.set_attr_dtype(reshape_input_node, "T", dtypes.float32)
            self.add_output_graph_node(reshape_input_node)
            min_input_node = helper.create_node("Min", min_input_name, [reshape_input_name, reduction_dims_name])
            helper.set_attr_dtype(min_input_node, "T", dtypes.float32)
            helper.set_attr_dtype(min_input_node, "Tidx", dtypes.int32)
            helper.set_attr_bool(min_input_node, "keep_dims", False)
            self.add_output_graph_node(min_input_node)
            max_input_node = helper.create_node("Max", max_input_name, [reshape_input_name, reduction_dims_name])
            helper.set_attr_dtype(max_input_node, "T", dtypes.float32)
            helper.set_attr_dtype(max_input_node, "Tidx", dtypes.int32)
            helper.set_attr_bool(max_input_node, "keep_dims", False)
            self.add_output_graph_node(max_input_node)
            quantize_input_node = helper.create_node(
                "QuantizeV2", quantize_input_name, [original_input_name, min_input_name, max_input_name]
            )

        helper.set_attr_dtype(quantize_input_node, "T", dtype)
        helper.set_attr_string(quantize_input_node, "mode", b"MIN_FIRST" if self.is_asymmetric else b"SCALED")
        if not self.is_asymmetric:
            helper.set_attr_string(quantize_input_node, "round_mode", b"HALF_TO_EVEN")
        # if FLAGS.model_name in ["wide_deep_large_ds"]:
        #    set_attr_string(quantize_input_node, "mode", b"MIN_FIRST")
        # else:
        #    set_attr_string(quantize_input_node, "mode",
        #                    b"SCALED" if self.intel_cpu_eightbitize else b"MIN_FIRST")
        #    set_attr_string(quantize_input_node, "round_mode",
        #                    b"HALF_TO_EVEN" if self.intel_cpu_eightbitize
        #                    else b"HALF_AWAY_FROM_ZERO")
        self.add_output_graph_node(quantize_input_node)
        min_output_name = quantize_input_name + ":1"
        max_output_name = quantize_input_name + ":2"
        self.quantized_node_dict[unique_input_name] = (quantize_input_name, min_output_name, max_output_name)
        return quantize_input_name, min_output_name, max_output_name

    def _intel_cpu_quantize_weight_eightbit(self, parent, input_node, per_channel, enter_node=None):
        """Quantize weight node."""
        (
            qint8_const_node,
            min_node,
            max_node,
            qint8_const_enter_node,
            min_enter_node,
            max_enter_node,
        ) = helper.generate_quantized_weight_node(
            parent, input_node, per_channel, self.weight_bit, self.device, enter_node
        )
        if enter_node:  # pragma: no cover
            self.add_output_graph_node(qint8_const_node)
            self.add_output_graph_node(qint8_const_enter_node)
            self.add_output_graph_node(min_node)
            self.add_output_graph_node(min_enter_node)
            self.add_output_graph_node(max_node)
            self.add_output_graph_node(max_enter_node)

            return qint8_const_enter_node.name, min_enter_node.name, max_enter_node.name
        else:
            self.add_output_graph_node(qint8_const_node)
            self.add_output_graph_node(min_node)
            self.add_output_graph_node(max_node)

            return qint8_const_node.name, min_node.name, max_node.name
