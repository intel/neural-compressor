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

import re
import numpy as np
import tensorflow as tf

from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

class QuantizeGraphHelper():
    """
    This class contains several staticmethod functions.
    """
    node_name_cache = {}
    node_name_port_cache = {}

    def __init__(self):
        pass

    def _recursive_graph_sorting(self, node_name):
        if node_name in self.op_list or not self.node_name_mapping[
                node_name].input:
            return

        for input_name in self.node_name_mapping[node_name].input:
            if input_name not in self.node_name_mapping:
                continue
            else:
                self._recursive_graph_sorting((input_name))

        if node_name not in self.op_list:
            self.op_list.append(node_name)

        return

    def _get_op_list(self, output_node_names):
        for output_name in output_node_names:
            self._recursive_graph_sorting(output_name)

    def get_sorted_graph(self, input_graph, input_node_names, output_node_names):
        """Return a sorted graphdef object.Sometimes the input graphdef was composed of
        the randome nodedef objects, we reorder the graph to make the parsing more easier.
        Args:
            input_graph (graphdef]): the input graphdef object
            input_node_names (string list): the input node names
            output_node_names (string list): the output node names

        Returns:
            [type]: [description]
        """
        self.node_name_mapping = {}
        self.op_list = [input_node_name for input_node_name in input_node_names]
        for node in input_graph.node:
            self.node_name_mapping[node.name] = node
        self._get_op_list(output_node_names)

        self.op_list.extend(
            set(self.node_name_mapping.keys()) - set(self.op_list))
        self.out_graph_def = graph_pb2.GraphDef()
        for i in self.op_list:
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(self.node_name_mapping[i])
            self.out_graph_def.node.extend([new_node])

        return self.out_graph_def

    @staticmethod
    def split_shared_inputs(input_graph_def):
        """
        Split shared inputs(like weights and bias) of the graph.
        :param in_graph: input graph file.
        :return: path to ouput graph file.
        """

        node_map = {}
        for node in input_graph_def.node:
            if node.name not in node_map:
                node_map[node.name] = node

        output_graph_def = graph_pb2.GraphDef()
        is_shared_input = False
        # map of: input_name - op_name
        input_map = {}
        for node_name in node_map.keys():
            node = node_map[node_name]
            for input_idx, input_node_name in enumerate(node.input):
                if node_map[QuantizeGraphHelper.node_name_from_input(
                        input_node_name)].op == 'Const':
                    # is shared and current node is not the first one
                    # sharing the input
                    if input_node_name in input_map.keys():
                        is_shared_input = True
                        input_map[input_node_name].append(node.name)
                        new_input_node = node_def_pb2.NodeDef()
                        new_input_node.CopyFrom(node_map[input_node_name])
                        new_input_node.name = input_node_name + '_' + str(
                            len(input_map[input_node_name]))
                        node.input[input_idx] = new_input_node.name
                        output_graph_def.node.extend([new_input_node])
                    else:
                        input_map[input_node_name] = [node.name]
            output_graph_def.node.extend([node])

        return output_graph_def if is_shared_input else input_graph_def

    @staticmethod
    def remove_training_nodes(input_graph, protected_nodes=[],
                              types_to_splice=['Identity', 'CheckNumerics']):
        """Prunes out nodes that aren't needed for inference.
        Args:
            input_graph: Model to analyze and prune.
            types_to_splice: An optional list of types of nodes to be removed
            unconditionally.

        Returns:
            A optimized graphdef object.
        """
        input_nodes = input_graph.node

        control_input_names = set()
        node_names_with_control_input = set()
        for node in input_nodes:
            for node_input in node.input:
                if "^" in node_input:
                    control_input_names.add(node_input.replace("^", ""))
                    node_names_with_control_input.add(node.name)

        names_to_splice = {}
        for node in input_nodes:
            if node.op in types_to_splice:
                # We don't want to remove nodes that have control edge inputs, because
                # they might be involved in subtle dependency issues that removing them
                # will jeopardize.
                if node.name not in node_names_with_control_input:
                    names_to_splice[node.name] = node.input[0]

        # We also don't want to remove nodes which are used as control edge inputs.
        names_to_splice = {
            name: value
            for name, value in names_to_splice.items()
            if name not in control_input_names
        }

        nodes_after_splicing = []

        for node in input_nodes:
            if node.name in names_to_splice and node.name not in protected_nodes:
                continue

            if node.name in protected_nodes and node.name in types_to_splice:
                nodes_after_splicing.append(node)
                continue

            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(node)
            input_before_removal = node.input
            del new_node.input[:]
            for full_input_name in input_before_removal:
                input_name = re.sub(r"^\^", "", full_input_name)
                while input_name in names_to_splice:
                    full_input_name = names_to_splice[input_name]
                    input_name = re.sub(r"^\^", "", full_input_name)
                new_node.input.append(full_input_name)
            nodes_after_splicing.append(new_node)

        output_graph = graph_pb2.GraphDef()
        output_graph.node.extend(nodes_after_splicing)
        return output_graph

    @staticmethod
    def create_node(op, name, inputs):
        """Create a nodedef object

        Args:
            op (string): op type
            name (string): op name
            inputs (string list): op's inputs name

        Returns:
            nodedef: the created nodedef object
        """
        new_node = node_def_pb2.NodeDef()
        new_node.op = op
        new_node.name = name
        for input_name in inputs:
            new_node.input.extend([input_name])
        return new_node

    @staticmethod
    def create_constant_node(name, value, dtype, shape=None, device='cpu'):
        """create constant node.

        Args:
            name (string): op name
            value (np.array): input data
            dtype (datatype): data type of the input value
            shape (int list, optional): the value's shape. Defaults to None.
            device (str, optional): the device type, it may be the 'cpu' or 'gpu'.
                                    Defaults to 'cpu'.

        Returns:
            [type]: [description]
        """
        node = QuantizeGraphHelper.create_node("Const" if device == 'cpu' else "HostConst", name,
                                                 [])
        QuantizeGraphHelper.set_attr_dtype(node, "dtype", dtype)
        QuantizeGraphHelper.set_attr_tensor(node, "value", value, dtype, shape)
        return node

    @staticmethod
    def copy_attr(node, key, attr_value):
        """Copy the specified attr value to node.

        Args:
            node (nodedef): a nodedef object
            key (string): string name
            attr_value (any): the specified attribute value
        """
        node.attr[key].CopyFrom(attr_value)

    @staticmethod
    def set_attr_dtype(node, key, value):
        """Set the attribute data type
        """
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(type=value.as_datatype_enum))

    @staticmethod
    def set_attr_tensor(node, key, value, dtype, shape=None):
        """Set the tensor value to specified attribute field.

        Args:
            node (nodedef): the target nodedef object
            key (string): attribute name
            value (np.array): the content
            dtype (dtypes): data type
            shape (int list, optional): the input tensor's shape. Defaults to None.
        """
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(value, dtype=dtype, shape=shape)))

    @staticmethod
    def set_attr_string(node, key, value):
        """Set the node's attr which data type is string.
        """
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(s=value))

    @staticmethod
    def set_attr_bool(node, key, value):
        """Set the node's attr which data type is bool.
        """
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(b=value))

    @staticmethod
    def set_attr_int(node, key, value):
        """Set the node's attr which data type is int.
        """
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(i=value))

    @staticmethod
    def set_attr_float(node, key, value):
        """Set the node's attr which data type is float.
        """
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(f=value))

    @staticmethod
    def node_name_from_input(node_name):
        """Static method that get the valid node name from input name.

        Args:
            node_name (string): node name defined in the input field.

        Returns:
            string: node's name
        """
        if node_name not in QuantizeGraphHelper.node_name_cache:
            key = node_name
            if node_name.startswith("^"):
                node_name = node_name[1:]
            m = re.search(r"(.*):\d+$", node_name)
            if m:
                node_name = m.group(1)
            QuantizeGraphHelper.node_name_cache[key] = node_name
            return node_name

        return QuantizeGraphHelper.node_name_cache[node_name]

    @staticmethod
    def unique_node_name_from_input(node_name):
        """Get the node name from other node name's input field.
        """
        return node_name.replace(":", "__port__").replace("^", "__hat__")

    @staticmethod
    def ensure_tensor_name_has_port(node_name):
        """Makes sure that a tensor name has :0 if no explicit port exists."""
        if node_name not in QuantizeGraphHelper.node_name_port_cache:
            key = node_name
            m = re.search(r"(.*):\d+$", node_name)
            if not m:
                node_name = node_name + ":0"
            QuantizeGraphHelper.node_name_port_cache[key] = node_name
            return node_name

        return QuantizeGraphHelper.node_name_port_cache[node_name]

    @staticmethod
    def generate_quantized_weight_node(host_op_type,
                                            input_node,
                                            per_channel,
                                            quantization_mode=b"SCALED",
                                            weight_bit=7.0,
                                            device='cpu'):
        base_name = input_node.name + "_"
        qint8_const_name = base_name + "qint8_const"
        min_name = base_name + "min"
        max_name = base_name + "max"
        float_tensor = tensor_util.MakeNdarray(input_node.attr["value"].tensor)
        epsilon = 1e-4  # Needs to be set empirically if accuracy is not satisfactory
        range_coefficent = 127 / (2 ** weight_bit - 1)
        if host_op_type in ("Conv2D", "MatMul"):
            if per_channel:
                ranges = np.abs(float_tensor).max(axis=(0, 1, 2))
                ranges *= range_coefficent
                min_value = -ranges
                max_value = ranges
                # nudging min-max values outside epsilon radius around zero
                ranges[ranges < epsilon] = epsilon
                min_value[np.abs(min_value) < epsilon] = -epsilon
                max_value[np.abs(max_value) < epsilon] = epsilon
                qint8_tensor = (np.around(float_tensor *127.0/ranges)).astype(np.int8)
            else:
                min_value = np.min(float_tensor.flatten())
                max_value = np.max(float_tensor.flatten())
                min_value *= range_coefficent
                max_value *= range_coefficent
                # Same processing of min-max as in quantize_weight_eightbit
                # function.
                min_value = min(min_value, 0.0)
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
                        mode=quantization_mode,
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
        elif host_op_type == "DepthwiseConv2dNative":
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
        shape = tensor_util.TensorShapeProtoToList(input_node.attr["value"].tensor.tensor_shape)
        qint8_const_node = QuantizeGraphHelper.create_constant_node(qint8_const_name,
                                                                    qint8_tensor,
                                                                    dtypes.qint8,
                                                                    shape=shape)

        min_node = QuantizeGraphHelper.create_constant_node(min_name, min_value,
                                                            dtypes.float32, device=device)

        max_node = QuantizeGraphHelper.create_constant_node(max_name, max_value,
                                                            dtypes.float32, device=device)

        return qint8_const_node, min_node, max_node
