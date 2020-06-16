#  -*- coding: utf-8 -*-
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

import re


class QuantizeGraphHelper(object):
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

    def get_sorted_graph(self, input_graph, output_node_names):
        self.node_name_mapping = {}
        self.op_list = []
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
    def split_shared_inputs(input_graph_def, ops=[]):
        """
        Split shared inputs(like weights and bias) of ops list.
        :param in_graph: input graph file.
        :param ops: ops list to processing.
        :return: path to ouput graph file.
        """
        if not ops:
            return input_graph_def
        node_map = {}
        for node in input_graph_def.node:
            if node.name not in node_map.keys():
                node_map[node.name] = node

        output_graph_def = graph_pb2.GraphDef()
        is_shared_input = False
        # map of: input_name - op_name
        input_map = {}
        for node_name in node_map.keys():
            node = node_map[node_name]
            if node.op in ops:
                for input_idx, input_node_name in enumerate(node.input):
                    if node_map[QuantizeGraphHelper.node_name_from_input(
                            input_node_name)].op == 'Const':
                        # is shared and current node is not the first one sharing the input
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
    def create_node(op, name, inputs):
        new_node = node_def_pb2.NodeDef()
        new_node.op = op
        new_node.name = name
        for input_name in inputs:
            new_node.input.extend([input_name])
        return new_node

    @staticmethod
    def create_constant_node(name, value, dtype, shape=None):
        node = QuantizeGraphHelper.create_node("Const", name, [])
        QuantizeGraphHelper.set_attr_dtype(node, "dtype", dtype)
        QuantizeGraphHelper.set_attr_tensor(node, "value", value, dtype, shape)
        return node

    @staticmethod
    def copy_attr(node, key, attr_value):
        node.attr[key].CopyFrom(attr_value)

    @staticmethod
    def set_attr_dtype(node, key, value):
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(type=value.as_datatype_enum))

    @staticmethod
    def set_attr_shape(node, key, value):
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(
                shape=tensor_shape.as_shape(value).as_proto()))

    @staticmethod
    def set_attr_tensor(node, key, value, dtype, shape=None):
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                value, dtype=dtype, shape=shape)))

    @staticmethod
    def set_attr_string(node, key, value):
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(s=value))

    @staticmethod
    def set_attr_int_list(node, key, value):
        list_value = attr_value_pb2.AttrValue.ListValue(i=value)
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=list_value))

    @staticmethod
    def set_attr_bool(node, key, value):
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(b=value))

    @staticmethod
    def set_attr_int(node, key, value):
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(i=value))

    @staticmethod
    def set_attr_float(node, key, value):
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(f=value))

    @staticmethod
    def node_name_from_input(node_name):
        if node_name not in QuantizeGraphHelper.node_name_cache:
            key = node_name
            if node_name.startswith("^"):
                node_name = node_name[1:]
            m = re.search(r"(.*):\d+$", node_name)
            if m:
                node_name = m.group(1)
            QuantizeGraphHelper.node_name_cache[key] = node_name
            return node_name
        else:
            return QuantizeGraphHelper.node_name_cache[node_name]

    @staticmethod
    def unique_node_name_from_input(node_name):
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
        else:
            return QuantizeGraphHelper.node_name_port_cache[node_name]
