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
"""ONNX Node wrapper for Tensorflow model converting to ONNX model."""

import copy
import logging

import numpy as np
from onnx import AttributeProto, TensorProto, helper, numpy_helper

from . import tf2onnx_utils as utils
from .onnx_schema import get_schema

logger = logging.getLogger("neural_compressor")


class OnnxNode:
    """A ONNX Node Wrapper used for graph manipulations."""

    def __init__(self, node, graph, skip_conversion=False):
        """Create ONNX Node.

        Args:
            node: Onnx node in NodeProto.
            graph: OnnxGraph.
            skip_conversion: bool for skipping node conversion.
        """
        self._op = node
        self.graph = graph
        self._input = list(node.input)
        self._output = list(node.output)
        self._attr = {}

        graph.set_node_by_name(self)
        # dict to original attributes
        for attr in node.attribute:
            self._attr[attr.name] = attr
        self._skip_conversion = skip_conversion

    @property
    def input(self):
        """Get node input."""
        return self._input

    @input.setter
    def input(self, val):
        """Node input setter.

        The setter can catch that all inputs are change but it cannot catch that one input is changed.
        That's method replace_input and replace_inputs must be used to change inputs to let
        the graph instance update its internal indices.
        """
        self._input = copy.deepcopy(val)

    @property
    def output(self):
        """Get node output."""
        return self._output

    @output.setter
    def output(self, val):
        """Set op output.

        Output should be updated explicitly,
        changing it would require output mapping changed.
        """
        self._graph_check()
        for each_output in self._output:
            del self.graph._output_to_node_name[each_output]

        self._output = val.copy()
        for each_output in self._output:
            self.graph._output_to_node_name[each_output] = self.name

    @property
    def inputs(self):
        """Input node objects."""
        self._graph_check()
        val = [self.graph.get_node_by_output(n) for n in self._input]
        return val

    @property
    def attr(self):
        """Get node attributes."""
        return self._attr

    def get_value_attr(self):
        """Return onnx attr for value property of node."""
        return self._attr["value"]

    def get_onnx_attrs(self):
        """Return onnx valid attributes."""
        schema = get_schema(self.type, self.graph.opset, self.domain)
        if schema is None and not (self.is_const() or self.is_graph_input()):
            logger.debug(
                "Node %s uses non-stardard onnx op <%s, %s>, skip attribute check", self.name, self.domain, self.type
            )
        onnx_attrs = {}
        for a in self._attr.values():
            if a.name == "value":
                onnx_attrs[a.name] = self.get_value_attr()
            elif schema is None or schema.has_attribute(a.name):
                onnx_attrs[a.name] = a
        return onnx_attrs

    @property
    def name(self):
        """Get op name."""
        return self._op.name

    def child_name(self):
        """Set child name."""
        return utils.set_name(self.name)

    @property
    def op(self):
        """Get node's op."""
        return self._op

    @property
    def type(self):
        """Return Op type."""
        return self._op.op_type

    @type.setter
    def type(self, val):
        """Set Op type."""
        self._op.op_type = val

    @property
    def domain(self):
        """Return Op domain."""
        return self._op.domain

    @domain.setter
    def domain(self, val):
        """Set Op domain."""
        self._op.domain = val

    @property
    def data_format(self):
        """Return data_format."""
        attr_str = self.get_attr_value("data_format")
        return "unknown" if attr_str is None else attr_str.decode("utf-8")

    @data_format.setter
    def data_format(self, val):
        """Set data_format."""
        self.set_attr("data_format", val)

    def is_nhwc(self):
        """Return True if node is in NHWC format."""
        utils.assert_error(
            "D" not in self.data_format,
            "is_nhwc called on %s with spatial=2 but data_format=%s",
            self.name,
            self.data_format,
        )
        return self.data_format == "NHWC"

    def is_const(self):
        """Return True if node is a constant."""
        return self.type in ["Const", "ConstV2"]

    def is_scalar(self):
        """Return True if node is a constant with a scalar value."""
        if not self.is_const():
            return False
        t = self.get_attr("value", default=None)
        if t is None:
            return False
        t = numpy_helper.to_array(helper.get_attribute_value(t))
        return t.shape == tuple()

    def is_graph_input(self):
        """Check if the node is the input of the graph."""
        return self.type in ["Placeholder", "PlaceholderWithDefault", "PlaceholderV2"]

    def is_graph_input_default_const(self):
        """Check if the node is the input of the graph and const."""
        return self.is_const() and any(out.is_graph_input() for out in self.graph.find_output_consumers(self.output[0]))

    def is_while(self):
        """Check if the node is while op."""
        return self.type in ["While", "StatelessWhile", "Loop"]

    def __str__(self):
        """Return string of the node op."""
        return str(self._op)

    def __repr__(self):
        """Return string of op type and name."""
        return "<onnx op type='%s' name=%s>" % (self.type, self._op.name)

    @property
    def summary(self):
        """Return node summary information."""
        lines = []
        lines.append("OP={}".format(self.type))
        lines.append("Name={}".format(self.name))

        g = self.graph
        if self.input:
            lines.append("Inputs:")
            for name in self.input:
                node = g.get_node_by_output(name)
                op = node.type if node else "N/A"
                lines.append("\t{}={}, {}, {}".format(name, op, g.get_shape(name), g.get_dtype(name)))

        if self.output:
            for name in self.output:
                lines.append("Outputs:")
                lines.append("\t{}={}, {}".format(name, g.get_shape(name), g.get_dtype(name)))

        return "\n".join(lines)

    def get_attr(self, name, default=None):
        """Get raw attribute value."""
        attr = self.attr.get(name, default)
        return attr

    def get_attr_value(self, name, default=None):
        """Get attribute value."""
        attr = self.get_attr(name)
        if attr:
            return helper.get_attribute_value(attr)
        return default

    def get_attr_int(self, name):
        """Get attribute value as int."""
        attr_int = self.get_attr_value(name)
        utils.assert_error(attr_int is not None and isinstance(attr_int, int), "attribute %s is None", name)
        return attr_int

    def get_attr_str(self, name, encoding="utf-8"):
        """Get attribute value as string."""
        attr_str = self.get_attr_value(name)
        utils.assert_error(attr_str is not None and isinstance(attr_str, bytes), "attribute %s is None", name)
        return attr_str.decode(encoding)

    def set_attr(self, name, value):
        """Set node's attribute."""
        self.attr[name] = helper.make_attribute(name, value)

    def set_attr_onnx(self, value):
        """Set node's onnx attributes."""
        self.attr[value.name] = value

    @property
    def skip_conversion(self):
        """Get skip conversion setting."""
        return self._skip_conversion

    @skip_conversion.setter
    def skip_conversion(self, val):
        """Set skip conversion."""
        self._skip_conversion = val

    # If some Node is created as onnx_node, then we don't need convert it
    def need_skip(self):
        """Check if need to skip conversion."""
        return self._skip_conversion

    @property
    def output_shapes(self):
        """Get output shapes."""
        self._graph_check()
        val = [self.graph.get_shape(n) for n in self._output]
        return val

    @property
    def output_dtypes(self):
        """Get output dtypes."""
        self._graph_check()
        val = [self.graph.get_dtype(n) for n in self._output]
        return val

    def get_tensor_value(self, as_list=True):
        """Get value for onnx tensor.

        Args:
            as_list: whether return numpy ndarray in list.

        Returns:
            If as_list=True, return the array as a (possibly nested) list.
            Otherwise, return data of type np.ndarray.

            If a tensor is a scalar having value 1,
                when as_list=False, return np.array(1), type is <class 'numpy.ndarray'>
                when as_list=True, return 1, type is <class 'int'>.
        """
        if not self.is_const():
            raise ValueError("get tensor value: '{}' must be Const".format(self.name))

        t = self.get_attr("value")
        if t:
            t = numpy_helper.to_array(helper.get_attribute_value(t))
            if as_list is True:
                t = t.tolist()  # t might be scalar after tolist()
        return t

    def scalar_to_dim1(self):
        """Get value for onnx tensor."""
        if not self.is_const():
            raise ValueError("get tensor value: {} must be Const".format(self.name))

        t = self.get_attr("value")
        if t:
            t = helper.get_attribute_value(t)
            if not t.dims:
                t.dims.extend([1])
        return t.dims

    def set_tensor_value(self, new_val):
        """Set new value for existing onnx tensor.

        Args:
            new_val: value of type numpy ndarray
        """
        if not self.is_const():
            raise ValueError("set tensor value: {} must be Const".format(self.name))
        t = self.get_attr("value")
        if not t:
            raise ValueError("set tensor value: {} is None".format(self.name))
        t = helper.get_attribute_value(t)
        onnx_tensor = numpy_helper.from_array(new_val, t.name)
        del t
        self.set_attr("value", onnx_tensor)
        # track shapes in _output_shapes
        self._graph_check()
        self.graph.set_shape(onnx_tensor.name, list(onnx_tensor.dims))

    def get_body_graphs(self):
        """Get body graphs."""
        self._graph_check()
        return self.graph.contained_graphs.get(self.name, None)

    def set_body_graph_as_attr(self, attr_name, graph):
        """Set body graphs as attribute."""
        self._graph_check()
        if self.name not in self.graph.contained_graphs:
            self.graph.contained_graphs[self.name] = {}

        self.graph.contained_graphs[self.name].update({attr_name: graph})
        graph.parent_graph = self.graph

    def update_proto(self):
        """Update protobuf from internal structure."""
        nodes = list(self._op.input)
        for node in nodes:
            self._op.input.remove(node)
        self._op.input.extend(self.input)
        nodes = list(self._op.output)
        for node in nodes:
            self._op.output.remove(node)
        self._op.output.extend(self.output)

        # update attributes to proto
        del self._op.attribute[:]

        # check attribute of type GraphProto
        attr_graphs = self.get_body_graphs()
        if attr_graphs:
            for attr_name, sub_graph in attr_graphs.items():
                graph_proto = sub_graph.make_graph("graph for " + self.name + " " + attr_name)
                self.set_attr(attr_name, graph_proto)

        attr = list(self.get_onnx_attrs().values())
        if attr:
            self._op.attribute.extend(attr)

    def get_implicit_inputs(self, recursive=True):
        """Get implicit inputs if the node has attributes being GraphProto."""
        output_available_in_cur_graph = set()
        all_node_inputs = set()

        graphs = []
        body_graphs = self.get_body_graphs()
        if body_graphs:
            graphs.extend(body_graphs.values())

        while graphs:
            graph = graphs.pop()
            for n in graph.get_nodes():
                output_available_in_cur_graph |= set(n.output)
                for i in n.input:
                    all_node_inputs.add(i)

                if recursive:
                    b_graphs = n.get_body_graphs()
                    if b_graphs:
                        graphs.extend(b_graphs.values())

        outer_scope_node_input_ids = all_node_inputs - output_available_in_cur_graph
        return list(outer_scope_node_input_ids)

    def _graph_check(self):
        """Check the graph is None."""
        utils.assert_error(self.graph is not None, "Node %s not belonging any graph", self.name)
