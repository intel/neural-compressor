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
"""ONNX Graph wrapper for Tensorflow model converting to ONNX model."""

import collections
import logging
import re

import numpy as np
import six
from onnx import AttributeProto, TensorProto, helper, numpy_helper

from . import tf2onnx_utils as utils
from .onnx_node import OnnxNode

logger = logging.getLogger("neural_compressor")


class OnnxGraph:
    """Class that provides graph manipulation and matching."""

    def __init__(
        self,
        nodes,
        output_shapes=None,
        dtypes=None,
        target=None,
        opset=None,
        extra_opset=None,
        input_names=None,
        output_names=None,
        is_subgraph=False,
        graph_name=None,
    ):
        """Create ONNX Graph.

        Args:
            nodes: list of Node()
            output_shapes: dict of tensorflow output shapes
            dtypes: dict of tensorflow dtype
            target: list of workarounds applied to help certain platforms
            opset: the opset to be used (int, default is latest)
            extra_opset: list of extra opset's, for example the opset's used by custom ops
            input_names: list of input node names in graph, input name format as node_name:port_id. Optional.
            output_names: list of output node names in graph, format is node_name:port_id. Optional.
            is_subgraph: bool, check subgraph.
            graph_name: str, graph name.
        """
        if target is None:
            target = []
        self._nodes = []
        self._nodes_by_name = {}
        self._output_to_node_name = {}
        self._output_to_consumers = {}
        self._input_to_graph = {}
        self.shapes = {}
        self.graph_name = graph_name or utils.set_name("tfqdq_2_onnxqdq")
        self._is_subgraph = is_subgraph
        self.ta_reads = []
        # A list of index, output tuples of potential scan outputs in this graph
        # Used by the tflite while loop handler
        self.scan_outputs = []
        # Used by lstm_tf2_rewriter to indicate this subgraph is an LSTM cell
        self.lstm_rewriter_context = None
        self.gru_rewriter_context = None
        self.func_inputs = []
        self.ragged_variant_list_reads = []
        self.ragged_variant_list_writes = []

        self._dtypes = dtypes
        self._output_shapes = output_shapes

        self.set_config(target, opset, extra_opset)

        self.inputs = []
        self.outputs = output_names if output_names is not None else []

        self.parent_graph = None
        self.contained_graphs = {}  # {node_name: {node_attribute_name: Graph}}

        ops = [OnnxNode(node, self) for node in nodes]
        if input_names is not None:
            input_names_set = set(input_names)
            for n in ops:
                for i, out in enumerate(n.output):
                    if out in input_names_set and not n.is_graph_input():
                        n.output[i] = utils.set_name("@@ALLOC")
                        ops.append(OnnxNode(helper.make_node("Placeholder", [], outputs=[out], name=out), self))
                        logger.info("Created placeholder for input %s", out)

        input_nodes = {n.output[0]: n for n in ops if n.is_graph_input()}
        if input_names is not None:
            self.inputs = [input_nodes[n] for n in input_names]
        else:
            self.inputs = list(input_nodes.values())

        self.reset_nodes(ops)

        # add identity node after each output, in case it is renamed during conversion.
        for o in self.outputs:
            n = self.get_node_by_output_in_current_graph(o)
            if n.is_graph_input():
                # Don't add identity if the node is also an input. We want to keep input names the same.
                continue
            new_output_name = utils.add_port_to_name(n.name + "_" + utils.set_name("raw_output_"))
            n_shapes = n.output_shapes
            n_dtypes = n.output_dtypes
            o_shape = self.get_shape(o)
            o_dtype = self.get_dtype(o)
            body_graphs = n.graph.contained_graphs.pop(n.name, None)
            self.remove_node(n.name)

            new_outputs = [output if output != o else new_output_name for output in n.output]
            # domain should be passed to new node
            branches = {}
            if body_graphs:
                for attr_name, body_graph in body_graphs.items():
                    body_graph.parent_graph = self
                    branches[attr_name] = body_graph

            _ = self.make_node(
                n.type,
                n.input,
                outputs=new_outputs,
                attr=n.attr,
                name=n.name,
                skip_conversion=n._skip_conversion,
                dtypes=n_dtypes,
                shapes=n_shapes,
                domain=n.domain,
                branches=branches,
            )

            self.replace_all_inputs(o, new_output_name, ops=self.get_nodes())
            self.make_node(
                "Identity",
                [new_output_name],
                outputs=[o],
                op_name_scope=n.name + "_" + "graph_outputs",
                dtypes=[o_dtype],
                shapes=[o_shape],
            )
            self.copy_shape(new_output_name, o)
            self.copy_dtype(new_output_name, o)

    def set_config(self, target=None, opset=None, extra_opset=None):
        """Set graph fields containing conversion options."""
        if target is None:
            target = utils.DEFAULT_TARGET

        self._opset = utils.find_opset(opset)
        self._target = set(target)

        if extra_opset is not None:
            utils.assert_error(isinstance(extra_opset, list), "invalid extra_opset")
        self._extra_opset = extra_opset

    @property
    def input_names(self):
        """Placeholder node outputs."""
        return [node.output[0] for node in self.inputs]

    @property
    def opset(self):
        """Get opset."""
        return self._opset

    @property
    def extra_opset(self):
        """Get extra opset."""
        return self._extra_opset

    def is_target(self, *names):
        """Return True if target platform contains any name."""
        return any(name in self._target for name in names)

    def make_const(self, name, np_val, skip_conversion=False, raw=True):
        """Make a new constant node in the graph.

        Args:
            name: const node name, must be unique.
            np_val: value of type numpy ndarray.
            skip_conversion: bool, indicate whether this created node would be mapped during conversion.
            raw: whether to store data at field of raw_data or the specific field according to its dtype
        """
        np_val_flat = np_val.flatten()
        is_bytes = np_val.dtype == object and len(np_val_flat) > 0 and isinstance(np_val_flat[0], bytes)
        if raw and not is_bytes:
            onnx_tensor = numpy_helper.from_array(np_val, name)
        else:
            onnx_tensor = helper.make_tensor(
                name, utils.map_numpy_to_onnx_dtype(np_val.dtype), np_val.shape, np_val_flat, raw=False
            )
        dtype = onnx_tensor.data_type
        node = self.make_node(
            "Const",
            [],
            outputs=[name],
            name=name,
            attr={"value": onnx_tensor},
            skip_conversion=skip_conversion,
            dtypes=[dtype],
            infer_shape_dtype=False,
        )
        self.set_shape(name, np_val.shape)
        self.set_dtype(name, utils.map_numpy_to_onnx_dtype(np_val.dtype))
        return node

    def make_node(
        self,
        op_type,
        inputs,
        attr=None,
        output_count=1,
        outputs=None,
        skip_conversion=True,
        op_name_scope=None,
        name=None,
        shapes=None,
        dtypes=None,
        domain=utils.ONNX_DOMAIN,
        infer_shape_dtype=True,
        branches=None,
    ):
        """Make a new onnx node in the graph."""
        if attr is None:
            attr = {}
        if shapes is None:
            shapes = []
        if dtypes is None:
            dtypes = []
        if branches is None:
            branches = {}
        if name is None:
            name = utils.set_name(op_type)

        if op_name_scope:
            name = "_".join([op_name_scope, name])

        logger.debug("Making node: Name=%s, OP=%s", name, op_type)

        if outputs is None:
            outputs = [name + ":" + str(i) for i in range(output_count)]

        output_count = len(outputs)
        raw_attr = {}
        onnx_attrs = []
        for a, v in attr.items():
            if isinstance(v, AttributeProto):
                onnx_attrs.append(v)
            else:
                raw_attr[a] = v

        n = self.get_node_by_name(name)
        utils.assert_error(n is None, "name %s already exists in node: \n%s", name, n)
        for o in outputs:
            n = self.get_node_by_output_in_current_graph(o)
            utils.assert_error(n is None, "output tensor named %s already exists in node: \n%s", o, n)

        onnx_node = helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **raw_attr)

        for name2 in onnx_node.input:
            self._register_input_name(name2, onnx_node)

        if op_type in ["If", "Loop", "Scan"]:
            # we force the op containing inner graphs not skipped during conversion.
            skip_conversion = False

        node = OnnxNode(onnx_node, self, skip_conversion=skip_conversion)
        if onnx_attrs:
            _ = [node.set_attr_onnx(a) for a in onnx_attrs]

        for branch, body in branches.items():
            node.set_body_graph_as_attr(branch, body)

        if shapes:
            utils.assert_error(
                len(shapes) == output_count,
                "output shape count %s not equal to output count %s",
                len(shapes),
                output_count,
            )
            for i in range(output_count):
                self.set_shape(node.output[i], shapes[i])

        if dtypes:
            utils.assert_error(
                len(dtypes) == output_count,
                "output dtypes count %s not equal to output count %s",
                len(dtypes),
                output_count,
            )
            for i in range(output_count):
                self.set_dtype(node.output[i], dtypes[i])

        if (not shapes or not dtypes) and infer_shape_dtype:
            self.update_node_shape_dtype(node, override=False)

        logger.debug("Made node: %s\n%s", node.name, node.summary)
        self._nodes.append(node)
        return node

    def append_node(self, node):
        """Add a node to the graph."""
        output_shapes = node.output_shapes
        output_dtypes = node.output_dtypes
        node.graph = self
        self._nodes.append(node)
        self._nodes_by_name[node.name] = node
        for i, name in enumerate(node.output):
            self._output_to_node_name[name] = node.name
            self.set_dtype(name, output_dtypes[i])
            self.set_shape(name, output_shapes[i])
        for name in node.input:
            self._register_input_name(name, node)

    def remove_node(self, node_name):
        """Remove node in current graph."""
        utils.assert_error(node_name in self._nodes_by_name, "node %s not in current graph, cannot remove", node_name)
        node = self.get_node_by_name(node_name)
        del self._nodes_by_name[node_name]
        if node_name in self.contained_graphs:
            del self.contained_graphs[node_name]

        if node in self.inputs:
            self.inputs.remove(node)

        for op_output in node.output:
            if op_output == "":
                continue
            del self._output_to_node_name[op_output]

            if op_output in self._output_shapes:
                del self._output_shapes[op_output]
            if op_output in self._dtypes:
                del self._dtypes[op_output]

        for op_input in node.input:
            if op_input == "":
                continue
            utils.assert_error(
                op_input in self._output_to_consumers, "Input %r of node %r not found.", op_input, node_name
            )
            self._unregister_input_name(op_input, node)

        self._nodes.remove(node)
        node.graph = None

    def safe_remove_nodes(self, to_delete):
        """Delete nodes in `to_delete` without third-party node consuming it."""
        delete_set = set(to_delete)
        for n in delete_set:
            out_consumers = set()
            for out in n.output:
                out_consumers |= set(self.find_output_consumers(out))
            if out_consumers.issubset(delete_set):
                self.remove_node(n.name)

    def reset_nodes(self, ops):
        """Reset the graph with node list."""
        remained_dtypes = {}
        remained_shapes = {}
        remained_sub_graphs = {}
        for op in ops:
            for op_output in op.output:
                # this check should be removed once we make sure all output tensors have dtype/shape.
                if op_output in self._dtypes:
                    remained_dtypes[op_output] = self._dtypes[op_output]
                if op_output in self._output_shapes:
                    remained_shapes[op_output] = self._output_shapes[op_output]

            if op.name in self.contained_graphs:
                remained_sub_graphs[op.name] = self.contained_graphs[op.name]

        self._nodes = ops
        self.contained_graphs = remained_sub_graphs
        self._nodes_by_name = {op.name: op for op in ops}
        self._output_to_node_name = {}
        self._output_to_consumers = {}
        for op in ops:
            for op_output in op.output:
                self._output_to_node_name[op_output] = op.name
            inps = op.input
            for op_input in inps:
                self._register_input_name(op_input, op)

        for n in self.inputs:
            if n not in ops:
                raise ValueError("graph input '" + n.name + "' not exist")

        for o in self.outputs:
            if o not in self._output_to_node_name:
                raise ValueError("graph output '" + o.name + "' not exist")

        self._dtypes = remained_dtypes
        self._output_shapes = remained_shapes

    def create_new_graph_with_same_config(self):
        """Create a clean graph inheriting current graph's configuration."""
        return OnnxGraph(
            [],
            output_shapes={},
            dtypes={},
            target=self._target,
            opset=self._opset,
            extra_opset=self.extra_opset,
            output_names=[],
        )

    def is_empty_input(self, name):
        """Check if the input is empty.

        in ONNX, operation may have optional input and an empty string may be used
        in the place of an actual argument's name to indicate a missing argument.
        """
        return name == utils.ONNX_EMPTY_INPUT

    def update_node_shape_dtype(self, node, override=False):
        """Try the best to infer shapes and dtypes for outputs of the node."""
        if node.is_const() or node.is_graph_input():
            return
        # NOTE: only support onnx node for now
        if not utils.is_onnx_domain(node.domain):
            return

        logger.debug("Infer shape and dtype for [%s]", node.name)
        # NOTE: shape inference for some ops need the input values of the op, e.g., Reshape
        # op needs the "Shape" value to infer output shape.
        initializers = []
        for i, inp in enumerate(node.inputs):
            if inp is None:
                if not self.is_empty_input(node.input[i]):
                    if logger.isEnabledFor(logging.INFO):
                        logger.warning(
                            "[%s] infer a inexistent node: [%s], please check the code", node.name, node.input[i]
                        )
                continue
            if inp.is_const():
                t = inp.get_attr("value")
                tensor = helper.get_attribute_value(t)
                tensor.name = inp.output[0]
                initializers.append(tensor)

        input_shapes = [self.get_shape(i) for i in node.input]
        input_dtypes = [self.get_dtype(i) for i in node.input]

        shapes, dtypes = utils.infer_onnx_shape_dtype(node, self._opset, input_shapes, input_dtypes, initializers)
        if not shapes or not dtypes:
            return

        for output, shape, dtype in zip(node.output, shapes, dtypes):
            if dtype == TensorProto.UNDEFINED:
                logger.debug("Inferred dtype for [%s, type: %s] is UNDEFINED, SKIP", node.name, node.type)
            else:
                existing_dtype = self.get_dtype(output)
                if existing_dtype is not None and existing_dtype != dtype and not override:
                    dtype = existing_dtype
                self.set_dtype(output, dtype)
                logger.debug("Set dtype of [%s] to %s", output, dtype)

            if shape is None:
                logger.debug("Inferred shape for [%s, type: %s] is None, SKIP", node.name, node.type)
            else:
                existing_shape = self.get_shape(output)
                if existing_shape is not None and not utils.are_shapes_equal(existing_shape, shape) and not override:
                    shape = existing_shape
                self.set_shape(output, shape)
                logger.debug("Set shape of [%s] to %s", output, shape)

    def update_proto(self):
        """Update the onnx protobuf from out internal Node structure."""
        for node in self._nodes:
            node.update_proto()

    def get_nodes(self):
        """Get node list."""
        return self._nodes

    def get_node_by_output(self, output, search_in_parent_graphs=True):
        """Get node by node output id recursively going through nested graphs.

        Args:
            output: node's output
            search_in_parent_graphs: search in all parent graphs
        """
        ret = None
        g = self
        while not ret and g:
            ret = g.get_node_by_output_in_current_graph(output)
            if ret:
                return ret

            if not search_in_parent_graphs:
                break
            g = g.parent_graph
        return ret

    def get_node_by_output_in_current_graph(self, output):
        """Get node by node output id."""
        name = self._output_to_node_name.get(output)
        ret = None
        if name:
            ret = self._nodes_by_name.get(name)
        return ret

    def get_node_by_name(self, name):
        """Get node by name."""
        ret = self._nodes_by_name.get(name)
        return ret

    def set_node_by_name(self, node):
        """Set node by name."""
        self._nodes_by_name[node.name] = node
        for op_output in node.output:
            self._output_to_node_name[op_output] = node.name
        for name in node.input:
            self._register_input_name(name, node)

    def is_const(self, output):
        """Check if the node is const."""
        return self.get_node_by_output(output).is_const()

    def get_tensor_value(self, output, as_list=True):
        """Get the tensor value of the node."""
        return self.get_node_by_output(output).get_tensor_value(as_list)

    def get_dtype(self, name):
        """Get dtype for node."""
        node = self.get_node_by_output(name, search_in_parent_graphs=True)
        return node.graph._dtypes.get(name) if node else None

    def set_dtype(self, name, dtype):
        """Set dtype for node."""
        node = self.get_node_by_output(name, search_in_parent_graphs=True)
        node.graph._dtypes[name] = dtype

    def copy_dtype(self, src_name, dst_name):
        """Copy dtype from another node."""
        dtype = self.get_dtype(src_name)
        self.set_dtype(dst_name, dtype)

    def get_shape(self, name):
        """Get shape for node."""
        utils.assert_error(isinstance(name, six.text_type), "get_shape name is invalid type: %s", name)
        node = self.get_node_by_output(name, search_in_parent_graphs=True)
        shape = node.graph._output_shapes.get(name) if node else None
        if shape:
            for i, v in enumerate(shape):
                if v is None:
                    # pylint: disable=unsupported-assignment-operation
                    shape[i] = -1
            # hack to allow utils.ONNX_UNKNOWN_DIMENSION to override batchsize if needed.
            # default is -1.
            if shape[0] == -1:  # pylint: disable=E1136  # pylint/issues/3139
                # pylint: disable=unsupported-assignment-operation
                shape[0] = utils.ONNX_UNKNOWN_DIMENSION
            return shape
        return shape

    def get_rank(self, name):
        """Returns len(get_shape(name)) or None if shape is None."""
        shape = self.get_shape(name)
        if shape is None:
            return None
        return len(shape)

    def set_shape(self, name, val):
        """Set new shape of node."""
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, tuple):
            val = list(val)
        node = self.get_node_by_output(name, search_in_parent_graphs=True)
        utils.assert_error(node is not None, "cannot find node by output id %s", name)
        node.graph._output_shapes[name] = val

    def copy_shape(self, input_name, output_name):
        """Copy shape from another node."""
        shape = self.get_shape(input_name)
        # assert shape is not None
        if shape is not None:
            self.set_shape(output_name, shape)

    def add_graph_output(self, name, dtype=None, shape=None):
        """Add node output as graph's output."""
        utils.assert_error(name in self._output_to_node_name, "output %s not exist in the graph", name)

        if dtype is None:
            dtype = self.get_dtype(name)

        if shape is None:
            shape = self.get_shape(name)

        if name not in self.outputs:
            utils.assert_error(shape is not None, "shape for output %s should not be None", name)
            utils.assert_error(dtype is not None, "dtype for output %s should not be None", name)
            self.outputs.append(name)
            self.set_shape(name, shape)
            self.set_dtype(name, dtype)
        else:
            raise ValueError("graph output " + name + " already exists")

    def topological_sort(self, ops):
        """Topological sort of graph."""
        # sort by name, the result will be reversed alphabeta
        ops.sort(key=lambda op: op.name)

        def _push_stack(stack, node, in_stack):
            stack.append(node)
            if node in in_stack:
                raise ValueError("Graph has cycles, node.name=%r." % ops[node].name)
            in_stack[node] = True

        def _get_unvisited_child(g, node, not_visited):
            for child in g[node]:
                if child in not_visited:
                    return child
            return -1

        n = len(ops)
        g = [[] for _ in range(n)]
        op_name_to_index = {}
        for i, op in enumerate(ops):
            op_name_to_index[op.name] = i

        for i, op in enumerate(ops):
            all_input = set(op.input)
            implicit_inputs = op.get_implicit_inputs()
            all_input |= set(implicit_inputs)
            # remove those empty inputs
            all_input = list(filter(lambda a: a != "", all_input))
            for inp in sorted(all_input):
                j = self.get_node_by_output(inp)
                utils.assert_error(j is not None, "Cannot find node with output %r in graph %r", inp, self.graph_name)
                if self.parent_graph and j.name not in op_name_to_index:
                    # there might be some outer-scoped inputs for an inner Graph.
                    pass
                else:
                    g[op_name_to_index[j.name]].append(i)

        # label for each op. highest = sink nodes.
        label = [-1 for _ in range(n)]
        stack = []
        in_stack = dict()
        not_visited = dict.fromkeys(range(n))
        label_counter = n - 1

        while not_visited:
            node = list(not_visited.keys())[0]
            _push_stack(stack, node, in_stack)
            while stack:
                node = _get_unvisited_child(g, stack[-1], not_visited)
                if node != -1:
                    _push_stack(stack, node, in_stack)
                else:
                    node = stack.pop()
                    in_stack.pop(node)
                    not_visited.pop(node)
                    label[node] = label_counter
                    label_counter -= 1

        ret = [x for _, x in sorted(zip(label, ops))]
        self.reset_nodes(ret)

    def make_graph(self, doc, graph_name=None):
        """Create GraphProto for onnx from internal graph.

        Args:
            doc: text for doc string of the graph
            graph_name: optimize graph name
        """
        graph_name = graph_name or self.graph_name
        self.delete_unused_nodes(self.outputs)
        self.topological_sort(self.get_nodes())
        self.update_proto()

        ops = []
        const_ops = []
        graph_inputs = self.inputs.copy()
        for op in self.get_nodes():
            if op.is_const():
                const_ops.append(op)
            elif op.is_graph_input():
                if op not in graph_inputs:
                    graph_inputs.append(op)
            else:
                ops.append(op)

        # create initializers for placeholder with default nodes
        initializers = []
        placeholder_default_const_ops = []
        for op in graph_inputs:
            if op.type == "PlaceholderWithDefault":
                utils.assert_error(op.inputs[0] is not None, "Cannot find node with output {}".format(op.input[0]))
                utils.assert_error(
                    op.inputs[0].is_const(),
                    "non-const default value for PlaceholderWithDefault node '%s' is not supported. "
                    "Use the --use_default or --ignore_default flags to convert this node.",
                    op.name,
                )
                # copy the tensor value, set its name to current node's output, add as initializer
                value = op.inputs[0].get_tensor_value(as_list=False)
                tensor = numpy_helper.from_array(value, op.output[0])
                initializers.append(tensor)
                placeholder_default_const_ops.append(op.inputs[0])

        # create initializers for constant nodes
        const_ops = [op for op in const_ops if op not in placeholder_default_const_ops]
        for op in const_ops:
            # not to use numpy_helper.from_array to create a new tensor
            # because sometimes onnx will have a bug that only check the tensor data in specific field
            # such as at upsample it only checks the float_data field.
            t = op.get_value_attr()
            tensor = helper.get_attribute_value(t)
            tensor.name = op.output[0]
            initializers.append(tensor)

        # create input_tensor_values
        input_ids = [op.output[0] for op in graph_inputs]
        # onnx with IR version below 4 requires initializer should be in inputs.
        # here we check opset version rather than IR version for the reason:
        # https://github.com/onnx/tensorflow-onnx/pull/557
        # opset 9 come with IR 4.
        if self.opset < 9:
            input_ids += [op.output[0] for op in const_ops]

        input_tensor_values = self.make_onnx_graph_io(input_ids)

        # create output_tensor_values
        output_tensor_values = self.make_onnx_graph_io(self.outputs)

        tensor_value_info = []

        for op in ops:
            if op.domain in [utils.ONNX_DOMAIN, utils.AI_ONNX_ML_DOMAIN]:
                continue
            # We still don't 100% trust the accuracy of all the shapes in graph.py, but for custom ops they are
            # almost certainly accurate and onnx has no other way of knowing them.
            for out in op.output:
                if out == "" or out in self.outputs:
                    continue
                dtype = self.get_dtype(out)
                shape = self.get_shape(out)
                v = utils.make_onnx_inputs_outputs(out, dtype, shape)
                tensor_value_info.append(v)

        # create graph proto
        graph = helper.make_graph(
            [op.op for op in ops],
            graph_name,
            input_tensor_values,
            output_tensor_values,
            initializer=initializers,
            doc_string=doc,
            value_info=tensor_value_info,
        )

        return graph

    def make_model(self, graph_doc, graph_name="tfqdq_to_onnxqdq", **kwargs):
        """Create final ModelProto for onnx from internal graph.

        Args:
            graph_doc: text for doc string of the model
            graph_name: optimize graph name
        """
        graph = self.make_graph(graph_doc, graph_name)

        if "producer_name" not in kwargs:
            kwargs = {"producer_name": "neural compressor", "producer_version": "1.0.0"}
        if "opset_imports" not in kwargs:
            opsets = [helper.make_opsetid(utils.ONNX_DOMAIN, self._opset)]
            opsets.append(utils.AI_ONNX_ML_OPSET)
            if self.extra_opset is not None:
                opsets.extend(self.extra_opset)
            kwargs["opset_imports"] = opsets
        model_proto = helper.make_model(graph, **kwargs)

        utils.assert_error(
            self.opset in utils.OPSET_TO_IR_VERSION,
            "Opset %s is not supported yet. Please use a lower opset" % self.opset,
        )

        # set the IR version based on opset
        try:
            model_proto.ir_version = utils.OPSET_TO_IR_VERSION.get(self.opset, model_proto.ir_version)
        except:  # pylint: disable=bare-except
            logger.error("ir_version override failed - install the latest onnx version")

        return model_proto

    def make_onnx_graph_io(self, ids):
        """Create tensor_value_info for passed input/output ids."""
        tensor_value_infos = []
        for name in ids:
            dtype = self.get_dtype(name)
            shape = self.get_shape(name)

            utils.assert_error(dtype is not None, "missing output dtype for " + name)
            # TODO: allow None output shape or not? e.g. shape=(?,)
            # utils.assert_error(shape is not None, "missing output shape for " + name)
            if shape is None:
                logger.warning("missing output shape for %s", name)

            v = utils.make_onnx_inputs_outputs(name, dtype, shape)
            tensor_value_infos.append(v)
        return tensor_value_infos

    def dump_graph(self):
        """Dump graph with shapes (helpful for debugging)."""
        for node in self.get_nodes():
            input_names = ["{}{}".format(n, self.get_shape(n)) for n in node.input]
            logger.debug("%s %s %s %s", node.type, self.get_shape(node.output[0]), node.name, ", ".join(input_names))

    def dump_node_statistics(self, include_attrs=False, include_subgraphs=True):
        """Return a counter of op types (and optionally attribute names) within the graph."""
        op_cnt = collections.Counter()
        attr_cnt = collections.Counter()
        for n in self.get_nodes():
            op_cnt[n.type] += 1
            for k in n.attr.keys():
                attr_cnt[k] += 1
            body_graphs = n.get_body_graphs()
            if body_graphs and include_subgraphs:
                for b_g in body_graphs.values():
                    g_op_cnt, g_attr_cnt = b_g.dump_node_statistics(include_attrs=True, include_subgraphs=True)
                    op_cnt += g_op_cnt
                    attr_cnt += g_attr_cnt

        if include_attrs:
            return op_cnt, attr_cnt
        return op_cnt

    def remove_input(self, node, to_be_removed, input_index=None):
        """Remove input from Node.

        Args:
            node: the node we expect the input on
            to_be_removed: the node name we want to remove
            input_index: if not None, index of the input to be removed,
                the method is more efficient if *input_index* is specified,
                otherwise, it has to look for every input named *old_input*.
        """
        assert isinstance(node, OnnxNode) and isinstance(to_be_removed, six.text_type)
        if input_index is not None:
            assert node.input[input_index] == to_be_removed
            if node.input[input_index] in self._output_to_consumers:
                to_ops = self._output_to_consumers[node.input[input_index]]
                if node.name in to_ops:
                    to_ops.remove(node.name)
            del node.input[input_index]
            return

        for i, name in enumerate(node.input):
            if name == to_be_removed:
                utils.assert_error(
                    node.input.count(node.input[i]) <= 1,
                    "Node %r takes multiple times the same input %r. This case is not handled.",
                    node.name,
                    node.input[i],
                )
                self._unregister_input_name(node.input[i], node)
                del node.input[i]
                break

    def insert_new_node_on_input(self, node, op_type, input_name, name=None, domain=None, input_index=None, **kwargs):
        """Create and insert a new node into the graph.

        Args:
            node: we want to replace the input for this node
            op_type: type for new operation
            input_name: the name(s) of the outputs above us
                if scalar, new node placed above input_name
                if list, new node placed above input_name[0]. list is inputs into new node
            name: the name of the new op
            kwargs: attributes of the new node

        Returns:
            node that was inserted
        """
        if name is None:
            name = utils.set_name(node.name)
        new_output = utils.add_port_to_name(name)
        if not isinstance(input_name, list):
            input_name = [input_name]

        new_node = self.make_node(op_type, input_name, attr=kwargs, outputs=[new_output], name=name, domain=domain)
        if input_index is None:
            for i, n in enumerate(node.input):
                if n == input_name[0]:
                    self.replace_input(node, node.input[i], new_output, i)
                    break
        else:
            self.replace_input(node, node.input[input_index], new_output, input_index)
        return new_node

    def add_graph_input(self, name, dtype=None, shape=None):
        """Add placeholder node as graph's input. Order matters only for subgraph.

        Placeholders in original graph are assumed for main graph, order not matters.
        """
        if dtype is None:
            dtype = self.get_dtype(name)

        if shape is None:
            shape = self.get_shape(name)

        new_node = self.make_node("Placeholder", [], outputs=[name], dtypes=[dtype], shapes=[shape])
        self.inputs.append(new_node)

    def insert_node_on_output(self, node, output_name=None):
        """Insert a node into the graph.

        The inserted node takes the *output_name* as input and produces a
        new output. The function goes through every node taking *output_name*
        and replaces it by the new output name.
        """
        if output_name is None:
            output_name = node.input[0]
        new_output = node.output[0]

        to_replace = [self.get_node_by_name(n) for n in self._output_to_consumers[output_name]]
        to_replace = [n for n in to_replace if n != node]
        self.replace_all_inputs(output_name, new_output, ops=to_replace)
        return node

    def insert_new_node_on_output(self, op_type, output_name=None, name=None, inputs=None, domain=None, **kwargs):
        """Create and insert a new node into the graph. It then calls insert_node_on_output.

        Args:
            op_type: type for new operation
            output_name: the names of the outputs above us
            name: the name of the new op
            kwargs: attributes of the new node

        Returns:
            node that was inserted
        """
        utils.assert_error(
            isinstance(output_name, six.text_type), "output_name's type is not expected: %s", type(output_name)
        )
        utils.assert_error(isinstance(op_type, six.text_type), "op_type's type is not expected: %s", type(op_type))
        utils.assert_error(output_name is not None, "output_name cannot be None for op_type=%r.", op_type)

        if inputs is None:
            inputs = [output_name]
        if name is None:
            name = utils.set_name(op_type)

        new_output = utils.add_port_to_name(name)
        new_node = self.make_node(op_type, inputs, attr=kwargs, outputs=[new_output], name=name, domain=domain)
        return self.insert_node_on_output(new_node, output_name)

    def find_output_consumers(self, output_name):
        """Find all nodes consuming a given output."""
        if output_name in self._output_to_consumers:
            ops = self._output_to_consumers[output_name]
            ops = [self.get_node_by_name(n) for n in ops]
        else:
            ops = []  # self.get_nodes()
        nodes = []
        for node in ops:
            if node is None:
                continue
            if output_name in node.input:
                nodes.append(node)

        # find consumers in sub graphs
        if output_name in self._input_to_graph:
            for g in self._input_to_graph[output_name].values():
                nodes.extend(g.find_output_consumers(output_name))
        return nodes

    def _register_input_name(self, input_name, node, only_graph=False):
        """Register node taking a specific input."""
        if not only_graph:
            if input_name not in self._output_to_consumers:
                self._output_to_consumers[input_name] = set()
            self._output_to_consumers[input_name].add(node.name)
        if self.parent_graph is not None:
            if input_name not in self.parent_graph._input_to_graph:
                self.parent_graph._input_to_graph[input_name] = {}
            self.parent_graph._input_to_graph[input_name][id(self)] = self
            self.parent_graph._register_input_name(input_name, node, only_graph=True)

    def _unregister_input_name(self, input_name, node, only_graph=False):
        """Unregister node taking a specific input."""
        node_name = node.name
        if not only_graph:
            if input_name in self._output_to_consumers[input_name]:
                if node_name in self._output_to_consumers[input_name]:
                    self._output_to_consumers[input_name].remove(node_name)
        if (
            self.parent_graph is not None
            and input_name in self.parent_graph._input_to_graph
            and id(self) in self.parent_graph._input_to_graph[input_name]
        ):
            del self.parent_graph._input_to_graph[input_name][id(self)]
            self.parent_graph._unregister_input_name(input_name, node, only_graph=True)

    def replace_all_inputs(self, old_input, new_input, ops=None):
        """Replace all inputs pointing to old_input with new_input.

        *ops* is used if defined, otherwise `_output_to_consumers` is used to determine the impacted nodes.
        """
        if old_input == new_input:
            return
        if new_input not in self._output_to_consumers:
            self._output_to_consumers[new_input] = set()

        if ops is not None:
            keep_ops = True
        elif old_input in self._output_to_consumers:
            ops = list(
                filter(lambda a: a is not None, map(self.get_node_by_name, self._output_to_consumers[old_input]))
            )
            keep_ops = False
        else:
            ops = []
            keep_ops = False

        for node in ops:
            assert node is not None
            if old_input in node.input and new_input in node.output:
                raise RuntimeError("creating a circle in the graph is not allowed: " + node.name)
            self._register_input_name(new_input, node)

            for i, input_name in enumerate(node.input):
                if input_name == old_input:
                    self.replace_input(node, node.input[i], new_input, i)

        # modify references in sub graphs
        if old_input in self._input_to_graph:
            for g in self._input_to_graph[old_input].values():
                g.replace_all_inputs(old_input, new_input, ops=g.get_nodes() if keep_ops else None)

    def replace_input(self, node, old_input, new_input, input_index=None):
        """Replace one input in a node.

        The method is more efficient if *input_index* is specified.
        Otherwise, it renames every output named *old_input*.
        """
        assert (
            isinstance(node, OnnxNode) and isinstance(old_input, six.text_type) and isinstance(new_input, six.text_type)
        )
        is_replaced = False
        if input_index is None:
            for i, input_name in enumerate(node.input):
                if input_name == old_input:
                    node.input[i] = new_input
                    is_replaced = True
        elif node.input[input_index] == old_input:
            node.input[input_index] = new_input
            is_replaced = True
        else:
            raise RuntimeError("Unable to replace input %r into %r for node %r." % (old_input, new_input, node.name))

        to_ops = self._output_to_consumers.get(old_input, None)
        if to_ops is not None:
            if node.name in to_ops:
                # A node may take twice the same entry.
                to_ops.remove(node.name)

        self._register_input_name(new_input, node)
        return is_replaced

    def replace_inputs(self, node, new_inputs):
        """Replace node inputs."""
        assert isinstance(node, OnnxNode) and isinstance(new_inputs, list)

        for old_input in node.input:
            to_ops = self._output_to_consumers.get(old_input, None)
            if to_ops is not None and old_input in to_ops:
                # To avoid issues when a node
                # takes twice the same entry.
                to_ops.remove(old_input)

        for input_name in new_inputs:
            assert isinstance(input_name, six.text_type)
            self._register_input_name(input_name, node)

        node.input = new_inputs
        return True

    def _extract_sub_graph_nodes(self, dest_node, input_checker=None):
        """Return nodes of subgraph ending with dest_node.

        Args:
            dest_node: output node of the subgraph to find.
            input_checker: customized input check function: bool func(node)

        Return:
            a set of nodes.
        """
        res_set = set()
        if not dest_node or (input_checker and input_checker(dest_node) is False):
            return res_set

        processing_set = set([dest_node])
        while processing_set:
            top_node = processing_set.pop()
            res_set.add(top_node)
            all_inputs = top_node.input + list(top_node.get_implicit_inputs())
            for input_id in all_inputs:
                # we don't care about nested graph here, just handle current graph cropping.
                node = self.get_node_by_output(input_id, search_in_parent_graphs=False)
                if not node:
                    # some nodes (for example Scan) have optional inputs, which
                    # might have empty input.
                    # subgraph might have input defined in outer graph
                    continue
                if node not in res_set:
                    if input_checker and input_checker(node) is False:
                        continue
                    processing_set.add(node)
        return res_set

    def extract_sub_graph_nodes(self, outputs_name, input_checker=None, remove_unused_inputs=True):
        """Return nodes of subgraph having output_ids as outputs.

        Args:
            outputs_name: output node name of the subgraph to find.
            input_checker: customized input check function: bool func(node).
            remove_unused_inputs: bool, indicates whether unused placeholder inputs will be removed.
                in the resulting nodes.

        Return:
            a list of nodes.
        """
        res_set = set()

        outputs_to_keep = list(outputs_name)
        if not remove_unused_inputs:
            # add placeholder nodes even if they are not connected to outputs.
            # placeholder nodes with defaults can have inputs themselves
            outputs_to_keep += [inp.output[0] for inp in self.inputs]

        for output in outputs_to_keep:
            node = self.get_node_by_output(output, search_in_parent_graphs=False)
            res_set = res_set.union(self._extract_sub_graph_nodes(node, input_checker))

        return list(res_set)

    def delete_unused_nodes(self, outputs_name):
        """Delete nodes not in subgraph ending with output_names."""
        if not outputs_name:
            logger.debug("Outputs not specified, delete_unused_nodes not taking effect.")
            return

        # we need keep those placeholders that are used as input of Loop's body graph.
        # some of them are not used in the graph, but still need be there to keep the graph complete.
        related_nodes = self.extract_sub_graph_nodes(outputs_name, remove_unused_inputs=False)
        for node in related_nodes:
            attr_body_graphs = node.get_body_graphs()
            if attr_body_graphs:
                for body_graph in attr_body_graphs.values():
                    body_graph.delete_unused_nodes(body_graph.outputs)
        self.reset_nodes(related_nodes)

    def safe_to_remove_nodes(self, to_delete):
        """List of nodes that safe to delete, i.e. outputs not consumed by other nodes."""
        safe_to_remove = []
        delete_set = set(to_delete)
        for n in delete_set:
            out_consumers = set()
            for out in n.output:
                out_consumers |= set(self.find_output_consumers(out))
            if out_consumers.issubset(delete_set):
                safe_to_remove.append(n)
        return safe_to_remove

    def convert_qdq_nodes(self, q_node, dq_node):
        """Convert tensorflow QuantizeV2/Dequantize nodes to QuantizeLinear/DequantizeLinear."""
        qdq_node_output_dtype = self.get_dtype(dq_node.output[0])
        qdq_node_output_shape = self.get_shape(dq_node.output[0])

        # Get the attributes of qdq node
        signed_input = bool(q_node.get_attr_value("T", TensorProto.INT8) == TensorProto.INT8)

        max_quantized = 127

        if not signed_input:
            max_quantized = 255

        # Get axis attribute for per channel implementation.
        axis = q_node.get_attr_value("axis", -1)
        q_attrs = {}

        quantized_dtype = TensorProto.INT8 if signed_input else TensorProto.UINT8

        if axis != -1:
            utils.assert_error(self.opset >= 13, "Opset >= 13 is required for per channel quantization")
            q_attrs["axis"] = axis

            inp_rank = self.get_rank(q_node.input[0])
            utils.assert_error(inp_rank is not None, "Input rank cannot be unknown for qdq op %s", q_node.name)

        # Get the min and max value of the inputs to QDQ op
        min_value = self.get_tensor_value(q_node.input[1])
        max_value = self.get_tensor_value(q_node.input[2])

        if isinstance(min_value, list):
            num_channels = len(min_value)
        else:
            num_channels = 1

        scales = np.zeros(num_channels, dtype=np.float32)

        # Per-Tensor
        if num_channels == 1:
            # sing U8 as default for per tensor
            max_quantized = 255
            # Calculate scale from the min and max values
            scale = (float(max_value) - min_value) / max_quantized if min_value != max_value else 1

            zero_point = round((0 - min_value) / scale)
            zero_point = np.uint8(round(max(0, min(255, zero_point))))

            utils.assert_error(scale > 0, "Quantize/Dequantize scale must be greater than zero")
            scales = np.float32(scale)
            zero_point_np = zero_point
        # Per-Channel
        else:
            zero_point = np.zeros(num_channels, dtype=np.int8 if signed_input else np.uint8)
            for i in range(num_channels):
                # Calculate scales from the min and max values
                if signed_input:
                    max_range = max(abs(min_value[i]), abs(max_value[i]))
                    scale = (float(max_range) * 2) / max_quantized if max_range > 0 else 1
                else:
                    scale = (float(max_value[i]) - min_value[i]) / max_quantized if min_value[i] != max_value[i] else 1

                if scale == 1 or signed_input:
                    zero_point[i] = np.int8(0)
                else:
                    zero_point[i] = round((0 - min_value[i]) / scale)
                    zero_point[i] = np.uint8(round(max(0, min(255, zero_point[i]))))

                utils.assert_error(scale > 0, "Quantize/Dequantize scale must be greater than zero")
                scales[i] = np.float32(scale)
            utils.assert_error(axis != -1, "Axis must be specified for per channel quantization")
            zero_point_np = zero_point

        # Add QuantizeLinear and DequantizeLinear and remove the TF QDQ node reference
        cast_scale = scales.astype(np.float32)
        scale = self.make_const(name=utils.set_name("quant_scale"), np_val=cast_scale).output[0]
        zero_point = self.make_const(utils.set_name("zero_point"), zero_point_np).output[0]

        quant_node = self.make_node(
            op_type="QuantizeLinear",
            inputs=[q_node.input[0], scale, zero_point],
            shapes=[qdq_node_output_shape],
            attr=q_attrs,
            dtypes=[quantized_dtype],
            name=utils.set_name("QuantLinearNode"),
        )

        self.set_shape(quant_node.output[0], qdq_node_output_shape)

        self.remove_node(q_node.name)
        self.remove_node(dq_node.name)

        dequant_node = self.make_node(
            op_type="DequantizeLinear",
            inputs=[quant_node.output[0], scale, zero_point],
            outputs=[dq_node.output[0]],
            shapes=[qdq_node_output_shape],
            attr=q_attrs,
            dtypes=[qdq_node_output_dtype],
            name=utils.set_name("DequantLinearNode"),
        )
        self.set_shape(dequant_node.output[0], qdq_node_output_shape)

    def delete_qdq_nodes(self, q_node, dq_node):
        """Delete tensorflow QuantizeV2/Dequantize in the onnx graph."""
        qdq_input = q_node.input[0]
        qdq_output = dq_node.output[0]

        qdq_output_consumers = self.find_output_consumers(qdq_output)
        for consumer in qdq_output_consumers:
            self.replace_input(consumer, qdq_output, qdq_input)

        self.remove_node(dq_node.name)
        self.remove_node(q_node.name)

    def optimize_conv_add_fusion(self, node):
        """Fuse conv and add."""
        if node.type != "Add":
            return []

        conv_node = self.get_node_by_output(node.input[0])
        if conv_node.type != "Conv":
            return []

        if len(self.find_output_consumers(conv_node.output[0])) > 1:
            return []

        next_nodes = self.find_output_consumers(node.output[0])
        for next_node in next_nodes:
            if next_node.type == "Add":
                return []

        if self.is_const(node.input[1]):
            bias_tensor = self.get_node_by_name(node.input[1]).get_tensor_value(as_list=False)
            if bias_tensor.ndim > 1:
                bias_tensor = np.squeeze(bias_tensor)
                self.get_node_by_name(node.input[1]).set_tensor_value(bias_tensor)
        else:
            return []

        input_dequantize_node = self.get_node_by_output(conv_node.input[0])
        weight_dequantize_node = self.get_node_by_output(conv_node.input[1])
        if re.search(r"\w+:\d+", input_dequantize_node.input[1]):
            input_dequantize_node.input[1] = input_dequantize_node.input[1].rsplit(":", 1)[0]
        if re.search(r"\w+:\d+", weight_dequantize_node.input[1]):
            weight_dequantize_node.input[1] = weight_dequantize_node.input[1].rsplit(":", 1)[0]
        input_scale = self.get_node_by_name(input_dequantize_node.input[1]).get_tensor_value(as_list=False)
        weight_scale = self.get_node_by_name(weight_dequantize_node.input[1]).get_tensor_value(as_list=False)
        bias_scale_val = input_scale * weight_scale
        bias_zp_val = np.zeros(bias_scale_val.shape, dtype=np.int32).reshape(-1)
        quantized_bias = (bias_tensor / bias_scale_val).round().astype(np.int32)
        bias_scale = self.make_const(name=utils.set_name(node.name + "_scale"), np_val=bias_scale_val).output[0]
        bias_zero_point = self.make_const(utils.set_name(node.name + "_zero_point"), bias_zp_val).output[0]
        bias_input = self.make_const(name=utils.set_name(node.name + "_x"), np_val=quantized_bias).output[0]

        dequant_bias_node = self.make_node(
            op_type="DequantizeLinear",
            inputs=[bias_input, bias_scale, bias_zero_point],
            outputs=[conv_node.name],
            shapes=[bias_scale_val.shape],
            attr=weight_dequantize_node.attr,
            dtypes=[TensorProto.INT32],
            name=utils.set_name("DequantLinearNode"),
        )

        # Backup the conv and biasadd values
        conv_type = conv_node.type
        conv_input = conv_node.input
        conv_attr = conv_node.attr
        dtype = self.get_dtype(conv_node.output[0])
        shape = self.get_shape(conv_node.output[0])
        conv_name = conv_node.name
        conv_output = node.output
        if len(conv_input) == 3:
            conv_inputs = [conv_input[0], conv_input[1], conv_input[2], dequant_bias_node.output[0]]
        else:
            conv_inputs = [conv_input[0], conv_input[1], dequant_bias_node.output[0]]

        # Remove the Conv and Add node
        self.remove_node(conv_node.name)
        self.remove_node(node.name)

        self.make_node(
            conv_type,
            conv_inputs,
            attr=conv_attr,
            name=conv_name,
            outputs=conv_output,
            shapes=[shape],
            dtypes=[dtype],
            skip_conversion=False,
        )
        return []

    def apply_onnx_fusion(self):
        """Optimize graph with fusion."""
        graph_changed = True
        while graph_changed:
            graph_changed = False
            nodes = self.get_nodes()
            for node in nodes:
                if node.type == "Add" and self.optimize_conv_add_fusion(node):
                    graph_changed = True
        return self
