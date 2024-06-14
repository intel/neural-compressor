#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2021 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Tensorflow Utils Helper functions."""

import os
from collections import OrderedDict, UserDict

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2, variable_pb2
from tensorflow.core.protobuf import config_pb2, meta_graph_pb2
from tensorflow.python.eager import context, wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import load, save, signature_constants, tag_constants
from tensorflow.python.training import saver
from tensorflow.python.util import nest

from neural_compressor.common import logger
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer, GraphRewriterHelper


def read_graph(in_graph, in_graph_is_binary=True):
    """Reads input graph file as GraphDef.

    :param in_graph: input graph file.
    :param in_graph_is_binary: whether input graph is binary, default True.
    :return: input graphDef.
    """
    assert gfile.Exists(in_graph), "Input graph pb file %s does not exist." % in_graph

    input_graph_def = graph_pb2.GraphDef()
    mode = "rb" if in_graph_is_binary else "r"
    with gfile.Open(in_graph, mode) as f:
        data = f.read()
        if in_graph_is_binary:
            input_graph_def.ParseFromString(data)
        else:
            text_format.Merge(data, input_graph_def)

    return input_graph_def


def write_graph(out_graph_def, out_graph_file):
    """Write output graphDef to file.

    :param out_graph_def: output graphDef.
    :param out_graph_file: path to output graph file.
    :return: None.
    """
    assert isinstance(out_graph_def, tf.compat.v1.GraphDef), "out_graph_def is not instance of TensorFlow GraphDef."

    assert out_graph_file and os.path.exists(
        os.path.dirname(out_graph_file)
    ), '"output_graph" directory does not exists.'

    f = gfile.GFile(out_graph_file, "wb")
    f.write(out_graph_def.SerializeToString())


def is_ckpt_format(model_path):
    """Check the model_path format is ckpt or not.

    Args:
        model_path (string): the model folder path

    Returns:
        string: return the ckpt prefix if the model_path contains ckpt format data else None.
    """
    file_list = [os.path.splitext(i)[-1] for i in os.listdir(model_path)]
    if file_list.count(".meta") == 1 and file_list.count(".index") == 1:
        return True
    return False


def _parse_ckpt_bn_input(graph_def):
    """Parse ckpt batch norm inputs to match correct moving mean and variance.

    Args:
        graph_def (graph_def): original graph_def
    Returns:
        graph_def: well linked graph_def
    """
    for node in graph_def.node:
        if node.op == "FusedBatchNorm":
            moving_mean_op_name = node.input[3]
            moving_var_op_name = node.input[4]
            moving_mean_op = _get_nodes_from_name(moving_mean_op_name, graph_def)[0]
            moving_var_op = _get_nodes_from_name(moving_var_op_name, graph_def)[0]

            if moving_mean_op.op == "Const":
                name_part = moving_mean_op_name.rsplit("/", 1)[0]
                real_moving_mean_op_name = name_part + "/moving_mean"
                if len(_get_nodes_from_name(real_moving_mean_op_name, graph_def)) > 0:
                    # replace the real moving mean op name
                    node.input[3] = real_moving_mean_op_name

            if moving_var_op.op == "Const":
                name_part = moving_var_op_name.rsplit("/", 1)[0]
                real_moving_var_op_name = name_part + "/moving_variance"
                if len(_get_nodes_from_name(real_moving_var_op_name, graph_def)) > 0:
                    # replace the real moving mean op name
                    node.input[4] = real_moving_var_op_name

    return graph_def


def _get_nodes_from_name(node_name, graph_def):
    """Get nodes from graph_def using node name.

    Args:
        graph_def (graph_def): graph_def
        node_name (str): node name

    Returns:
        node (NodeDef): graph node
    """
    return [node for node in graph_def.node if node.name == node_name]


def is_saved_model_format(model_path):
    """Check the model_path format is saved_model or not.

    Args:
        model_path (string): the model folder path

    Returns:
        bool: return True if the model_path contains saved_model format else False.
    """
    file_list = [os.path.splitext(i)[-1] for i in os.listdir(model_path)]
    # TF 2.11.0 added a new fingerprint.pb to the SavedModel directory.
    return bool(file_list.count(".pb") in [1, 2, 3] and ("variables") in os.listdir(model_path))


def get_tensor_by_name(graph, name, try_cnt=3):
    """Get the tensor by name.

    Considering the 'import' scope when model may be imported more then once,
    handle naming format like both name:0 and name.

    Args:
        graph (tf.compat.v1.GraphDef): the model to get name from
        name (string): tensor of tensor_name:0 or tensor_name without suffixes
        try_cnt: the times to add 'import/' to find  tensor

    Returns:
        tensor: tensor got by name.
    """
    if name.find(":") == -1:
        name = name + ":0"
    for _ in range(try_cnt):
        try:
            return graph.get_tensor_by_name(name)
        except BaseException:
            name = "import/" + name
    raise ValueError("can not find tensor by name")


def iterator_sess_run(sess, iter_op, feed_dict, output_tensor, iteration=-1, measurer=None):
    """Run the graph that have iterator integrated in the graph.

    Args:
        sess (tf.compat.v1.Session): the model sess to run the graph
        iter_op (Operator): the MakeIterator op
        feed_dict(dict): the feeds to initialize a new iterator
        output_tensor(list): the output tensors
        iteration(int): iterations to run, when -1 set, run to end of iterator

    Returns:
        preds: the results of the predictions
    """
    sess.run(iter_op, feed_dict)
    preds = []
    idx = 0
    while idx + 1 != iteration:
        try:
            if measurer:
                measurer.start()
                prediction = sess.run(output_tensor)
                measurer.end()
            else:
                prediction = sess.run(output_tensor)
            preds.append(prediction)
            idx += 1
        except tf.errors.OutOfRangeError:
            break

    preds = collate_tf_preds(preds)
    return preds


def collate_tf_preds(results):
    """Collate the prediction results."""
    batch = results[0]
    if isinstance(batch, list):
        results = zip(*results)
        collate_results = []
        for output in results:
            if isinstance(output[0], np.ndarray):
                collate_results.append(np.concatenate(output))
            elif np.isscalar(output[0]):
                collate_results.extend(output)
    elif isinstance(batch, np.ndarray):
        collate_results = np.concatenate(results)

    return collate_results


def get_input_output_node_names(graph_def):
    """Get the input node name and output node name of the graph_def."""
    g = GraphAnalyzer()
    g.graph = graph_def
    g.parse_graph()
    return g.get_graph_input_output()


def fix_ref_type_of_graph_def(graph_def):
    """Fix ref type of the graph_def."""
    # according to https://github.com/onnx/tensorflow-onnx/issues/77
    for node in graph_def.node:
        if node.op == "RefSwitch":
            node.op = "Switch"
            for index in range(len(node.input)):
                if "moving_" in node.input[index]:
                    node.input[index] = node.input[index] + "/read"
        elif node.op == "AssignSub":
            node.op = "Sub"
            if "use_locking" in node.attr:
                del node.attr["use_locking"]
        elif node.op == "AssignAdd":
            node.op = "Add"
            if "use_locking" in node.attr:
                del node.attr["use_locking"]
        elif node.op == "Assign":
            node.op = "Identity"
            if "use_locking" in node.attr:
                del node.attr["use_locking"]
            if "validate_shape" in node.attr:
                del node.attr["validate_shape"]
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]
    return graph_def


def strip_unused_nodes(graph_def, input_node_names, output_node_names):
    """Strip unused nodes of the graph_def.

    The strip_unused_nodes pass is from tensorflow/python/tools/strip_unused_lib.py
    of official tensorflow r1.15 branch
    """
    cur_graph = GraphAnalyzer()
    cur_graph.graph = graph_def
    graph_info = cur_graph.parse_graph()
    type_attr = {"Sub": "T", "RealDiv": "T", "Identity": "T"}
    # this op should not be stripped for table initialization
    if "init_all_tables" in graph_info.keys():
        output_node_names.append("init_all_tables")
    not_found = {name for name in input_node_names}
    for node_name in list(graph_info.keys()):
        if node_name in not_found:
            not_found.remove(node_name)
            node = graph_info[node_name].node
            # skip the conversion to Placeholder that with type list
            if "component_types" in node.attr:
                continue
            original_output = graph_info[node_name].outputs
            placeholder_node = node_def_pb2.NodeDef()
            placeholder_node.op = "Placeholder"
            placeholder_node.name = node.name

            if "dtype" in node.attr:
                placeholder_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=node.attr["dtype"].type))
            elif node.op in type_attr.keys():
                placeholder_node.attr["dtype"].CopyFrom(
                    attr_value_pb2.AttrValue(type=node.attr[type_attr[node.op]].type)
                )
            else:
                raise KeyError("%s op's type attribute is not found," "you should add it to type_attr dict" % node.op)
            if "_output_shapes" in node.attr:
                placeholder_node.attr["_output_shapes"].CopyFrom(node.attr["_output_shapes"])
            if "shape" in node.attr:
                placeholder_node.attr["shape"].CopyFrom(node.attr["shape"])

            cur_graph.remove_node(node_name)

            cur_graph.replace_const_node(placeholder_node, [node_name], original_output)

    return tf.compat.v1.graph_util.extract_sub_graph(cur_graph.dump_graph(), output_node_names)


def strip_equivalent_nodes(graph_def, output_node_names):
    """Strip nodes with the same input and attr."""
    stripped_graph = GraphAnalyzer()
    stripped_graph.graph = graph_def
    stripped_graph_info = stripped_graph.parse_graph()

    def is_equivalent_input(input_tensor_list_1, input_tensor_list_2):
        if len(input_tensor_list_1) != len(input_tensor_list_2):
            return False
        const_num = 0
        for input_tensor_1, input_tensor_2 in zip(input_tensor_list_1, input_tensor_list_2):
            input_node_1 = stripped_graph_info[GraphRewriterHelper.node_name_from_input(input_tensor_1)].node
            input_node_2 = stripped_graph_info[GraphRewriterHelper.node_name_from_input(input_tensor_2)].node
            if input_node_1.op in ["Const", "HostConst"] and input_node_2.op in ["Const", "HostConst"]:
                if input_node_1.attr != input_node_2.attr:
                    return False
                const_num += 1
            elif input_tensor_1 != input_tensor_2:
                return False
        if const_num == len(input_tensor_list_1):
            return False
        return True

    nodes_to_remove = []
    replaced_nodes_type = {}
    stripped_graph_node_names = list(stripped_graph_info.keys())
    len_nodes = len(stripped_graph_node_names)
    for idx_1 in range(len_nodes - 1):
        node_name_1 = stripped_graph_node_names[idx_1]
        node_1 = stripped_graph_info[node_name_1].node
        if node_1.op in ["Const", "HostConst", "MatMul", "TensorArrayV3"] or node_name_1 in nodes_to_remove:
            continue
        for idx_2 in range(idx_1 + 1, len_nodes):
            node_name_2 = stripped_graph_node_names[idx_2]
            node_2 = stripped_graph_info[node_name_2].node
            if (
                node_1.op == node_2.op
                and node_name_1 != node_name_2
                and node_2 not in nodes_to_remove
                and node_1.input
                and is_equivalent_input(node_1.input, node_2.input)
                and node_1.attr == node_2.attr
            ):
                for ouput_node_name in stripped_graph_info[node_name_2].outputs:
                    output_node = stripped_graph_info[ouput_node_name].node
                    for idx_output_node_input, output_node_input_name in enumerate(output_node.input):
                        if GraphRewriterHelper.node_name_from_input(output_node_input_name) == node_name_2:
                            new_input = output_node_input_name.replace(node_name_2, node_name_1)
                            output_node.input[idx_output_node_input] = new_input
                            logger.debug(
                                "Replacing {} node '{}' with equivalent node '{}': "
                                "set {} node '{}'.input[{}] = '{}'".format(
                                    node_1.op,
                                    node_name_2,
                                    node_name_1,
                                    output_node.op,
                                    output_node.name,
                                    idx_output_node_input,
                                    new_input,
                                )
                            )
                            replaced_nodes_type[node_1.op] = replaced_nodes_type.get(node_1.op, 0) + 1
                            nodes_to_remove.append(node_name_2)
    for node_to_remove in nodes_to_remove:
        stripped_graph.remove_node(node_to_remove)
    return (
        tf.compat.v1.graph_util.extract_sub_graph(
            stripped_graph.dump_graph(), list(set(stripped_graph_node_names).intersection(output_node_names))
        ),
        replaced_nodes_type,
    )


# THIS API IS TO BE DEPRECATED!
def get_graph_def(model, outputs=[], auto_input_output=False):
    """Get the model's graph_def."""
    from neural_compressor.tensorflow.utils import BaseModel, Model

    if not isinstance(model, BaseModel):
        model = Model(model)
        model.output_tensor_names = outputs
    return model.graph_def


def get_model_input_shape(model):
    """Get the input shape of the input model."""
    for node in model.graph_def.node:
        if node.op == "Placeholder":
            _shape = list(tf.compat.v1.TensorShape(node.attr["shape"].shape))
            if tf.__version__ < "2.0.0":
                _shape = [item.value for item in _shape]
            if len(_shape) > 1 and isinstance(_shape[0], int):
                return _shape[0]
            elif isinstance(_shape, list) and hasattr(_shape[0], "value"):
                return _shape[0].value
    return 1


def generate_feed_dict(input_tensor, inputs):
    """Generate feed dict helper function."""
    if len(input_tensor) == 1:
        feed_dict = {}
        if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) or isinstance(inputs, UserDict):
            for name in inputs:
                for tensor in input_tensor:
                    pos = tensor.name.rfind(":")
                    t_name = tensor.name if pos < 0 else tensor.name[:pos]
                    if name == t_name:
                        feed_dict[tensor] = inputs[name]
                        break
        else:
            feed_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
    else:
        assert len(input_tensor) == len(inputs), "inputs len must equal with input_tensor"
        feed_dict = {}
        if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) or isinstance(inputs, UserDict):
            for name in inputs:
                for tensor in input_tensor:
                    pos = tensor.name.rfind(":")
                    t_name = tensor.name if pos < 0 else tensor.name[:pos]
                    if name in [tensor.name, t_name]:
                        feed_dict[tensor] = inputs[name]
                        break
        else:
            # sometimes the input_tensor is not the same order with inputs
            # we should check and pair them
            def check_shape(tensor, data):
                # scalar or 1 dim default True
                if tensor.shape is None or len(tensor.shape.dims) == 1 or not hasattr(data, "shape"):
                    return True
                tensor_shape = tuple(tensor.shape)
                data_shape = tuple(data.shape)
                for tensor_dim, data_dim in zip(tensor_shape, data_shape):
                    if tensor_dim is not None and tensor_dim != data_dim:
                        return False
                return True

            disorder_tensors = []
            disorder_inputs = []
            for idx, sort_tensor in enumerate(input_tensor):
                sort_input = inputs[idx]
                if check_shape(sort_tensor, sort_input):
                    feed_dict.update({sort_tensor: sort_input})
                else:
                    disorder_tensors.append(sort_tensor)
                    disorder_inputs.append(sort_input)
            for i, dis_tensor in enumerate(disorder_tensors):
                for j, dis_input in enumerate(disorder_inputs):
                    if check_shape(dis_tensor, dis_input):
                        feed_dict.update({dis_tensor: dis_input})
                        break
    return feed_dict


def apply_inlining(func):
    """Apply an inlining optimization to the function's graph definition.

    Args:
        func: A concrete function get from saved_model.

    Returns:
        new_graph_def: The optimized graph in graph_def format.
    """
    graph_def = func.graph.as_graph_def()

    # In some cases, a secondary implementation of the function (e.g. for GPU) is
    # written to the "api_implements" attribute. (e.g. `tf.keras.layers.LSTM` in
    # TF2 produces a CuDNN-based RNN for GPU).
    # This function suppose to inline all functions calls, but "api_implements"
    # prevents this from happening. Removing the attribute solves the problem.
    # To learn more about "api_implements", see:
    #   tensorflow/core/grappler/optimizers/implementation_selector.h
    for function in graph_def.library.function:
        if "api_implements" in function.attr:
            del function.attr["api_implements"]

    meta_graph = saver.export_meta_graph(graph_def=graph_def, graph=func.graph)

    # Clear the initializer_name for the variables collections, since they are not
    # needed after saved to saved_model.
    for name in ["variables", "model_variables", "trainable_variables", "local_variables"]:
        raw_list = []
        for raw in meta_graph.collection_def["variables"].bytes_list.value:
            variable = variable_pb2.VariableDef()
            variable.ParseFromString(raw)
            variable.ClearField("initializer_name")
            raw_list.append(variable.SerializeToString())
        meta_graph.collection_def[name].bytes_list.value[:] = raw_list

    # Add a collection 'train_op' so that Grappler knows the outputs.
    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in func.inputs + func.outputs:
        fetch_collection.node_list.value.append(array.name)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

    # Initialize RewriterConfig with everything disabled except function inlining.
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.min_graph_nodes = -1  # do not skip small graphs
    rewrite_options.optimizers.append("function")

    new_graph_def = tf_optimizer.OptimizeGraph(config, meta_graph)

    return new_graph_def


def construct_function_from_graph_def(func, graph_def, frozen_func=None):
    """Rebuild function from graph_def.

    Args:
        func: The original concrete function get from saved_model.
        graph_def: The optimized graph after applying inlining optimization.

    Returns:
        new_func: The reconstructed function.
    """
    if frozen_func is None:
        frozen_func = func

    # If a function is converted, then the TF context contains the original
    # function while the converted_graph_def contains the converted function.
    # Remove the original function from the TF context in this case.
    for f in graph_def.library.function:
        while context.context().has_function(f.signature.name):
            context.context().remove_function(f.signature.name)

    captures = {c[1].name.split(":")[0]: c[0] for c in frozen_func.graph.captures}
    new_func = wrap_function.function_from_graph_def(
        graph_def,
        [tensor.name for tensor in frozen_func.inputs],
        [tensor.name for tensor in frozen_func.outputs],
        captures,
    )
    new_func.graph.structured_outputs = nest.pack_sequence_as(
        func.graph.structured_outputs, new_func.graph.structured_outputs
    )
    # new_func._function_type = func.function_type  # pylint: disable=protected-access

    # Copy structured input signature from original function (used during
    # serialization)
    new_func.graph.structured_input_signature = func.structured_input_signature

    return new_func


def parse_saved_model(model, freeze=False, input_tensor_names=[], output_tensor_names=[]):
    """Parse a input saved_model.

    Args:
        model(string or AutoTrackable object): The input saved_model.

    Returns:
        graph_def: The graph_def parsed from saved_model.
        _saved_model: TF AutoTrackable object loaded from saved_model.
        func: The concrete function get from saved_model.
        frozen_func: The reconstructed function from inlining optimized graph.
    """
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1

    if isinstance(model, str):
        _saved_model = load.load(model, [tag_constants.SERVING])
    else:
        _saved_model = model

    func = _saved_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    if freeze:
        frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)
    else:
        inlined_graph_def = apply_inlining(func)
        frozen_func = construct_function_from_graph_def(func, inlined_graph_def)

    if len(input_tensor_names) == 0:
        # skip all inputs for ReadVariableOp
        input_tensor_names = [i.name.split(":")[0] for i in frozen_func.inputs if "unknown" not in i.name]
    if len(output_tensor_names) == 0:
        output_tensor_names = [i.name.split(":")[0] for i in frozen_func.outputs]

    frozen_graph_def = frozen_func.graph.as_graph_def()
    grappler_meta_graph_def = saver.export_meta_graph(graph_def=frozen_graph_def, graph=frozen_func.graph)

    # Add a collection 'train_op' so that Grappler knows the outputs.
    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in frozen_func.inputs + frozen_func.outputs:
        fetch_collection.node_list.value.append(array.name)
        grappler_meta_graph_def.collection_def["train_op"].CopyFrom(fetch_collection)

    grappler_session_config = config_pb2.ConfigProto()
    rewrite_options = grappler_session_config.graph_options.rewrite_options
    rewrite_options.min_graph_nodes = -1
    graph_def = tf_optimizer.OptimizeGraph(grappler_session_config, grappler_meta_graph_def, graph_id=b"tf_graph")
    return graph_def, _saved_model, func, frozen_func, input_tensor_names, output_tensor_names


def reconstruct_saved_model(graph_def, func, frozen_func, trackable, path):
    """Reconstruct a saved_model.

    Args:
        graph_def: The input graph_def.
        func: The concrete function get from the original saved_model.
        frozen_func: The reconstructed function from inlining optimized graph.
        trackable: TF AutoTrackable object loaded from the original saved_model.
        path: The destination path to save the reconstructed saved_model.
    """
    converted_func = construct_function_from_graph_def(func, graph_def, frozen_func)
    signatures = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: converted_func}
    save.save(trackable, path, signatures, options=None)
