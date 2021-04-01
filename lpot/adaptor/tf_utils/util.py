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

import os
import numpy as np
from google.protobuf import text_format
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from lpot.utils import logger
from .graph_rewriter.graph_util import GraphAnalyzer
from lpot.model.model import TensorflowModel as LpotModel

def disable_random(seed=1):
    """A Decorator to disable tf random seed.
    """
    def decorator(func):
        def wrapper(*args, **kw):
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.reset_default_graph()
            tf.compat.v1.set_random_seed(seed)
            return func(*args, **kw)
        return wrapper
    return decorator

def read_graph(in_graph, in_graph_is_binary=True):
    """Reads input graph file as GraphDef.

    :param in_graph: input graph file.
    :param in_graph_is_binary: whether input graph is binary, default True.
    :return: input graphDef.
    """
    assert gfile.Exists(in_graph), 'Input graph pb file %s does not exist.' % in_graph

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
    assert isinstance(
        out_graph_def,
        tf.compat.v1.GraphDef), 'out_graph_def is not instance of TensorFlow GraphDef.'

    assert out_graph_file and os.path.exists(os.path.dirname(
        out_graph_file)), '"output_graph" directory does not exists.'

    f = gfile.GFile(out_graph_file, 'wb')
    f.write(out_graph_def.SerializeToString())


def is_ckpt_format(model_path):
    """check the model_path format is ckpt or not.

    Args:
        model_path (string): the model folder path

    Returns:
        string: return the ckpt prefix if the model_path contains ckpt format data else None.
    """
    file_list = [os.path.splitext(i)[-1] for i in os.listdir(model_path)]
    if file_list.count('.meta') == 1 and file_list.count('.index') == 1:
        return True
    return False

def _parse_ckpt_bn_input(graph_def):
    """parse ckpt batch norm inputs to match correct moving mean and variance
    Args:
        graph_def (graph_def): original graph_def
    Returns:
        graph_def: well linked graph_def
    """
    for node in graph_def.node:
        if node.op == 'FusedBatchNorm':
            moving_mean_op_name = node.input[3]
            moving_var_op_name = node.input[4]
            moving_mean_op = _get_nodes_from_name(moving_mean_op_name, graph_def)[0]
            moving_var_op = _get_nodes_from_name(moving_var_op_name, graph_def)[0]

            if moving_mean_op.op == 'Const':
                name_part = moving_mean_op_name.rsplit('/', 1)[0]
                real_moving_mean_op_name = name_part + '/moving_mean'
                if len(_get_nodes_from_name(real_moving_mean_op_name, graph_def)) > 0:
                    # replace the real moving mean op name
                    node.input[3] = real_moving_mean_op_name

            if moving_var_op.op == 'Const':
                name_part = moving_var_op_name.rsplit('/', 1)[0]
                real_moving_var_op_name = name_part + '/moving_variance'
                if len(_get_nodes_from_name(real_moving_var_op_name, graph_def)) > 0:
                    # replace the real moving mean op name
                    node.input[4] = real_moving_var_op_name

    return graph_def

def _get_nodes_from_name(node_name, graph_def):
    """get nodes from graph_def using node name
    Args:
        graph_def (graph_def): graph_def
        node_name (str): node name

    Returns:
        node (NodeDef): graph node
    """
    return [node for node in graph_def.node if node.name == node_name]

def is_saved_model_format(model_path):
    """check the model_path format is saved_model or not

    Args:
        model_path (string): the model folder path

    Returns:
        bool: return True if the model_path contains saved_model format else False.
    """
    file_list = [os.path.splitext(i)[-1] for i in os.listdir(model_path)]
    return bool(file_list.count('.pb') == 1 and ('variables') in os.listdir(model_path))

def get_estimator_graph(estimator, input_fn):
    with tf.Graph().as_default() as g:
        features, input_hooks = estimator._get_features_from_input_fn(
            input_fn, tf.estimator.ModeKeys.PREDICT)
        estimator_spec = estimator._call_model_fn(features, None,
            tf.estimator.ModeKeys.PREDICT, estimator.config)

        outputs = [tensor.name for tensor in estimator_spec.predictions.values()] if\
            isinstance(estimator_spec.predictions, dict) else \
                [estimator_spec.predictions.name]
        logger.info('estimator output tensor names is {}'.format(outputs))
        with tf.compat.v1.Session(graph=g) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            # Freezing a graph requires output_node_names, which can be found in
            # estimator_spec.predictions that contains prediction tensors as a
            # dictionary
            # When a model uses Iterator, we need to have 'MakeIterator' (default
            # name used by TF) in the output_node_names as well.
            output_nodes = list(set([output.split(':')[0] for output in outputs]))
            if 'MakeIterator' in [node.op for node in g.as_graph_def().node]:
                output_nodes.append('MakeIterator')

            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
               g.as_graph_def(), output_nodes)

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
        return graph

def get_tensor_by_name(graph, name, try_cnt=3):
    """Get the tensor by name considering the 'import' scope when model
       may be imported more then once, handle naming format like both name:0 and name

    Args:
        graph (tf.compat.v1.GraphDef): the model to get name from
        name (string): tensor of tensor_name:0 or tensor_name without suffixes
        try_cnt: the times to add 'import/' to find  tensor

    Returns:
        tensor: tensor got by name.
    """
    if name.find(':') == -1:
        name = name + ':0'
    for _ in range(try_cnt):
        try:
            return graph.get_tensor_by_name(name)
        except BaseException:
            name = 'import/' + name
    raise ValueError('can not find tensor by name')

def iterator_sess_run(sess, iter_op, feed_dict, output_tensor, iteration=-1, measurer=None):
    """Run the graph that have iterator integrated in the graph

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
    while idx < iteration or iteration == -1:
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
    batch = results[0]
    if isinstance(batch, list):
        results = zip(*results)
        collate_results = []
        for output in results:
           collate_results.append(np.concatenate(output))
    elif isinstance(batch, np.ndarray):
        collate_results = np.concatenate(results)
    return collate_results

def get_input_node_names(graph_def):
    g = GraphAnalyzer()
    g.graph = graph_def
    g.parse_graph()
    return g.get_graph_input_output()[0]

def get_output_node_names(graph_def):
    g = GraphAnalyzer()
    g.graph = graph_def
    g.parse_graph()
    return g.get_graph_input_output()[1]

def fix_ref_type_of_graph_def(graph_def):
    # according to https://github.com/onnx/tensorflow-onnx/issues/77
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
            if 'validate_shape' in node.attr:
                del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]
    return graph_def

def strip_unused_nodes(graph_def, input_node_names, output_node_names):
    cur_graph = GraphAnalyzer()
    cur_graph.graph = graph_def
    graph_info = cur_graph.parse_graph()
    type_attr = {"Sub": "T"}
    not_found = {name for name in input_node_names}
    for node_name in list(graph_info.keys()):
        if node_name in not_found:
            not_found.remove(node_name)
            node = graph_info[node_name].node
            # skip the convertion to Placeholder that with type list
            if 'component_types' in node.attr:
                continue
            original_output = graph_info[node_name].outputs
            placeholder_node = node_def_pb2.NodeDef()
            placeholder_node.op = "Placeholder"
            placeholder_node.name = node.name

            if "dtype" in node.attr:
                placeholder_node.attr["dtype"].CopyFrom(
                    attr_value_pb2.AttrValue(type=node.attr["dtype"].type))
            elif node.op in type_attr.keys():
                placeholder_node.attr["dtype"].CopyFrom(
                    attr_value_pb2.AttrValue(type=node.attr[type_attr[node.op]].type))
            else:
                raise KeyError("%s op's type attribute is not found,"
                               "you should add it to type_attr dict" % node.op)
            if "_output_shapes" in node.attr:
                placeholder_node.attr["_output_shapes"].CopyFrom(node.attr["_output_shapes"])
            if "shape" in node.attr:
                placeholder_node.attr["shape"].CopyFrom(node.attr["shape"])

            cur_graph.remove_node(node_name)

            cur_graph.replace_const_node(placeholder_node, [node_name], original_output)

    return tf.compat.v1.graph_util.extract_sub_graph(cur_graph.dump_graph(),
                                                     output_node_names)

# THIS API IS TO BE DEPRECATED!
def get_graph_def(model, outputs=[], auto_input_output=False):
    if not isinstance(model, LpotModel): 
        framework_info = {'output_tensor_names': outputs}
        model = LpotModel(model, framework_info)
    return model.graph_def

