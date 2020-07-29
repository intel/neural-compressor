#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
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

from google.protobuf import text_format

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

def read_graph(in_graph, in_graph_is_binary=True):
    """Reads input graph file as GraphDef.

    :param in_graph: input graph file.
    :param in_graph_is_binary: whether input graph is binary, default True.
    :return: input graphDef.
    """
    if not gfile.Exists(in_graph):
        raise ValueError('Input graph pb file %s does not exist.' % in_graph)

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
    if not isinstance(out_graph_def, tf.compat.v1.GraphDef):
        raise ValueError(
            'out_graph_def is not instance of TensorFlow GraphDef.')
    if out_graph_file and not os.path.exists(os.path.dirname(out_graph_file)):
        raise ValueError('"output_graph" directory does not exists.')
    f = gfile.GFile(out_graph_file, 'wb')
    f.write(out_graph_def.SerializeToString())


def split_shared_inputs(in_graph, ops=[]):
    """
    Split shared inputs(like weights and bias) of ops list.
    :param in_graph: input graph file.
    :param ops: ops list to processing.
    :return: path to ouput graph file.
    """
    if not ops:
        return in_graph

    input_graph_def = read_graph(in_graph)

    # map of node_name - node
    node_map = {}
    for node in input_graph_def.node:
        if node.name not in node_map.keys():
            node_map[node.name] = node

    output_graph_def = graph_pb2.GraphDef()
    # map of input_name - op_name
    input_map = {}
    for node_name in node_map.keys():
        node = node_map[node_name]
        if node.op in ops:
            for input_idx, input_node_name in enumerate(node.input):
                if node_map[input_node_name].op == 'Const':
                    # is shared and current node is not the first one sharing
                    # the input
                    if input_node_name in input_map.keys():
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
    rewrite_graph = os.path.join(os.path.dirname(in_graph),
                                 'frozen_inference_graph_rewrite.pb')
    write_graph(output_graph_def, rewrite_graph)
    return rewrite_graph


def is_ckpt_format(model_path):
    """check the model_path format is ckpt or not.

    Args:
        model_path (string): the model folder path

    Returns:
        string: return the ckpt prefix if the model_path contains ckpt format data else None.
    """
    file_list = [os.path.splitext(i)[-1] for i in os.listdir(model_path)]
    if file_list.count('.meta') == 1 and file_list.count('.index') == 1:
        return [os.path.splitext(i)[0] for i in os.listdir(model_path) if i.endswith(".meta")][0]
    else:
        return None


def parse_ckpt_model(ckpt_prefix, outputs):
    """Parse the ckpt model

    Args:
        ckpt_prefix (string): the ckpt prefix for parsing
    """
    saver = tf.compat.v1.train.import_meta_graph(ckpt_prefix + '.meta',
                                                 clear_devices=True)
    graph = tf.compat.v1.get_default_graph()
    input_graph_def = graph.as_graph_def()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, ckpt_prefix)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=outputs)

    return output_graph_def

def is_saved_model_format(model_path):
    """check the model_path format is saved_model or not

    Args:
        model_path (string): the model folder path

    Returns:
        bool: return True if the model_path contains saved_model format else False.
    """
    file_list = [os.path.splitext(i)[-1] for i in os.listdir(model_path)]
    if file_list.count('.pb') == 1 and ('variables') in os.listdir(model_path):
        return True
    else:
        return False

def parse_savedmodel_model(model_path):
    """Convert SavedModel to graphdef

    Args:
        model_path (string): the model folder path

    Returns:
        graphdef: the parsed graphdef object.
        input_names: input node names
        output_names: output node name
    """

    with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            meta_graph = tf.compat.v1.saved_model.loader.load(
                sess, ["serve"], model_path)

            model_graph_signature = list(
                meta_graph.signature_def.items())[0][1]

            input_names = [input_item[1].name
                              for input_item in model_graph_signature.inputs.items()]

            output_names = [output_item[1].name
                               for output_item in model_graph_signature.outputs.items()]

            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[output_item[0]
                                   for output_item in model_graph_signature.outputs.items()])

            return output_graph_def, input_names, output_names

def convert_pb_to_savedmodel(graph_def, input_tensor_names, output_tensor_names, output_dir):
    """Convert the graphdef to SavedModel

    Args:
        graph_def (graphdef): parsed graphdef object.
        input_tensor_names (list): input tensor names list.
        output_tensor_names (list): output tensor names list.
        output_dir (string): Converted SavedModel store path.
    """
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(output_dir)

    sigs = {}
    with tf.compat.v1.Session() as sess:
        tf.import_graph_def(graph_def, name="")
        g = tf.compat.v1.get_default_graph()

        input_tensors = {}
        for input_tensor_name in output_tensor_names:
            input_tensors[input_tensor_name.split(':')[0]] = g.get_tensor_by_name(
                "{}".format(input_tensor_name))

        output_tensors = {}
        for output_tensor_name in input_tensor_names:
            output_tensors[output_tensor_name.split(':')[0]] = g.get_tensor_by_name(
                "{}".format(output_tensor_name))

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
            output_tensors, input_tensors)

        builder.add_meta_graph_and_variables(sess,
                                            [tag_constants.SERVING],
                                            signature_def_map=sigs)

    builder.save()
