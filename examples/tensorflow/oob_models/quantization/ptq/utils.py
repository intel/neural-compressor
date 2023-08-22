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
# ==============================================================================

import os
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.compat.v1 import graph_util

try:
    import tensorflow.compat.v1 as tf_v1
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
except ImportError:
    import tensorflow as tf_v1

def generate_data(input_shape, input_dtype="float32",
                  batch_size=1, max_int_value=35,
                  newaxis=True, is_one_dim=False):
    np.random.seed(1024)
    if input_dtype in ["uint8", "int8", "int32", "int64"]:
        if is_one_dim:
            return np.random.randint(1, max_int_value, batch_size).astype(input_dtype)
        dummy_input = np.random.randint(1, max_int_value, input_shape).astype(input_dtype)
    else:
        if is_one_dim:
            return np.random.randn(batch_size).astype(input_dtype)
        dummy_input = np.random.randn(*input_shape).astype(input_dtype)
    # handle the case that the shape of the input is one-dimensional
    if newaxis is False:
        return np.repeat(dummy_input, batch_size, axis=0)
    return np.repeat(dummy_input[np.newaxis, :], batch_size, axis=0)

def generate_sparse_indice(dense_shape, input_dtype="int64", batch_size=1):
    dummy_indice = np.ones(shape=[batch_size, 2], dtype=input_dtype)
    dummy_indice[:,0] = np.random.randint(0, dense_shape[0], batch_size).astype(input_dtype)
    dummy_indice[:,1] = np.random.randint(0, dense_shape[1], batch_size).astype(input_dtype)
    return dummy_indice

def freeze_graph(input_checkpoint, output_graph, output_node_names):
    tf_v1.disable_eager_execution()
    meta_data_path = input_checkpoint + ".meta"
    saver = tf_v1.train.import_meta_graph(meta_data_path, clear_devices=True)

    with tf.compat.v1.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names)
    if output_graph:
        with tf.io.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    print("convert done!!") 
    print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

def delete_assign(graph_def):
    for node in graph_def.node:
        if "_class" in node.attr:
            del node.attr["_class"]
            # tf.compat.v1.logging.warning(f"Removing _class attr of {node.name}")
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            # for index in range(len(node.input)):
            #     if 'moving_' in node.input[index]:
            #         node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
            print("************ deal with a AssignAdd !")
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
            print("************ deal with a AssignSub !")
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]
    return graph_def

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
