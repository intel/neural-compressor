import os
import sys
import numpy as np
from google.protobuf import text_format
from tensorflow.python.framework import graph_util

try:
    import tensorflow.compat.v1 as tf_v1
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
except ImportError:
    import tensorflow as tf_v1

def generate_data(input_shape, input_dtype="float32", batch_size=1):
    np.random.seed(1024)
    if input_dtype in ["uint8", "int8", "int32", "int64"]:
        dummy_input = np.random.randint(1, 128, input_shape).astype(input_dtype)
    else:
        dummy_input = np.random.randn(*input_shape).astype(input_dtype)
    return np.repeat(dummy_input[np.newaxis, :], batch_size, axis=0)

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
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
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
    return graph_def