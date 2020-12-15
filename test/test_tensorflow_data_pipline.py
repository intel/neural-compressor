#
#  -*- coding: utf-8 -*-
#
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from lpot.adaptor.tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper
from lpot.adaptor.tf_utils.util import get_tensor_by_name, iterator_sess_run

class TestDataPipelineConvert(unittest.TestCase):

    def test_data_pipeline(self):
        tf.compat.v1.disable_eager_execution()
        raw_dataset = np.ones([100,224, 224, 3], dtype=np.float32)
        tf_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(raw_dataset)
        tf_dataset = tf_dataset.batch(1)
        ds_iterator = tf_dataset.make_initializable_iterator()
        iter_tensors = ds_iterator.get_next()

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
            initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
            initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(iter_tensors, conv_weights, 
            strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)
        relu = tf.nn.relu(conv_bias, name='Relu_1')

        output_names=[relu.name.split(':')[0], 'MakeIterator']

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=output_names)
            output_graph_def = QuantizeGraphHelper.remove_training_nodes(
                output_graph_def, protected_nodes=output_names)
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(output_graph_def, name='')
        print('graph has been generated....')
        
        iter_op = graph.get_operation_by_name('MakeIterator')
        output_tensor = get_tensor_by_name(graph, output_names[0]) 
        sess = tf.compat.v1.Session(graph=graph)
        iterator_sess_run(sess, iter_op, \
            feed_dict={}, output_tensor=output_tensor)


if __name__ == "__main__":
    unittest.main()
