#
#  -*- coding: utf-8 -*-
#

import unittest
import tensorflow as tf

from lpot.adaptor.tf_utils.graph_rewriter.generic.convert_layout import ConvertLayoutOptimizer
from tensorflow.python.framework import graph_util

class TestConvertLayout(unittest.TestCase):
    def test_convert_layout(self):
        if tf.version.VERSION < '2.4.0':
            return
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1, 3, 224, 224], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                    initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], 
                                    padding="SAME", data_format='NCHW')
        relu = tf.nn.relu(conv)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[relu.name.split(':')[0]])
        outputs = [relu.name.split(':')[0]]
        post_graph = ConvertLayoutOptimizer(graph_def, outputs).do_transformation()
        converted = False
        for node in post_graph.node:
            if 'data_format' in node.attr and node.attr['data_format'].s == b'NHWC':
                converted = True
        self.assertEqual(converted, True)
        
if __name__ == '__main__':
    unittest.main()
