#
#  -*- coding: utf-8 -*-
#

import unittest
import tensorflow as tf

from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.convert_leakyrelu import ConvertLeakyReluOptimizer
from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tf_utils.util import disable_random

class TestConvertLeaklyRelu(unittest.TestCase):
    @disable_random()
    def test_convert_leakyrelu(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1, 3, 224, 224], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                    initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], 
                                    padding="SAME", data_format='NCHW')
        relu = tf.nn.relu(conv)
        mul = tf.math.multiply(relu, tf.constant([0.1]))
        maximum = tf.math.maximum(relu, mul)
        identity = tf.identity(maximum)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[identity.name.split(':')[0]])
        post_graph = ConvertLeakyReluOptimizer(graph_def).do_transformation()
        converted = False
        for node in post_graph.node:
            if node.op == 'LeakyRelu':
                converted = True
        self.assertEqual(converted, True)
    
    @disable_random()
    def test_convert_leakyrelu_with_alpha_large_than_one(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1, 3, 224, 224], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                    initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], 
                                    padding="SAME", data_format='NCHW')
        relu = tf.nn.relu(conv)
        mul = tf.math.multiply(relu, tf.constant([3.1]))
        maximum = tf.math.maximum(relu, mul)
        identity = tf.identity(maximum)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[identity.name.split(':')[0]])
        post_graph = ConvertLeakyReluOptimizer(graph_def).do_transformation()
        converted = False
        for node in post_graph.node:
            if node.op == 'LeakyRelu':
                converted = True
        self.assertEqual(converted, False)

    @disable_random()
    def test_convert_leakyrelu_with_invalid_maximum(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1, 3, 224, 224], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                    initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], 
                                    padding="SAME", data_format='NCHW')
        relu = tf.nn.relu(conv)
        mul = tf.math.multiply(relu, tf.constant([2.1]))

        maximum = tf.math.maximum(conv, mul)
        identity = tf.identity(maximum)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[identity.name.split(':')[0]])
        post_graph = ConvertLeakyReluOptimizer(graph_def).do_transformation()
        converted = False
        for node in post_graph.node:
            if node.op == 'LeakyRelu':
                converted = True
        self.assertEqual(converted, False)

    @disable_random()
    def test_convert_leakyrelu_with_invalid_mul(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1, 3, 224, 224], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                    initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], 
                                    padding="SAME", data_format='NCHW')
        relu = tf.nn.relu(conv)
        const_identity = tf.identity(tf.constant([0.1]))
        mul = tf.math.multiply(relu, const_identity)

        maximum = tf.math.maximum(conv, mul)
        identity = tf.identity(maximum)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[identity.name.split(':')[0]])
        post_graph = ConvertLeakyReluOptimizer(graph_def).do_transformation()
        converted = False
        for node in post_graph.node:
            if node.op == 'LeakyRelu':
                converted = True
        self.assertEqual(converted, False)

if __name__ == '__main__':
    unittest.main()
