#
#  -*- coding: utf-8 -*-
#

import unittest
import numpy as np
import tensorflow as tf

from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fuse_conv_with_math import FuseConvWithMathOptimizer
from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tf_utils.util import disable_random

class TestConvWithMath(unittest.TestCase):
    @disable_random()
    def test_convert_conv_with_math(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1,224, 224, 3], name="input")

        conv = tf.nn.conv2d(x, tf.constant(np.random.random([3, 3, 3, 16]), dtype=tf.float32), strides=[1, 1, 1, 1], 
                                    padding="SAME")
        sub = tf.math.subtract(conv, tf.constant(np.random.random(16), dtype=tf.float32))
        realdiv = tf.realdiv(sub, tf.constant(np.random.random(16), dtype=tf.float32))

        mul = tf.math.multiply(realdiv, tf.constant(np.random.random(16), dtype=tf.float32))
        conv_add = tf.nn.bias_add(mul, tf.constant(np.random.random(16), dtype=tf.float32), name='bias_add')

        relu = tf.nn.relu(conv_add)
        identity = tf.identity(relu)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[identity.name.split(':')[0]])
        fixed_input = np.random.random([1,224, 224, 3])
        
        default_g = tf.Graph()
        with default_g.as_default():
            tf.import_graph_def(graph_def, name='')
        with tf.compat.v1.Session(graph=default_g) as sess:
            output = sess.run(['Identity:0'], {'input:0': fixed_input})

        post_graph = FuseConvWithMathOptimizer(graph_def).do_transformation()
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(post_graph, name='optimized')
        with tf.compat.v1.Session(graph=g) as sess:
            optimized_output = sess.run(['optimized/Identity:0'], {'optimized/input:0': fixed_input})

        converted = True
        for node in post_graph.node:
            if node.op.find("Sub") != -1:
                converted = False
        self.assertEqual(converted, True)

        self.assertEqual(np.allclose (output[0], optimized_output[0]), True)

if __name__ == '__main__':
    unittest.main()