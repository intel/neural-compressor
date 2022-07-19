#
#  -*- coding: utf-8 -*-
#
import unittest
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fuse_column_wise_mul import FuseColumnWiseMulOptimizer
from neural_compressor.adaptor.tf_utils.util import disable_random


class TestColumnWiseMulFusion(unittest.TestCase):

    @disable_random()
    def test_conv_mul_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.constant(np.random.random((3, 3, 16, 16)), dtype=tf.float32)
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        mul_tensor = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7], dtype=tf.float32)
        mul = tf.math.multiply(conv, mul_tensor)
        relu = tf.nn.relu(mul)
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            output_graph_def = FuseColumnWiseMulOptimizer(output_graph_def).do_transformation()

            found_mul = False

            for i in output_graph_def.node:
                if i.op == 'Mul':
                    found_mul = True
                    break

            self.assertEqual(found_mul, False)


if __name__ == '__main__':
    unittest.main()
