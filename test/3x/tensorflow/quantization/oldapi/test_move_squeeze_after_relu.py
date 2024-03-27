import imp
import os
import unittest

import numpy as np
import tensorflow as tf
from numpy.core.fromnumeric import squeeze
from tensorflow.compat.v1 import graph_util

from neural_compressor.tensorflow.utils import disable_random


class TestSwitchOptimizer(unittest.TestCase):
    @disable_random()
    def test_move_squeeze_after_relu(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")

        conv_weights = tf.constant(np.random.random((3, 3, 16, 16)).astype(np.float32), name="y")
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        sque = tf.squeeze(conv)
        bias_add = tf.nn.bias_add(sque, tf.constant(np.random.random((16)).astype(np.float32), name="bias_add"))
        relu = tf.nn.relu(bias_add, name="op_to_store")
        out_name = relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.move_squeeze_after_relu import (
            MoveSqueezeAfterReluOptimizer,
        )

        convert_graph = MoveSqueezeAfterReluOptimizer(output_graph_def).do_transformation()

        move_relu = False
        for node in convert_graph.node:
            if node.op == "Squeeze" and node.input[0] == "op_to_store":
                move_relu = True

        self.assertEqual(move_relu, True)

    @disable_random()
    def test_move_squeeze_after_relu_with_outputs(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")

        conv_weights = tf.constant(np.random.random((3, 3, 16, 16)).astype(np.float32), name="y")
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        sque = tf.squeeze(conv)
        bias_add = tf.nn.bias_add(sque, tf.constant(np.random.random((16)).astype(np.float32), name="bias_add"))
        relu = tf.nn.relu(bias_add, name="relu")
        smax = tf.nn.softmax(relu, name="op_to_store")
        out_name = smax.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.move_squeeze_after_relu import (
            MoveSqueezeAfterReluOptimizer,
        )

        convert_graph = MoveSqueezeAfterReluOptimizer(output_graph_def).do_transformation()

        move_relu = False
        for node in convert_graph.node:
            if node.op == "Squeeze" and node.input[0] == "op_to_store":
                move_relu = True

        self.assertEqual(move_relu, False)

    @disable_random()
    def test_move_squeeze_after_sigmoid(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")

        conv_weights = tf.constant(np.random.random((3, 3, 16, 16)).astype(np.float32), name="y")
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        sque = tf.squeeze(conv)
        bias_add = tf.nn.bias_add(sque, tf.constant(np.random.random((16)).astype(np.float32), name="bias_add"))
        sigmoid = tf.nn.sigmoid(bias_add, name="op_to_store")
        out_name = sigmoid.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.move_squeeze_after_relu import (
            MoveSqueezeAfterReluOptimizer,
        )

        convert_graph = MoveSqueezeAfterReluOptimizer(output_graph_def).do_transformation()

        move_sigmoid = False
        for node in convert_graph.node:
            if node.op == "Squeeze" and node.input[0] == "op_to_store":
                move_sigmoid = True

        self.assertEqual(move_sigmoid, True)

    @disable_random()
    def test_move_reshape_after_relu(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")

        conv_weights = tf.constant(np.random.random((3, 3, 16, 16)).astype(np.float32), name="y")
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        bias_add = tf.nn.bias_add(conv, tf.constant(np.random.random((16)).astype(np.float32), name="bias_add"))
        reshape = tf.reshape(bias_add, [1, 28, 28, 16])
        relu = tf.nn.relu(reshape, name="op_to_store")
        out_name = relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.move_squeeze_after_relu import (
            MoveSqueezeAfterReluOptimizer,
        )

        convert_graph = MoveSqueezeAfterReluOptimizer(output_graph_def).do_transformation()

        move_relu = False
        for node in convert_graph.node:
            if node.op == "Reshape" and node.input[0] == "op_to_store":
                move_relu = True

        self.assertEqual(move_relu, True)


if __name__ == "__main__":
    unittest.main()
