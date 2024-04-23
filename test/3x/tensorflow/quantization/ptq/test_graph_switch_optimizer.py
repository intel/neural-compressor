import imp
import os
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.python.ops import control_flow_ops

from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


class TestSwitchOptimizer(unittest.TestCase):
    @disable_random()
    def test_switch_optimizer(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        y = tf.compat.v1.placeholder_with_default(True, [], name="place_true")

        conv_weights = tf.constant(np.random.random((3, 3, 16, 16)).astype(np.float32), name="y")
        _, switch_true = control_flow_ops.switch(conv_weights, y)
        conv = tf.nn.conv2d(x_pad, switch_true, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )
        relu = tf.nn.relu(normed, name="op_to_store")
        out_name = relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.switch_optimizer import (
            SwitchOptimizer,
        )

        convert_graph = SwitchOptimizer(output_graph_def).do_transformation()
        found_switch = False
        for node in convert_graph.node:
            if node.op == "Switch":
                found_switch = True
        self.assertEqual(found_switch, False)

    @disable_random()
    def test_switch_optimizer_with_const_boolean(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        place = tf.constant(True)
        y = tf.compat.v1.placeholder_with_default(place, [], name="place_true")
        conv_weights = tf.constant(np.random.random((3, 3, 16, 16)).astype(np.float32), name="y")
        _, switch_true = control_flow_ops.switch(conv_weights, y)
        conv = tf.nn.conv2d(x_pad, switch_true, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )
        relu = tf.nn.relu(normed, name="op_to_store")
        out_name = relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.convert_placeholder_to_const import (
            ConvertPlaceholderToConst,
        )
        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.switch_optimizer import (
            SwitchOptimizer,
        )

        convert_graph = ConvertPlaceholderToConst(output_graph_def).do_transformation()
        convert_graph = SwitchOptimizer(convert_graph).do_transformation()

        found_switch = False
        for node in convert_graph.node:
            if node.op == "Switch":
                found_switch = True

        self.assertEqual(found_switch, False)

    @disable_random()
    def test_switch_optimizer_invalid(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        y = tf.compat.v1.placeholder_with_default(True, [], name="place_true")

        conv_weights = tf.constant(np.random.random((3, 3, 16, 16)).astype(np.float32), name="y")
        switch_false, _ = control_flow_ops.switch(conv_weights, y)
        conv = tf.nn.conv2d(x_pad, switch_false, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )
        relu = tf.nn.relu(normed, name="op_to_store")
        out_name = relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.switch_optimizer import (
            SwitchOptimizer,
        )

        convert_graph = SwitchOptimizer(output_graph_def).do_transformation()

        found_switch = False
        for node in convert_graph.node:
            if node.op == "Switch":
                found_switch = True

        self.assertEqual(found_switch, True)


if __name__ == "__main__":
    unittest.main()
