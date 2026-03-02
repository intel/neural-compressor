import os
import unittest

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

from neural_compressor.tensorflow.utils import disable_random


class TestSwitchOptimizer(unittest.TestCase):
    @disable_random()
    def test_expanddims_optimizer(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")

        conv_weights = tf.constant(np.random.random((3, 16, 16)).astype(np.float32), name="y")
        conv_weights_expand = tf.expand_dims(conv_weights, axis=0, name="expanddims")
        conv = tf.nn.conv2d(x_pad, conv_weights_expand, strides=[1, 2, 2, 1], padding="VALID")
        out_name = conv.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.expanddims_optimizer import (
            ExpandDimsOptimizer,
        )

        convert_graph = ExpandDimsOptimizer(output_graph_def).do_transformation()
        handle_expanddims = True
        for node in convert_graph.node:
            if node.op == "Conv2D" and node.input[1] == "ExpandDims":
                handle_expanddims = False
                break
        self.assertEqual(handle_expanddims, True)


if __name__ == "__main__":
    unittest.main()
