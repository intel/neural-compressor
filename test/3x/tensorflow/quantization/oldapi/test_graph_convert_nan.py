#
#  -*- coding: utf-8 -*-
#

import unittest

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.python.framework import tensor_util

from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.convert_nan_to_random import (
    ConvertNanToRandom,
)
from neural_compressor.tensorflow.utils import disable_random


class TestNanConvert(unittest.TestCase):
    @disable_random()
    def test_convert_nan_to_float(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv = tf.nn.conv2d(
            x, tf.constant(np.random.random([3, 3, 3, 16]), dtype=tf.float32), strides=[1, 1, 1, 1], padding="SAME"
        )
        sub = tf.math.subtract(conv, tf.constant(np.random.random(16), dtype=tf.float32))
        realdiv = tf.realdiv(sub, tf.constant(np.random.random(16), dtype=tf.float32))

        mul = tf.math.multiply(realdiv, tf.constant(np.random.random(16), dtype=tf.float32))

        conv_add = tf.nn.bias_add(mul, tf.constant(np.full((16,), np.nan), dtype=tf.float32), name="bias_add")

        relu = tf.nn.relu(conv_add)
        identity = tf.identity(relu)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[identity.name.split(":")[0]]
            )

        post_graph = ConvertNanToRandom(graph_def).do_transformation()

        converted = True
        for node in post_graph.node:
            if node.op.find("Const") != -1:
                const_content = tensor_util.MakeNdarray(node.attr["value"].tensor)
                if np.any(np.isnan(const_content)):
                    converted = False
        self.assertEqual(converted, True)


if __name__ == "__main__":
    unittest.main()
