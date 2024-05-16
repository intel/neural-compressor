import os
import unittest

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


class TestPostCSEOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.enable_s8 = bool(tf.version.VERSION.find("1.15.0-up") != -1 or tf.version.VERSION >= "2.1.0")

    @disable_random()
    def test_post_cse(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        x = tf.nn.relu(x)
        xw = tf.constant(np.random.random((2, 2, 16, 16)), dtype=tf.float32, name="y")
        x = tf.nn.conv2d(input=x, filters=xw, strides=[1, 1, 1, 1], padding="VALID")

        y = tf.constant(np.random.random((1, 55, 55, 16)), dtype=tf.float32, name="y")

        z = tf.math.add(x, y, name="add")

        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(z, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )
        relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(z, conv_weights2, strides=[1, 2, 2, 1], padding="VALID")
        normed2 = (
            tf.keras.layers.BatchNormalization()(conv2)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv2)
        )
        relu2 = tf.nn.relu(normed2)
        add = tf.math.add(relu, relu2, name="op_to_store")
        out_name = add.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )
            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 56, 56, 16), label=True)
            calib_dataloader = BaseDataLoader(dataset)
            quant_config = {
                "static_quant": {
                    "global": {
                        "weight_dtype": "int8",
                        "weight_sym": True,
                        "weight_granularity": "per_tensor",
                        "weight_algorithm": "minmax",
                    },
                }
            }
            qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

            quantize_v2_count = 0
            for i in qmodel.graph_def.node:
                if i.op == "QuantizeV2":
                    quantize_v2_count += 1

            if self.enable_s8:
                self.assertEqual(quantize_v2_count, 2)
            else:
                self.assertEqual(quantize_v2_count, 1)

    @disable_random()
    def test_post_cse2(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        x = tf.nn.relu(x)
        xw = tf.constant(np.random.random((2, 2, 16, 16)), dtype=tf.float32, name="y")
        x = tf.nn.conv2d(input=x, filters=xw, strides=[1, 1, 1, 1], padding="VALID")

        y = tf.constant(np.random.random((1, 55, 55, 16)), dtype=tf.float32, name="y")

        z = tf.math.add(x, y, name="add")

        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(z, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )
        relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(z, conv_weights2, strides=[1, 2, 2, 1], padding="VALID")
        normed2 = (
            tf.keras.layers.BatchNormalization()(conv2)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv2)
        )
        relu2 = tf.nn.relu(normed2)
        add = tf.math.add(relu, relu2)
        ones_const = tf.constant(1, dtype=tf.float32)
        ones_const2 = tf.constant(1, dtype=tf.float32)
        mul1 = tf.math.multiply(add, ones_const)
        mul2 = tf.math.multiply(mul1, ones_const)
        mul3 = tf.math.multiply(mul2, ones_const2, name="op_to_store")
        out_name = mul3.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 56, 56, 16), label=True)
            calib_dataloader = BaseDataLoader(dataset)
            quant_config = {
                "static_quant": {
                    "global": {
                        "weight_dtype": "int8",
                        "weight_sym": True,
                        "weight_granularity": "per_tensor",
                        "weight_algorithm": "minmax",
                    },
                }
            }
            qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

            quantize_v2_count = 0
            for i in qmodel.graph_def.node:
                if i.op == "QuantizeV2":
                    quantize_v2_count += 1

            if self.enable_s8:
                self.assertEqual(quantize_v2_count, 2)
            else:
                self.assertEqual(quantize_v2_count, 1)


if __name__ == "__main__":
    unittest.main()
