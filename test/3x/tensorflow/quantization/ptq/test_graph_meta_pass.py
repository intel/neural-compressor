#
#  -*- coding: utf-8 -*-
#
import os
import unittest

import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


class TestMetaPass(unittest.TestCase):
    @disable_random()
    def test_tensorflow_graph_meta_pass_with_different_mode(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(top_relu, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )

        relu = tf.nn.relu(normed)
        sq = tf.squeeze(relu, [0])
        reshape = tf.reshape(sq, [729, 16])
        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [16, 729], initializer=tf.compat.v1.random_normal_initializer()
        )

        matmul = tf.matmul(reshape, conv_weights2)
        # normed2 = tf.compat.v1.layers.batch_normalization(matmul)
        bias = tf.compat.v1.get_variable("bias", [729], initializer=tf.compat.v1.random_normal_initializer())
        normed2 = tf.nn.bias_add(matmul, bias, name="bias_add")

        relu6 = tf.nn.relu6(normed2)
        reshape2 = tf.reshape(relu6, [1, 729, 729, 1], name="op_to_store")

        out_name = reshape2.name.split(":")[0]

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

            found_reshape = False
            for i in qmodel.graph_def.node:
                if i.op == "Reshape":
                    found_reshape = True
                    break

            self.assertEqual(found_reshape, True)

    @disable_random()
    def test_tensorflow_graph_meta_pass_with_same_mode(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(top_relu, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )

        relu = tf.nn.relu(normed)
        sq = tf.squeeze(relu, [0])
        reshape = tf.reshape(sq, [1, 27, 27, 16])
        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(reshape, conv_weights2, strides=[1, 2, 2, 1], padding="VALID")
        normed2 = (
            tf.keras.layers.BatchNormalization()(conv2)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv2)
        )

        relu6 = tf.nn.relu6(normed2, name="op_to_store")

        out_name = relu6.name.split(":")[0]

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

            quantize_count = 0
            dequantize_count = 0
            for i in qmodel.graph_def.node:
                if i.op == "QuantizeV2":
                    quantize_count += 1
                if i.op == "Dequantize":
                    dequantize_count += 1

            self.assertEqual(quantize_count, 1)
            self.assertEqual(dequantize_count, 1)

    @disable_random()
    def test_tensorflow_graph_meta_with_reshape_only(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(top_relu, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )

        relu = tf.nn.relu(normed)
        reshape = tf.reshape(relu, [1, 27, 27, 16])
        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(reshape, conv_weights2, strides=[1, 2, 2, 1], padding="VALID")
        normed2 = (
            tf.keras.layers.BatchNormalization()(conv2)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv2)
        )

        relu6 = tf.nn.relu6(normed2, name="op_to_store")

        out_name = relu6.name.split(":")[0]

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

            quantize_count = 0
            dequantize_count = 0
            for i in qmodel.graph_def.node:
                if i.op == "QuantizeV2":
                    quantize_count += 1
                if i.op == "Dequantize":
                    dequantize_count += 1

            self.assertEqual(quantize_count, 1)
            self.assertEqual(dequantize_count, 1)


if __name__ == "__main__":
    unittest.main()
