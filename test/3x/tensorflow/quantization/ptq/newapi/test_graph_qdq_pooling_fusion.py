#
#  -*- coding: utf-8 -*-
#
import os
import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import yaml
from tensorflow.compat.v1 import graph_util
from tensorflow.python.framework import dtypes

from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


class TestGraphQDQPoolingFusion(unittest.TestCase):
    @disable_random()
    def test_qdq_maxpool_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 30, 30, 1], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [2, 2, 1, 1], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv_bias = tf.compat.v1.get_variable("bias", [1], initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.relu(x)
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name="last")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )

        relu = tf.nn.relu(normed)
        relu2 = tf.nn.relu(relu)
        pool = tf.nn.max_pool(relu2, ksize=1, strides=[1, 2, 2, 1], name="maxpool", padding="SAME")
        conv1 = tf.nn.conv2d(pool, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name="last")
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        x = tf.nn.relu(conv_bias)
        final_node = tf.nn.relu(x, name="op_to_store")

        out_name = final_node.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 30, 30, 1), label=True)
            calib_dataloader = BaseDataLoader(dataset)
            quant_config = StaticQuantConfig()
            qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

            found_quantized_maxpool = False
            for i in qmodel.graph_def.node:
                if i.op == "QuantizedMaxPool":
                    found_quantized_maxpool = True
                    break

            self.assertEqual(found_quantized_maxpool, True)

    @disable_random()
    def test_qdq_avgpool_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 30, 30, 1], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [2, 2, 1, 1], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv_bias = tf.compat.v1.get_variable("bias", [1], initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.relu(x)
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name="last")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )

        relu = tf.nn.relu(normed)
        relu2 = tf.nn.relu(relu)
        pool = tf.nn.avg_pool(relu2, ksize=1, strides=[1, 2, 2, 1], name="avgpool", padding="SAME")
        conv1 = tf.nn.conv2d(pool, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name="last")
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        x = tf.nn.relu(conv_bias)
        final_node = tf.nn.relu(x, name="op_to_store")

        out_name = final_node.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 30, 30, 1), label=True)
            calib_dataloader = BaseDataLoader(dataset)
            quant_config = StaticQuantConfig()
            qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

            found_quantized_avgpool = False
            for i in qmodel.graph_def.node:
                if i.op == "QuantizedAvgPool":
                    found_quantized_avgpool = True
                    break

            self.assertEqual(found_quantized_avgpool, True)


if __name__ == "__main__":
    unittest.main()
