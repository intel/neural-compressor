#
#  -*- coding: utf-8 -*-
#
import os
import platform
import shutil
import unittest

import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util
from tensorflow.python.platform import gfile

from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2, version1_lt_version2


@unittest.skipIf(version1_lt_version2(tf.version.VERSION, "2.8.0"), "Only supports tf greater 2.7.0")
class TestItexEnabling(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        if version1_gte_version2(tf.version.VERSION, "2.8.0"):
            shutil.rmtree("workspace_1")
            shutil.rmtree("workspace_2")
            shutil.rmtree("workspace_3")

    @disable_random()
    def test_itex_convert_basic_default_device(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        add = tf.raw_ops.Add(x=normed, y=normed2, name="addv2")
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name="op_to_store")
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

            dequant_count = 0
            quantize_count = 0
            for i in qmodel.graph_def.node:
                if "min" in i.name or "max" in i.name:
                    self.assertEqual(i.op, "HostConst")
                if i.op == "HostConst":
                    self.assertTrue("min" in i.name or "max" in i.name)
                if i.op == "Dequantize":
                    dequant_count += 1
                if i.op == "QuantizeV2":
                    quantize_count += 1

            self.assertEqual(dequant_count, 5)
            self.assertEqual(quantize_count, 4)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, "2.8.0"), "Only supports tf greater 2.7.0")
    def test_itex_convert_basic_cpu(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        # relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        # relu2 = tf.nn.relu(normed2)
        add = tf.raw_ops.Add(x=normed, y=normed2, name="addv2")
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name="op_to_store")

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

            dequant_count = 0
            quantize_count = 0
            for i in qmodel.graph_def.node:
                if i.op == "Dequantize":
                    dequant_count += 1
                if i.op == "QuantizeV2":
                    quantize_count += 1

            self.assertEqual(dequant_count, 5)
            self.assertEqual(quantize_count, 4)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, "2.8.0"), "Only supports tf greater 2.7.0")
    def test_depthwiseconv2d_case(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.depthwise_conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="VALID")
        out_name = conv.name.split(":")[0]

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

            reshape_counter = 0
            for i in qmodel.graph_def.node:
                if i.op == "Reshape":
                    reshape_counter += 1

            self.assertEqual(reshape_counter, 2)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, "2.8.0"), "Only supports tf greater 2.7.0")
    def test_itex_convert_shared_y_pattern_normal_case(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 1, 1, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        add = tf.raw_ops.Add(x=top_relu, y=normed2, name="addv2")
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name="op_to_store")

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

            dequant_count = 0
            quantize_count = 0
            for i in qmodel.graph_def.node:
                if i.op == "Dequantize":
                    dequant_count += 1
                if i.op == "QuantizeV2":
                    quantize_count += 1

            self.assertEqual(dequant_count, 2)
            self.assertEqual(quantize_count, 2)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, "2.8.0"), "Only supports tf greater 2.7.0")
    def test_itex_convert_share_y_pattern_abnormal_case1(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        relu2 = tf.nn.relu(normed2)
        add = tf.raw_ops.Add(x=relu, y=relu2, name="addv2")
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name="op_to_store")

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

            dequant_count = 0
            quantize_count = 0
            for i in qmodel.graph_def.node:
                if i.op == "Dequantize":
                    dequant_count += 1
                if i.op == "QuantizeV2":
                    quantize_count += 1

            self.assertEqual(dequant_count, 4)
            self.assertEqual(quantize_count, 3)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, "2.8.0"), "Only supports tf greater 2.7.0")
    def test_itex_convert_share_y_pattern_abnormal_case2(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed)
        relu6 = tf.nn.relu6(relu, name="op_to_store")

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

            dequant_count = 0
            quantize_count = 0
            for i in qmodel.graph_def.node:
                if i.op == "Dequantize":
                    dequant_count += 1
                if i.op == "QuantizeV2":
                    quantize_count += 1

            self.assertEqual(dequant_count, 2)
            self.assertEqual(quantize_count, 2)


if __name__ == "__main__":
    unittest.main()