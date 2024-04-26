#
#  -*- coding: utf-8 -*-
#
import os
import platform
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import graph_util

import neural_compressor
from neural_compressor.tensorflow import quantize_model
from neural_compressor.tensorflow.algorithms.static_quant.tensorflow import TensorflowQuery
from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.fold_batch_norm import (
    FoldBatchNormNodesOptimizer,
)
from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.strip_unused_nodes import (
    StripUnusedNodesOptimizer,
)
from neural_compressor.tensorflow.quantization.utils.quantize_graph.quantize_graph_for_intel_cpu import (
    QuantizeGraphForIntel,
)
from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset, disable_random, version1_gte_version2


class TestConvBiasAddAddReluFusion(unittest.TestCase):

    @disable_random()
    def test_conv_relu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        relu = tf.nn.relu(conv)

        relu6 = tf.nn.relu6(relu, name="op_to_store")

        out_name = relu6.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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

            found_conv_fusion = True
            for i in qmodel.graph_def.node:
                if i.op == "Relu":
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, False)

    @disable_random()
    @unittest.skipIf(tf.__version__ < "2.0", "does not support on 1.15up3")
    def test_depthwiseconv_biasadd_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.depthwise_conv2d(x_pad, conv_weights, strides=[1, 1, 1, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv, name="op_to_store")
        )
        out_name = normed.name.split(":")[0]

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "QuantizedDepthwiseConv2DWithBias":
                    found_conv_fusion = True
                    break
                if i.op == "QuantizedDepthwiseConv2D" and version1_gte_version2(tf.__version__, "2.16.1"):
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_depthwiseconv_biasadd_fusion_with_negative_input(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.depthwise_conv2d(x_pad, conv_weights, strides=[1, 1, 1, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv, name="op_to_store")
        )
        out_name = normed.name.split(":")[0]

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "QuantizedDepthwiseConv2DWithBias":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, False)

    @unittest.skipUnless(
        bool(tf.version.VERSION.find("1.15.0-up") != -1 or tf.version.VERSION >= "2.1.0"),
        "not supported the current tf version.",
    )
    @disable_random()
    def test_conv_biasadd_relu6_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv, name="op_to_store")
        )

        relu6 = tf.nn.relu6(normed, name="op_to_store")

        out_name = relu6.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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

            found_conv_fusion = True
            for i in qmodel.graph_def.node:
                if i.op == "Relu6":
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_biasadd_add_relu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.nn.bias_add(conv2, tf.constant([3.0, 1.2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 12, 2, 3, 4]))
        relu = tf.nn.relu(normed2 + tf.constant([3.0]))
        relu6 = tf.nn.relu6(relu, name="op_to_store")

        out_name = relu6.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op.find("QuantizedConv2D") != -1:
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_squeeze_biasadd_relu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        squeeze = tf.squeeze(conv2)
        normed2 = tf.nn.bias_add(conv2, tf.constant([3.0, 1.2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 12, 2, 3, 4]))
        relu = tf.nn.relu(normed2)
        identity = tf.identity(relu, name="op_to_store")

        out_name = identity.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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

            correct_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "QuantizedConv2DWithBiasAndReluAndRequantize":
                    correct_conv_fusion = True
                    break

            self.assertEqual(correct_conv_fusion, True)

    @disable_random()
    def test_conv_biasadd_addv2_relu_fallback_fusion_1(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.leaky_relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv, name="op_to_store")
        )
        # relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv2)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv2, name="op_to_store")
        )
        # relu2 = tf.nn.relu(normed2)
        add = tf.raw_ops.AddV2(x=normed, y=normed2, name="addv2")
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name="op_to_store")

        out_name = relu6.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "QuantizedConv2DWithBiasAndRequantize":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_biasadd_addv2_relu_fallback_fusion_2(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv, name="op_to_store")
        )
        # relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv2)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv2, name="op_to_store")
        )
        # relu2 = tf.nn.relu(normed2)
        add = tf.raw_ops.AddV2(x=normed, y=normed2, name="addv2")
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name="op_to_store")

        out_name = relu6.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_fusion_with_last_matmul(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        # paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        # x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(top_relu, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv, name="op_to_store")
        )

        relu = tf.nn.relu(normed)
        pooling = tf.nn.max_pool(relu, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        reshape = tf.reshape(pooling, [-1, 3136])

        y_data = np.random.random([3136, 1])

        y = tf.constant(y_data, dtype=tf.float32, shape=[3136, 1])
        z = tf.matmul(reshape, y)
        relu1 = tf.nn.relu(z)
        y_data_1 = np.random.random([1, 1])
        y_1 = tf.constant(y_data_1, dtype=tf.float32, shape=[1, 1])

        z_2nd_matmul = tf.matmul(relu1, y_1)
        relu6 = tf.nn.relu6(z_2nd_matmul, name="op_to_store")

        out_name = relu6.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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
                    break

            self.assertEqual(quantize_v2_count, 1)

    @disable_random()
    def test_conv_fusion_with_last_conv(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(top_relu, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv, name="op_to_store")
        )

        relu = tf.nn.relu(normed)
        pooling = tf.nn.max_pool(relu, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        conv_weights_2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(pooling, conv_weights_2, strides=[1, 2, 2, 1], padding="VALID")
        conv_weights_3 = tf.compat.v1.get_variable(
            "weight3", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        relu2 = tf.nn.relu(conv2)
        conv3 = tf.nn.conv2d(relu2, conv_weights_3, strides=[1, 2, 2, 1], padding="VALID")

        relu3 = tf.nn.relu(conv3)
        relu6 = tf.nn.relu6(relu3, name="op_to_store")

        out_name = relu6.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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
                    break

            self.assertEqual(quantize_v2_count, 1)

    @disable_random()
    def test_conv_fusion_with_max_pooling(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")

        relu = tf.nn.relu(x)
        pooling = tf.nn.max_pool(relu, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        conv_weights = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(pooling, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        biasadd = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv, name="op_to_store")
        )
        out_name = biasadd.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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

            quantized_pool_data_type = None
            quantized_conv_data_type = None
            for i in qmodel.graph_def.node:
                if i.op.find("QuantizedMaxPool") != -1:
                    quantized_pool_data_type = i.attr["T"].type
                if i.op.find("QuantizedConv2D") != -1:
                    quantized_conv_data_type = i.attr["Tinput"].type

            self.assertNotEqual(quantized_pool_data_type, None)
            self.assertEqual(quantized_pool_data_type, quantized_conv_data_type)


class TestGraphConvFusion(unittest.TestCase):
    rn50_fp32_pb_url = (
        "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb"
    )
    pb_path = "/tmp/.neural_compressor/resnet50_fp32_pretrained_model.pb"
    platform = platform.system().lower()
    if platform == "windows":
        pb_path = "C:\\tmp\.neural_compressor\\resnet50_fp32_pretrained_model.pb"
    inputs = ["input"]
    outputs = ["predict"]

    op_wise_config = {
        "v0/resnet_v13/conv14/conv2d/Conv2D": (False, "minmax", False, 7.0),
        "v0/resnet_v13/conv11/conv2d/Conv2D": (False, "minmax", False, 7.0),
        "v0/resnet_v17/conv27/conv2d/Conv2D": (False, "minmax", False, 7.0),
    }

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path):
            if self.platform == "linux":
                os.system(
                    "mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(self.rn50_fp32_pb_url, self.pb_path)
                )
            elif self.platform == "windows":
                os.system("md C:\\tmp\.neural_compressor && cd C:\\tmp\.neural_compressor")
                from urllib import request

                request.urlretrieve(self.rn50_fp32_pb_url)
        self.input_graph = tf.compat.v1.GraphDef()
        with open(self.pb_path, "rb") as f:
            self.input_graph.ParseFromString(f.read())

    def test_conv_biasadd_relu_fusion(self):
        tf.compat.v1.disable_eager_execution()

        self._tmp_graph_def = graph_util.remove_training_nodes(self.input_graph, self.outputs)

        self._tmp_graph_def = StripUnusedNodesOptimizer(
            self._tmp_graph_def, self.inputs, self.outputs
        ).do_transformation()

        self._tmp_graph_def = FoldBatchNormNodesOptimizer(self._tmp_graph_def).do_transformation()
        op_wise_sequences = TensorflowQuery(
            local_config_file=neural_compressor.__path__[0] + "/tensorflow/algorithms/static_quant/tensorflow.yaml"
        ).get_eightbit_patterns()

        output_graph, _, _ = QuantizeGraphForIntel(
            self._tmp_graph_def, self.inputs, self.outputs, self.op_wise_config, op_wise_sequences, "cpu"
        ).do_transform()

        node_name_type_mapping = {}
        for i in output_graph.node:
            node_name_type_mapping[i.name] = i.op

        should_disable_sum_node_name = "v0/resnet_v17/conv27/conv2d/Conv2D_eightbit_quantized_conv"
        should_enable_sum_node_name = "v0/resnet_v13/conv11/conv2d/Conv2D_eightbit_quantized_conv"
        should_disable_sum_flag = (
            should_disable_sum_node_name in node_name_type_mapping
            and node_name_type_mapping[should_disable_sum_node_name] == "QuantizedConv2DWithBias"
        )
        should_enable_sum_flag = (
            should_enable_sum_node_name in node_name_type_mapping
            and node_name_type_mapping[should_enable_sum_node_name] == "QuantizedConv2DWithBiasSumAndRelu"
        )
        self.assertEqual(should_enable_sum_flag, True)
        self.assertEqual(should_disable_sum_flag, True)


if __name__ == "__main__":
    unittest.main()
