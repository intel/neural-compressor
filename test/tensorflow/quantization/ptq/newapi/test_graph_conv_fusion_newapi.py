#
#  -*- coding: utf-8 -*-
#
import os
import unittest

import numpy as np
import tensorflow as tf
import yaml
from pkg_resources import parse_version
from tensorflow.compat.v1 import graph_util
from tensorflow.python.framework import function

from neural_compressor.tensorflow.algorithms.static_quant.tensorflow import TensorflowQuery
from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.fold_batch_norm import (
    FoldBatchNormNodesOptimizer,
)
from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.strip_unused_nodes import (
    StripUnusedNodesOptimizer,
)
from neural_compressor.tensorflow.quantization.utils.quantize_graph.qdq.optimize_qdq import OptimizeQDQGraph
from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


class TestConvBiasAddAddReluFusion(unittest.TestCase):
    @disable_random()
    def test_conv_single_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv1_weights = tf.compat.v1.get_variable(
            "weight_conv1", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv1 = tf.nn.conv2d(x_pad, conv1_weights, strides=[1, 2, 2, 1], padding="VALID")
        matmul_weights = tf.compat.v1.get_variable(
            "weight_matmul", [1, 28, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        matmul = tf.linalg.matmul(conv1, matmul_weights)
        conv2_weights = tf.compat.v1.get_variable(
            "weight_conv2", [7, 7, 32, 1], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(matmul, conv2_weights, strides=[1, 2, 2, 1], padding="VALID")
        leaky_relu = tf.nn.leaky_relu(conv2, name="op_to_store")

        out_name = leaky_relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import Model, quantize_model
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
            fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
            qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

            find_single_qconv = []
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv2D":
                    find_single_qconv.append(i.attr["fused_ops"].list.s == [b"Requantize"])

            self.assertEqual(find_single_qconv, [False, False])

    @disable_random()
    def test_spacetobatchnd_conv2d_batchtospacend_fusion(self):
        i = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        x = tf.space_to_batch_nd(i, block_shape=[2, 2], paddings=[[0, 0], [0, 0]])
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        y = tf.compat.v1.batch_to_space_nd(conv, block_shape=[2, 2], crops=[[0, 0], [0, 0]])
        out = tf.identity(y, name="op_to_store")
        out_name = out.name.split(":")[0]

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

            found_op = False
            for i in qmodel.graph_def.node:
                if i.op == "SpaceToBatchND" or i.op == "BatchToSpaceND":
                    found_op = True
                    break

            self.assertEqual(found_op, False)

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

            found_conv_fusion = True
            for i in qmodel.graph_def.node:
                if i.op == "Relu":
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, False)

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
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )

        relu6 = tf.nn.relu6(normed, name="op_to_store")

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

            found_conv_fusion = True
            for i in qmodel.graph_def.node:
                if i.op == "Relu6":
                    found_conv_fusion = False
                    break
            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_biasadd_swishf32_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )

        @function.Defun(tf.float32, func_name="swish_f32")
        def swish_f32(x):
            return tf.nn.silu(x, beta=1.0)

        swish = swish_f32(normed, name="swish_f32_output_node")

        out_name = swish.name.split(":")[0]
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

            found_conv_fusion = True
            for i in qmodel.graph_def.node:
                if i.op == "swish_f32":
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_addv2_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        conv1_weights = tf.compat.v1.get_variable(
            "weight_conv1", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1, 2, 2, 1], padding="SAME")
        conv2_weights = tf.compat.v1.get_variable(
            "weight_conv2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(x, conv2_weights, strides=[1, 2, 2, 1], padding="SAME")
        sumadd = tf.raw_ops.AddV2(x=conv1, y=conv2, name="addv2")

        out_name = sumadd.name.split(":")[0]
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op.find("QuantizedConv2D") != -1:
                    found_conv_fusion = True
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op.find("QuantizedConv2D") != -1:
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

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
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )
        # relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = (
            tf.keras.layers.BatchNormalization()(conv2)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv2)
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv2D" and i.attr["fused_ops"].list.s == [
                    b"BiasAdd",
                    b"Sum",
                    b"Relu",
                    b"Requantize",
                ]:
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
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )
        # relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = (
            tf.keras.layers.BatchNormalization()(conv2)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv2)
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
            found_conv_fusion = []
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv2D" and i.attr["fused_ops"].list.s == [b"BiasAdd", b"Requantize"]:
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
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
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
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
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

            quantized_pool_data_type = None
            quantized_conv_data_type = None
            for i in qmodel.graph_def.node:
                if i.op.find("QuantizedMaxPool") != -1:
                    quantized_pool_data_type = i.attr["T"].type
                if i.op.find("QuantizedConv2D") != -1:
                    quantized_conv_data_type = i.attr["Tinput"].type

            self.assertNotEqual(quantized_pool_data_type, None)
            self.assertEqual(quantized_pool_data_type, quantized_conv_data_type)

    @disable_random()
    def test_conv3d_addv2_relu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight_conv3d_1", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add = tf.raw_ops.AddV2(x=conv3d_1, y=tf.constant(np.random.randn(32), dtype=tf.float32), name="addv2")
        relu = tf.nn.relu(add)
        conv3d_2_weights = tf.compat.v1.get_variable(
            "weight_conv3d_2", [3, 3, 3, 32, 1], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_2 = tf.nn.conv3d(relu, conv3d_2_weights, strides=[1, 2, 2, 2, 1], padding="SAME")

        out_name = conv3d_2.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 128, 64, 64, 16), label=True)
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

            found_conv_sumadd_fusion = False
            found_conv_biasadd_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv3D":
                    if b"Sum" in i.attr["fused_ops"].list.s:
                        found_conv_sumadd_fusion = True
                    if i.attr["fused_ops"].list.s == [b"BiasAdd", b"Relu", b"Requantize"]:
                        found_conv_biasadd_fusion = True

            self.assertEqual(found_conv_sumadd_fusion, False)
            self.assertEqual(found_conv_biasadd_fusion, True)

    # conv2d + dummybiasadd + addv2 fusion
    @disable_random()
    def test_conv_add_addn_non_const_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv2d_1_weights = tf.compat.v1.get_variable(
            "weight1", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2d_1 = tf.nn.conv2d(top_relu, conv2d_1_weights, strides=[1, 2, 2, 1], padding="SAME")
        conv2d_2_weights = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2d_2 = tf.nn.conv2d(top_relu, conv2d_2_weights, strides=[1, 2, 2, 1], padding="SAME")
        add_1 = tf.raw_ops.AddV2(x=conv2d_1, y=conv2d_2, name="addv2_1")
        conv2d_3_weights = tf.compat.v1.get_variable(
            "weight3", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2d_3 = tf.nn.conv2d(top_relu, conv2d_3_weights, strides=[1, 2, 2, 1], padding="SAME")
        add = tf.raw_ops.AddV2(x=add_1, y=conv2d_3, name="addv2_2")
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv2D" and i.attr["fused_ops"].list.s == [
                    b"BiasAdd",
                    b"Sum",
                    b"Requantize",
                ]:
                    found_conv_fusion = True

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    @unittest.skipIf(
        tf.__version__ not in ["2.11.0202242", "2.11.0202250", "2.11.0202317", "2.11.0202323"],
        "deconv2d quantization only support 2.11",
    )
    def test_deconv2d_biasadd_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 2, 2, 1], name="input")
        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 1, 1], initializer=tf.compat.v1.random_normal_initializer()
        )

        conv2 = tf.nn.conv2d_transpose(
            x, conv_weights2, output_shape=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME"
        )

        normed2 = tf.nn.bias_add(conv2, tf.constant([3.0]))
        out = tf.identity(normed2)

        out_name = out.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 2, 2, 1), label=True)
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

            found_deconv2d_fusion = False
            for i in qmodel.graph_def.node:
                if i.op.find("_FusedQuantizedDeconv2D") != -1:
                    found_deconv2d_fusion = True
                    break

            self.assertEqual(found_deconv2d_fusion, True)

    @disable_random()
    @unittest.skipIf(
        tf.__version__ not in ["2.11.0202242", "2.11.0202250", "2.11.0202317", "2.11.0202323"],
        "deconv2d quantization only support 2.11",
    )
    def test_single_deconv2d_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 2, 2, 1], name="input")
        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 1, 1], initializer=tf.compat.v1.random_normal_initializer()
        )

        conv2 = tf.nn.conv2d_transpose(
            x, conv_weights2, output_shape=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME"
        )

        out = tf.identity(conv2)

        out_name = out.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 2, 2, 1), label=True)
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

            found_deconv2d_fusion = False
            for i in qmodel.graph_def.node:
                if i.op.find("_FusedQuantizedDeconv2D") != -1:
                    found_deconv2d_fusion = True
                    break

            self.assertEqual(found_deconv2d_fusion, True)

    @disable_random()
    @unittest.skipIf(
        tf.__version__ not in ["2.11.0202242", "2.11.0202250", "2.11.0202317", "2.11.0202323"],
        "deconv2d quantization only support 2.11",
    )
    def test_deconv3d_biasadd_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 2, 2, 2, 1], name="input")
        conv3d_weights = tf.compat.v1.get_variable(
            "weight_conv3d_1", [3, 3, 3, 1, 1], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d = tf.nn.conv3d_transpose(
            x, conv3d_weights, output_shape=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding="SAME"
        )

        normed2 = tf.nn.bias_add(conv3d, tf.constant([3.0]))
        out = tf.identity(normed2)

        out_name = out.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 2, 2, 2, 1), label=True)
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

            found_deconv3d_fusion = False
            for i in qmodel.graph_def.node:
                if i.op.find("_FusedQuantizedDeconv3D") != -1:
                    found_deconv3d_fusion = True
                    break

            self.assertEqual(found_deconv3d_fusion, True)

    @disable_random()
    @unittest.skipIf(
        tf.__version__ not in ["2.11.0202242", "2.11.0202250", "2.11.0202317", "2.11.0202323"],
        "deconv2d quantization only support 2.11",
    )
    def test_single_deconv3d_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 2, 2, 2, 1], name="input")
        conv3d_weights = tf.compat.v1.get_variable(
            "weight_conv3d_1", [3, 3, 3, 1, 1], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d = tf.nn.conv3d_transpose(
            x, conv3d_weights, output_shape=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding="SAME"
        )

        out = tf.identity(conv3d)

        out_name = out.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 2, 2, 2, 1), label=True)
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

            found_deconv3d_fusion = False
            for i in qmodel.graph_def.node:
                if i.op.find("_FusedQuantizedDeconv3D") != -1:
                    found_deconv3d_fusion = True
                    break

            self.assertEqual(found_deconv3d_fusion, True)


if __name__ == "__main__":
    unittest.main()
