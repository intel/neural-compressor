#
#  -*- coding: utf-8 -*-
#
import logging
import os
import unittest

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

from neural_compressor.tensorflow.algorithms.static_quant.tensorflow import TensorflowQuery
from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


class TestConvRequantizedFusionNewAPI(unittest.TestCase):
    @disable_random()
    def test_conv_biasadd_relu6_fusion(self):
        logging.getLogger().info("test_conv_biasadd_relu6_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight0", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
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
    def test_single_conv3d_fusion(self):
        logging.getLogger().info("test_single_conv3d_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 64, 64, 64, 1], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight1", [4, 4, 4, 1, 64], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv3d(x_pad, conv_weights, strides=[1, 2, 2, 2, 1], padding="VALID", name="op_to_store")

        out_name = conv.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 64, 64, 64, 1), label=True)
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
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_biasadd_fusion(self):
        logging.getLogger().info("test_conv3d_biasadd_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 64, 64, 64, 1], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight2", [4, 4, 4, 1, 64], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv3d(x_pad, conv_weights, strides=[1, 2, 2, 2, 1], padding="VALID")
        relu6 = tf.nn.relu6(conv, name="op_to_store")
        out_name = relu6.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 64, 64, 64, 1), label=True)
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
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_add_relu_fusion(self):
        logging.getLogger().info("test_conv3d_add_relu_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 64, 64, 64, 1], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight6", [4, 4, 4, 1, 64], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv1_weights = tf.compat.v1.get_variable(
            "weight7", [4, 4, 4, 1, 64], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv3d(x, conv_weights, strides=[1, 2, 2, 2, 1], padding="VALID")
        conv1 = tf.nn.conv3d(x, conv1_weights, strides=[1, 2, 2, 2, 1], padding="VALID")
        add = conv + conv1
        relu = tf.nn.relu(add)

        out_name = relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import Model, quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 64, 64, 64, 1), label=True)
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv3D" and i.attr["fused_ops"].list.s == [b"BiasAdd", b"Dequantize"]:
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv2d_biasadd_elu_fusion(self):
        input = tf.compat.v1.placeholder(tf.float32, shape=(1, 3, 3, 1), name="input")
        weight = tf.compat.v1.constant(np.random.random((2, 2, 1, 1)).astype(np.float32), name="weight")
        bias = tf.constant(np.random.random((1)), name="bias", dtype=tf.float32)
        conv = tf.nn.conv2d(input=input, filters=weight, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        bias_add = tf.nn.bias_add(conv, bias, name="bias_add")
        res = tf.nn.elu(bias_add, name="res")
        output = tf.nn.softmax(res, name="op_to_store")

        out_name = output.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import Model, quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 3, 3, 1), label=True)
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
            self.assertNotEqual(qmodel, None)

            elu_fused = False
            for node in qmodel.graph_def.node:
                if node.name == "conv_eightbit_requantize_dequantize":
                    if b"Elu" in node.attr["fused_ops"].list.s:
                        elu_fused = True

            self.assertEqual(elu_fused, True)

    @disable_random()
    def test_conv3d_add_const_fusion(self):
        logging.getLogger().info("test_conv3d_add_const_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 64, 64, 64, 1], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight11", [4, 4, 4, 1, 64], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv3d(x, conv_weights, strides=[1, 2, 2, 2, 1], padding="VALID")
        add = conv + tf.constant(
            [
                [
                    [
                        [
                            [
                                0.000015179151887423359,
                                0.000022200847524800338,
                                -0.000009995766049541999,
                                -0.0000022956028260523453,
                                0.000008830029400996864,
                                0.0000017190360495078494,
                                0.000019561824956326745,
                                0.00014721050683874637,
                                -0.000005871841494808905,
                                0.000004377178811409976,
                                -0.000006191140982991783,
                                0.000009258330464945175,
                                -0.000009839599442784674,
                                0.000008547322067897767,
                                0.000004629391241905978,
                                2.345327061448188e-7,
                                0.000015179151887423359,
                                0.000022200847524800338,
                                -0.000009995766049541999,
                                -0.0000022956028260523453,
                                0.000008830029400996864,
                                0.0000017190360495078494,
                                0.000019561824956326745,
                                0.00014721050683874637,
                                -0.000005871841494808905,
                                0.000004377178811409976,
                                -0.000006191140982991783,
                                0.000009258330464945175,
                                -0.000009839599442784674,
                                0.000008547322067897767,
                                0.000004629391241905978,
                                2.345327061448188e-7,
                                0.000015179151887423359,
                                0.000022200847524800338,
                                -0.000009995766049541999,
                                -0.0000022956028260523453,
                                0.000008830029400996864,
                                0.0000017190360495078494,
                                0.000019561824956326745,
                                0.00014721050683874637,
                                -0.000005871841494808905,
                                0.000004377178811409976,
                                -0.000006191140982991783,
                                0.000009258330464945175,
                                -0.000009839599442784674,
                                0.000008547322067897767,
                                0.000004629391241905978,
                                2.345327061448188e-7,
                                0.000015179151887423359,
                                0.000022200847524800338,
                                -0.000009995766049541999,
                                -0.0000022956028260523453,
                                0.000008830029400996864,
                                0.0000017190360495078494,
                                0.000019561824956326745,
                                0.00014721050683874637,
                                -0.000005871841494808905,
                                0.000004377178811409976,
                                -0.000006191140982991783,
                                0.000009258330464945175,
                                -0.000009839599442784674,
                                0.000008547322067897767,
                                0.000004629391241905978,
                                2.345327061448188e-7,
                            ]
                        ]
                    ]
                ]
            ]
        )

        out_name = add.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import Model, quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 64, 64, 64, 1), label=True)
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

            found_conv_fusion = True
            for i in qmodel.graph_def.node:
                if i.op == "AddV2":
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_add_add_fusion(self):
        logging.getLogger().info("test_conv_add_add_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight12", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )
        add = normed + tf.constant(np.random.randn(16), dtype=tf.float32)
        relu6 = tf.nn.relu6(add, name="op_to_store")

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
                if i.op == "Add":
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_single_conv2d_fusion(self):
        logging.getLogger().info("test_single_conv2d_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight13", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv2D":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_add_addn_const_relu_fusion(self):
        logging.getLogger().info("test_conv3d_add_addn_const_relu_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight14", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add_1 = tf.raw_ops.AddV2(x=conv3d_1, y=tf.constant(np.random.randn(32), dtype=tf.float32), name="addv2")
        var = tf.compat.v1.get_variable(
            "add_y", [1, 64, 32, 32, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        add = tf.raw_ops.AddV2(x=add_1, y=var, name="addv2_1")
        relu = tf.nn.relu(add)
        out_name = relu.name.split(":")[0]
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
                    if str(b"Sum") in str(i.attr["fused_ops"].list.s):
                        found_conv_sumadd_fusion = True
                    if str(i.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Sum", b"Relu"]):
                        found_conv_biasadd_fusion = True

            self.assertEqual(found_conv_sumadd_fusion, False)
            self.assertEqual(found_conv_biasadd_fusion, False)

    @disable_random()
    def test_conv3d_add_const_addn_relu_fusion(self):
        logging.getLogger().info("test_conv3d_add_const_addn_relu_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight15", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add_1 = tf.raw_ops.AddV2(x=conv3d_1, y=tf.constant(np.random.randn(32), dtype=tf.float32), name="addv2_2")
        conv3d_2_weights = tf.compat.v1.get_variable(
            "weight16", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_2 = tf.nn.conv3d(top_relu, conv3d_2_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add = tf.raw_ops.AddV2(x=add_1, y=conv3d_2, name="addv2_3")
        relu = tf.nn.relu(add)
        out_name = relu.name.split(":")[0]
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
                    if str(b"Sum") in str(i.attr["fused_ops"].list.s):
                        found_conv_sumadd_fusion = True
                    if str(i.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Sum", b"Relu"]):
                        found_conv_biasadd_fusion = True

            self.assertEqual(found_conv_sumadd_fusion, True)
            self.assertEqual(found_conv_biasadd_fusion, False)

    @disable_random()
    def test_conv3d_add_addn_fusion(self):
        logging.getLogger().info("test_conv3d_add_addn_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight15", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add_1 = tf.raw_ops.AddV2(x=conv3d_1, y=tf.constant(np.random.randn(32), dtype=tf.float32), name="addv2_4")
        conv3d_2_weights = tf.compat.v1.get_variable(
            "weight16", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_2 = tf.nn.conv3d(top_relu, conv3d_2_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add = tf.raw_ops.AddV2(x=add_1, y=conv3d_2, name="addv2_5")
        out_name = add.name.split(":")[0]
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_add_addn_relu_fusion(self):
        logging.getLogger().info("test_conv3d_add_addn_relu_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight17", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_2_weights = tf.compat.v1.get_variable(
            "weight18", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        conv3d_2 = tf.nn.conv3d(top_relu, conv3d_2_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add_1 = tf.raw_ops.AddV2(x=conv3d_1, y=conv3d_2, name="addv2_6")
        conv3d_3_weights = tf.compat.v1.get_variable(
            "weight19", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_2 = tf.nn.conv3d(top_relu, conv3d_3_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add = tf.raw_ops.AddV2(x=add_1, y=conv3d_2, name="addv2_7")
        relu = tf.nn.relu(add)
        out_name = relu.name.split(":")[0]
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

            found_relu_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "Relu":
                    found_relu_fusion = True

            self.assertEqual(found_relu_fusion, True)

    @disable_random()
    def test_conv3d_relu_fusion(self):
        logging.getLogger().info("test_conv3d_relu_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 64, 64, 64, 1], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight20", [4, 4, 4, 1, 64], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv3d(x_pad, conv_weights, strides=[1, 2, 2, 2, 1], padding="VALID")
        relu = tf.nn.leaky_relu(conv)

        out_name = relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 64, 64, 64, 1), label=True)
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
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_add_fusion(self):
        logging.getLogger().info("test_conv3d_add_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight21", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        conv3d_2_weights = tf.compat.v1.get_variable(
            "weight22", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_2 = tf.nn.conv3d(top_relu, conv3d_2_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add = tf.raw_ops.AddV2(x=conv3d_1, y=conv3d_2, name="addv2_8")
        out_name = add.name.split(":")[0]
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_add_const_addn_relu_requantize_fusion(self):
        logging.getLogger().info("test_conv3d_add_const_addn_relu_requantize_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight23", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        y_const = tf.constant(np.random.randn(1, 1, 1, 1, 32), dtype=tf.float32)
        add_1 = tf.raw_ops.AddV2(x=conv3d_1, y=y_const, name="addv2_9")
        conv3d_2_weights = tf.compat.v1.get_variable(
            "weight24", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_2 = tf.nn.conv3d(top_relu, conv3d_2_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add_2 = tf.raw_ops.AddV2(x=add_1, y=conv3d_2, name="addv2_10")
        relu = tf.nn.relu(add_2)
        out_name = relu.name.split(":")[0]
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
                    if str(b"Sum") in str(i.attr["fused_ops"].list.s):
                        found_conv_sumadd_fusion = True
                    if str(i.attr["fused_ops"].list.s) == str([b"BiasAdd", b"Sum", b"Relu", b"Requantize"]):
                        found_conv_biasadd_fusion = True

            self.assertEqual(found_conv_sumadd_fusion, True)
            self.assertEqual(found_conv_biasadd_fusion, True)

    @disable_random()
    def test_conv3d_add_const_addn_fusion(self):
        logging.getLogger().info("test_conv3d_add_const_addn_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight25", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        y_const = tf.constant(np.random.randn(1, 1, 1, 1, 32), dtype=tf.float32)
        add_1 = tf.raw_ops.AddV2(x=conv3d_1, y=y_const, name="addv2_11")
        conv3d_2_weights = tf.compat.v1.get_variable(
            "weight26", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_2 = tf.nn.conv3d(top_relu, conv3d_2_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add_2 = tf.raw_ops.AddV2(x=add_1, y=conv3d_2, name="addv2_12")
        out_name = add_2.name.split(":")[0]
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_add_no_relu_fusion(self):
        logging.getLogger().info("test_conv3d_add_no_relu_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight27", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        y_const = tf.constant(np.random.randn(1, 1, 1, 1, 32), dtype=tf.float32)
        add = tf.raw_ops.AddV2(x=conv3d_1, y=y_const, name="addv2_13")
        pooling = tf.nn.max_pool(add, ksize=1, strides=[1, 2, 2, 2, 1], padding="SAME")
        out_name = pooling.name.split(":")[0]
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

            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_add_const_relu_fusion(self):
        logging.getLogger().info("test_conv3d_add_const_relu_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight28", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        y_const = tf.constant(np.random.randn(1, 1, 1, 1, 32), dtype=tf.float32)
        add = tf.raw_ops.AddV2(x=conv3d_1, y=y_const, name="addv2_10")
        relu = tf.nn.relu(add)
        out_name = relu.name.split(":")[0]
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

            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv2d_add_const_leakyrelu_add_fusion(self):
        logging.getLogger().info("test_conv2d_add_const_leakyrelu_add_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv2d_1_weights = tf.compat.v1.get_variable(
            "weight29", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2d_1 = tf.nn.conv2d(top_relu, conv2d_1_weights, strides=[1, 2, 2, 1], padding="SAME")
        y_const = tf.constant(np.random.randn(16), dtype=tf.float32)
        add_1 = tf.raw_ops.AddV2(x=conv2d_1, y=y_const, name="addv2_11")
        relu = tf.nn.leaky_relu(add_1)
        conv2d_2_weights = tf.compat.v1.get_variable(
            "weight30", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2d_2 = tf.nn.conv2d(top_relu, conv2d_2_weights, strides=[1, 2, 2, 1], padding="SAME")
        add_2 = tf.raw_ops.AddV2(x=relu, y=conv2d_2, name="addv2_12")
        out_name = add_2.name.split(":")[0]
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
                if i.op == "_FusedQuantizedConv2D":
                    found_conv_fusion = True

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_add_const_leakyrelu_add_fusion(self):
        logging.getLogger().info("test_conv3d_add_const_leakyrelu_add_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight31", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        y_const = tf.constant(np.random.randn(1, 1, 1, 1, 32), dtype=tf.float32)
        add_1 = tf.raw_ops.AddV2(x=conv3d_1, y=y_const, name="addv2_13")
        relu = tf.nn.leaky_relu(add_1)
        conv3d_2_weights = tf.compat.v1.get_variable(
            "weight32", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_2 = tf.nn.conv3d(top_relu, conv3d_2_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add_2 = tf.raw_ops.AddV2(x=relu, y=conv3d_2, name="addv2_14")
        out_name = add_2.name.split(":")[0]
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_add_addn_non_const_fusion(self):
        logging.getLogger().info("test_conv3d_add_addn_non_const_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_1_weights = tf.compat.v1.get_variable(
            "weight33", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        conv3d_2_weights = tf.compat.v1.get_variable(
            "weight34", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_2 = tf.nn.conv3d(top_relu, conv3d_2_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add_1 = tf.raw_ops.AddV2(x=conv3d_1, y=conv3d_2, name="addv2_15")
        conv3d_3_weights = tf.compat.v1.get_variable(
            "weight35", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d_3 = tf.nn.conv3d(top_relu, conv3d_3_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add = tf.raw_ops.AddV2(x=add_1, y=conv3d_3, name="addv2_16")
        out_name = add.name.split(":")[0]
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv3d_add_const_elu_add_fusion(self):
        logging.getLogger().info("test_conv3d_add_const_elufusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        top_relu = tf.nn.relu(x_pad)
        conv3d_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 3, 16, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv3d = tf.nn.conv3d(top_relu, conv3d_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        y_const = tf.constant(np.random.randn(1, 1, 1, 1, 32), dtype=tf.float32)
        add = tf.raw_ops.AddV2(x=conv3d, y=y_const, name="addv2")
        elu = tf.nn.elu(add)
        output = tf.nn.softmax(elu, name="op_to_store")
        out_name = output.name.split(":")[0]
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

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedConv3D":
                    found_conv_fusion = True

            self.assertEqual(found_conv_fusion, True)


if __name__ == "__main__":
    unittest.main()
