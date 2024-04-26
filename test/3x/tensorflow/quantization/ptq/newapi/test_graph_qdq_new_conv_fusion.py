#
#  -*- coding: utf-8 -*-
#
import logging
import os
import unittest

import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util
from tensorflow.python.framework import function

from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


class TestTensorflowNewQdqConvFusion(unittest.TestCase):
    @disable_random()
    def test_conv_biasadd_add_leakyrelu_fusion(self):
        logging.getLogger().info("test_conv_biasadd_add_leakyrelu_fusion")
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )
        conv2_weights = tf.compat.v1.get_variable(
            "weight_conv2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(x, conv2_weights, strides=[1, 2, 2, 1], padding="SAME")
        sumadd = tf.raw_ops.AddV2(x=normed, y=conv2, name="addv2")
        leaky_relu = tf.nn.leaky_relu(sumadd, name="op_to_store")

        out_name = leaky_relu.name.split(":")[0]
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
                if i.op == "LeakyRelu":
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, False)

    @disable_random()
    def test_resizebilinear_bf16_input(self):
        os.environ["FORCE_BF16"] = "1"
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
        resize_bili1 = tf.raw_ops.ResizeBilinear(images=leaky_relu, size=(28, 28))
        resize_bili2 = tf.raw_ops.ResizeBilinear(images=resize_bili1, size=(14, 14))
        out_name = resize_bili2.name.split(":")[0]
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

            bf16_matmul = False
            for i in qmodel.graph_def.node:
                if i.op == "BatchMatMulV2" and i.attr["T"].type == 14:
                    bf16_matmul = True

            self.assertEqual(bf16_matmul, True)
            os.environ["FORCE_BF16"] = "0"


if __name__ == "__main__":
    unittest.main()
