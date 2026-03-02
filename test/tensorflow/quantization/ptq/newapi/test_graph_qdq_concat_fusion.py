#
#
#  -*- coding: utf-8 -*-
import os
import unittest

import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

from neural_compressor.tensorflow.quantization.utils.quantize_graph.quantize_graph_for_intel_cpu import (
    QuantizeGraphForIntel,
)
from neural_compressor.tensorflow.quantization.utils.utility import read_graph
from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


class TestTensorflowQdqConcatFusion(unittest.TestCase):
    mb_model_url = (
        "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv3_fp32_pretrained_model.pb"
    )
    pb_path = "/tmp/.neural_compressor/inceptionv3_fp32.pb"

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path):
            os.system("mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(self.mb_model_url, self.pb_path))

    def test_tensorflow_concat_quantization(self):
        fp32_graph_def = read_graph(self.pb_path)

        from neural_compressor.tensorflow import Model, quantize_model
        from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

        dataset = DummyDataset(shape=(100, 299, 299, 3), label=True)
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

        found_quantized_concat_node = False
        target_concat_node_name = "v0/cg/incept_v3_a0/concat_eightbit_quantized_concatv2"
        from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer

        cur_graph = GraphAnalyzer()
        cur_graph.graph = qmodel.graph_def
        graph_info = cur_graph.parse_graph()
        found_quantized_concat_node = target_concat_node_name in graph_info

        self.assertEqual(found_quantized_concat_node, True)
        min_out, max_out = [], []
        for input_conv_name in graph_info[target_concat_node_name].node.input[:4]:
            min_freezed_out_name = graph_info[input_conv_name].node.input[-2]
            max_freezed_out_name = graph_info[input_conv_name].node.input[-1]
            min_freezed_out_value = (graph_info[min_freezed_out_name].node.attr["value"].tensor.float_val)[0]
            max_freezed_out_value = (graph_info[max_freezed_out_name].node.attr["value"].tensor.float_val)[0]
            min_out.append(min_freezed_out_value)
            max_out.append(max_freezed_out_value)

        self.assertEqual(len(set(min_out)), 1)
        self.assertEqual(len(set(max_out)), 1)

    @disable_random()
    def test_concat_with_different_input_type(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 128, 16], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [2, 2, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv_bias = tf.compat.v1.get_variable("bias", [16], initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.relu(x)
        sqrt = tf.math.sqrt(x)
        relu_sqrt = tf.nn.relu(sqrt)
        conv = tf.nn.conv2d(relu_sqrt, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name="last")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )

        relu = tf.nn.relu(normed)
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name="last")
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        concat = tf.concat([relu, conv_bias], 1)
        pool = tf.nn.avg_pool(concat, ksize=1, strides=[1, 2, 2, 1], name="avgpool", padding="SAME")
        final_node = tf.nn.relu(pool, name="op_to_store")
        out_name = final_node.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 128, 128, 16), label=True)
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

            quantized_concat = False
            for i in qmodel.graph_def.node:
                if i.op == "QuantizedConcatV2":
                    quantized_concat = True
            self.assertEqual(quantized_concat, False)

    @disable_random()
    def test_concat_with_same_input_type(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 128, 16], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [2, 2, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv_bias = tf.compat.v1.get_variable("bias", [16], initializer=tf.compat.v1.random_normal_initializer())
        conv1_bias = tf.compat.v1.get_variable("bias1", [16], initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.relu(x)
        sqrt = tf.math.sqrt(x)
        relu_sqrt = tf.nn.relu(sqrt)
        conv = tf.nn.conv2d(relu_sqrt, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name="last")
        conv_bias = tf.nn.bias_add(conv, conv_bias)
        relu1 = tf.nn.relu(conv_bias)

        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name="last")
        conv1_bias = tf.nn.bias_add(conv1, conv1_bias)
        relu2 = tf.nn.relu(conv1_bias)
        concat = tf.concat([relu1, relu2], 1)
        pool = tf.nn.avg_pool(concat, ksize=1, strides=[1, 2, 2, 1], name="avgpool", padding="SAME")
        final_node = tf.nn.relu(pool, name="op_to_store")
        out_name = final_node.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 128, 128, 16), label=True)
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

            quantized_concat = False
            for i in qmodel.graph_def.node:
                if i.op == "QuantizedConcatV2":
                    quantized_concat = True

            self.assertEqual(quantized_concat, True)

    @disable_random()
    def test_concat_with_qint8_and_fp32_input_type(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 128, 16], name="input")
        bias = tf.compat.v1.get_variable("bias", [16], initializer=tf.compat.v1.random_normal_initializer())

        bias_add = tf.nn.bias_add(x, bias)

        pool = tf.nn.avg_pool(x, ksize=1, strides=[1, 1, 1, 1], name="avgpool", padding="SAME")
        concat = tf.concat([bias_add, pool], 1)
        final_node = tf.nn.relu(concat, name="op_to_store")
        out_name = final_node.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import Model, quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 128, 128, 16), label=True)
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

            dtype = None
            quantized_concat = False
            from tensorflow.python.framework import dtypes

            for i in qmodel.graph_def.node:
                if i.op == "QuantizedConcatV2":
                    dtype = dtypes.DType(i.attr["T"].type)
                    quantized_concat = True

            self.assertEqual(quantized_concat, True)
            self.assertEqual(dtype, dtypes.qint8)


if __name__ == "__main__":
    unittest.main()
