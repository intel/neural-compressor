#
#
#  -*- coding: utf-8 -*-
import os
import platform
import unittest

import tensorflow as tf
from tensorflow.compat.v1 import graph_util

import neural_compressor
from neural_compressor.tensorflow.algorithms.static_quant.tensorflow import TensorflowQuery
from neural_compressor.tensorflow.quantization.utils.quantize_graph.quantize_graph_for_intel_cpu import (
    QuantizeGraphForIntel,
)
from neural_compressor.tensorflow.quantization.utils.utility import read_graph
from neural_compressor.tensorflow.utils import disable_random


class TestTensorflowConcat(unittest.TestCase):
    mb_model_url = (
        "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv3_fp32_pretrained_model.pb"
    )
    pb_path = "/tmp/.neural_compressor/inceptionv3_fp32.pb"
    platform = platform.system().lower()
    if platform == "windows":
        pb_path = "C:\\tmp\\.neural_compressor\\inceptionv3_fp32.pb"

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path) and self.platform == "linux":
            os.system("mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(self.mb_model_url, self.pb_path))
        self.op_wise_sequences = TensorflowQuery(
            local_config_file=neural_compressor.__path__[0] + "/tensorflow/algorithms/static_quant/tensorflow.yaml"
        ).get_eightbit_patterns()

    @unittest.skipIf(tf.__version__ < "2.0", "does not support on 1.15up3")
    def test_tensorflow_concat_quantization(self):
        from neural_compressor.tensorflow import quantize_model
        from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

        fp32_graph_def = read_graph(self.pb_path)

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
        qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

        found_quantized_concat_node = False
        target_concat_node_name = "v0/cg/incept_v3_a0/concat_eightbit_quantized_concatv2"
        from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer

        cur_graph = GraphAnalyzer()
        cur_graph.graph = qmodel.graph_def
        graph_info = cur_graph.parse_graph()
        found_quantized_concat_node = target_concat_node_name in graph_info

        self.assertEqual(found_quantized_concat_node, True)

    @disable_random()
    def test_concat_with_different_input_type(self):
        from neural_compressor.tensorflow import quantize_model
        from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset, version1_gte_version2

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
        final_node = tf.nn.relu(concat, name="op_to_store")
        out_name = final_node.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

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


if __name__ == "__main__":
    unittest.main()
