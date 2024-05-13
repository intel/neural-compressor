import os
import unittest

import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

import neural_compressor
from neural_compressor.tensorflow.algorithms.static_quant.tensorflow import TensorflowQuery
from neural_compressor.tensorflow.quantization.utils.quantize_graph.quantize_graph_for_intel_cpu import (
    QuantizeGraphForIntel,
)
from neural_compressor.tensorflow.quantization.utils.quantize_graph_common import QuantizeGraphHelper
from neural_compressor.tensorflow.quantization.utils.transform_graph.bias_correction import BiasCorrection
from neural_compressor.tensorflow.utils import disable_random


class TestBiasCorrectionNewApi(unittest.TestCase):
    @disable_random()
    def test_bias_correction_new_api(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        if tf.version.VERSION <= "2.1.0":
            x = tf.nn.relu(x)
        conv1_weights = tf.compat.v1.get_variable(
            "weights1", [3, 3, 3, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        normed = tf.nn.bias_add(
            conv1,
            tf.constant(
                [
                    3.0,
                    1.2,
                    1.0,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    0,
                    1,
                    4.0,
                    5.2,
                    8.1,
                    2,
                    4,
                    5,
                    8,
                    9,
                    10,
                    12,
                    11,
                    2,
                    5.0,
                    7.2,
                    3.2,
                    3,
                    4,
                    5,
                    7,
                    8,
                ]
            ),
        )
        relu1 = tf.nn.relu(normed, name="Relu_1")
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[relu1.name.split(":")[0]]
            )
            fp32_graph_def = QuantizeGraphHelper.remove_training_nodes(
                output_graph_def, protected_nodes=[relu1.name.split(":")[0]]
            )
            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 224, 224, 3), label=True)
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

        correct_graph_def = BiasCorrection(
            qmodel.graph_def, fp32_graph_def, "weight_empirical", True
        ).do_transformation()

        self.assertEqual(len(correct_graph_def.node), len(qmodel.graph_def.node))


class TestBiasCorrectionOldApi(unittest.TestCase):
    @disable_random()
    def test_bias_correction_old_api(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        if tf.version.VERSION <= "2.1.0":
            x = tf.nn.relu(x)
        conv_weights = tf.compat.v1.get_variable(
            "weights", [3, 3, 3, 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        normed = tf.nn.bias_add(
            conv,
            tf.constant(
                [
                    3.0,
                    1.2,
                    1.0,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    0,
                    1,
                    4.0,
                    5.2,
                    8.1,
                    2,
                    4,
                    5,
                    8,
                    9,
                    10,
                    12,
                    11,
                    2,
                    5.0,
                    7.2,
                    3.2,
                    3,
                    4,
                    5,
                    7,
                    8,
                ]
            ),
        )
        relu = tf.nn.relu(normed, name="Relu_0")
        op_wise_sequences = TensorflowQuery(
            local_config_file=neural_compressor.__path__[0] + "/tensorflow/algorithms/static_quant/tensorflow.yaml"
        ).get_eightbit_patterns()

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[relu.name.split(":")[0]]
            )
            output_graph_def = QuantizeGraphHelper.remove_training_nodes(
                output_graph_def, protected_nodes=[relu.name.split(":")[0]]
            )
            inputs = [x.name.split(":")[0]]
            outputs = [relu.name.split(":")[0]]
            op_wise_config = {
                "Conv2D": (False, "minmax", False, 7.0),
            }

            int8_graph_def, _, _ = QuantizeGraphForIntel(
                output_graph_def, inputs, outputs, op_wise_config, op_wise_sequences, "cpu"
            ).do_transform()

        correct_graph_def = BiasCorrection(int8_graph_def, output_graph_def).do_transformation()
        self.assertEqual(len(correct_graph_def.node), len(int8_graph_def.node))


if __name__ == "__main__":
    unittest.main()
