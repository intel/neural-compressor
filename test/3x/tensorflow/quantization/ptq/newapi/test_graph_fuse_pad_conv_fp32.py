import os
import unittest

import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


class TestFoldPadConv(unittest.TestCase):
    @disable_random()
    def test_fold_pad_conv(self):
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
        relu = tf.nn.relu(normed, name="op_to_store")
        out_name = relu.name.split(":")[0]
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

            found_pad = False
            for i in qmodel.graph_def.node:
                if i.op == "Pad":
                    found_pad = True
                    break

            self.assertEqual(found_pad, False)

    @disable_random()
    def test_fold_non_const_pad_conv(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        vec = tf.raw_ops.DataFormatVecPermute(x=paddings, src_format="NHWC", dst_format="NHWC")
        x_pad = tf.pad(x, vec, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )
        relu = tf.nn.relu(normed, name="op_to_store")
        out_name = relu.name.split(":")[0]
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

            found_pad = False
            for i in qmodel.graph_def.node:
                if i.op == "Pad":
                    found_pad = True
                    break

            self.assertEqual(found_pad, False)


if __name__ == "__main__":
    unittest.main()
