import imp
import os
import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import yaml
from numpy.core.fromnumeric import squeeze
from tensorflow.compat.v1 import graph_util

from neural_compressor.tensorflow.utils import disable_random


class TestFuseReshapeTransposeOptimizer(unittest.TestCase):
    @disable_random()
    def test_fuse_enter_reshape_transpose(self):
        x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
        y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
        y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
        enter = tf.raw_ops.Enter(data=y, frame_name="test")
        enter_perm = tf.raw_ops.Enter(data=[1, 0], frame_name="test", is_constant=True)
        transpose = tf.transpose(enter, perm=enter_perm)
        enter_reshape = tf.raw_ops.Enter(data=[2, 2], frame_name="test", is_constant=True)
        reshape = tf.reshape(transpose, enter_reshape)
        x_enter = tf.raw_ops.Enter(data=x, frame_name="test")
        z = tf.raw_ops.MatMul(a=x_enter, b=reshape, name="matmul_1")
        z = tf.raw_ops.Exit(data=z)
        found_quantized_matmul = True
        found_transpose = False
        found_reshape = False

        with tf.Session() as sess:
            sess.run(z, feed_dict={x: x_data, y: y_data})
            fp32_graph_def = sess.graph.as_graph_def()

            from neural_compressor.tensorflow import Model, quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(2, 2), label=True)
            calib_dataloader = BaseDataLoader(dataset, batch_size=2)
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

            for i in qmodel.graph_def.node:
                if i.op == "MatMul":
                    found_quantized_matmul = False
                if i.op == "Transpose":
                    found_transpose = True
                if i.op == "Reshape":
                    found_reshape = True

            self.assertEqual(found_quantized_matmul, True)
            self.assertEqual(found_transpose, False)
            self.assertEqual(found_reshape, False)

    @disable_random()
    def test_fuse_reshape_transpose(self):
        x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
        y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
        y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
        transpose = tf.transpose(y, perm=[1, 0])
        reshape = tf.reshape(transpose, [2, 2])
        z = tf.raw_ops.MatMul(a=x, b=reshape, name="matmul_2")
        z = tf.nn.bias_add(z, [1, 2], name="op_to_store")
        found_quantized_matmul = True
        found_transpose = False
        found_reshape = False

        with tf.Session() as sess:
            sess.run(z, feed_dict={x: x_data, y: y_data})
            fp32_graph_def = sess.graph.as_graph_def()

            from neural_compressor.tensorflow import Model, quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(2, 2), label=True)
            calib_dataloader = BaseDataLoader(dataset, batch_size=2)
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

            for i in qmodel.graph_def.node:
                if i.op == "MatMul":
                    found_quantized_matmul = False
                if i.op == "Transpose":
                    found_transpose = True
                if i.op == "Reshape":
                    found_reshape = True

            self.assertEqual(found_quantized_matmul, True)
            self.assertEqual(found_transpose, False)
            self.assertEqual(found_reshape, False)


if __name__ == "__main__":
    unittest.main()
