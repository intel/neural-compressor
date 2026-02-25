import os
import unittest

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

from neural_compressor.tensorflow.utils import disable_random


class TestDqCastFusion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.environ["FORCE_BF16"] = "1"

    @classmethod
    def tearDownClass(self):
        os.environ["FORCE_BF16"] = "0"

    @disable_random()
    def test_dq_all_outputs_bf16(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        conv_weights = tf.constant(np.random.random((1, 3, 16, 16)).astype(np.float32), name="y")
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        conv_reshape1 = tf.reshape(conv, [1, 28, 27, 16])
        conv_reshape2 = tf.reshape(conv, [1, 28, 27, 16])
        out = tf.math.add(conv_reshape1, conv_reshape2, name="op_to_store")
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

        found_cast = False
        for node in qmodel.graph_def.node:
            if node.op == "Cast":
                found_cast = True
                break

        self.assertEqual(found_cast, False)


if __name__ == "__main__":
    unittest.main()
