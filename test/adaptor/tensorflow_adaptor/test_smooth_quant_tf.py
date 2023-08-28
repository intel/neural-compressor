import unittest

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import graph_util

from neural_compressor.adaptor.tf_utils.util import disable_random
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.data.dataloaders.dataloader import DataLoader
from neural_compressor.quantization import fit
from neural_compressor.utils.utility import set_random_seed


class TestSmoothQuantTF(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @disable_random()
    def test_conv_sq(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        add = tf.raw_ops.Add(x=normed, y=normed2, name="addv2")
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name="op_to_store")

        out_name = relu6.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

        set_random_seed(9527)
        config = PostTrainingQuantConfig(
            quant_level=1,
            recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5}},
            calibration_sampling_size=[500],
        )

        from neural_compressor.data import Datasets

        dataset = Datasets("tensorflow")["dummy"](shape=(100, 56, 56, 16), label=True)
        dataloader = DataLoader(framework="tensorflow", dataset=dataset, batch_size=1)
        from neural_compressor import Metric

        top1 = Metric(name="topk", k=1)
        output_graph = fit(
            model=output_graph_def,
            conf=config,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_metric=top1,
        )

        mul_count = 0
        for i in output_graph.graph_def.node:
            if i.op == "Mul":
                mul_count += 1

        self.assertEqual(mul_count, 2)

    @disable_random()
    def test_sq_matmul(self):
        x_data = np.random.rand(1024, 1024).astype(np.float32)
        y_data = np.random.rand(1024, 1024).astype(np.float32)
        import tensorflow.compat.v1 as tf

        x = tf.placeholder(tf.float32, shape=[1024, 1024], name="x")
        y = tf.constant(y_data, dtype=tf.float32, shape=[1024, 1024])
        z = tf.matmul(x, y)
        bias = np.random.rand(1024).astype(np.float32)
        z = tf.nn.bias_add(z, bias)
        z = tf.nn.relu(z, name="op_to_store")

        with tf.Session() as sess:
            sess.run(z, feed_dict={x: x_data, y: y_data})
            output_graph_def = sess.graph.as_graph_def()

        set_random_seed(9527)
        config = PostTrainingQuantConfig(
            quant_level=1,
            recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5}},
            calibration_sampling_size=[1024],
        )

        from neural_compressor.data import Datasets

        dataset = Datasets("tensorflow")["dummy"](shape=(1024, 1024), label=True)
        dataloader = DataLoader(framework="tensorflow", dataset=dataset, batch_size=1024)
        from neural_compressor import Metric

        top1 = Metric(name="topk", k=1)
        output_graph = fit(
            model=output_graph_def,
            conf=config,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_metric=top1,
        )

        mul_count = 0
        for i in output_graph.graph_def.node:
            if i.op == "Mul":
                mul_count += 1

        self.assertEqual(mul_count, 1)

    @disable_random()
    def test_sq_conv_matmul(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv1_weights = tf.compat.v1.get_variable(
            "weight_conv1", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv1 = tf.nn.conv2d(x_pad, conv1_weights, strides=[1, 2, 2, 1], padding="VALID")
        matmul_weights = tf.compat.v1.get_variable(
            "weight_matmul", [28 * 28 * 16, 7 * 7 * 32], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv1_reshaped = tf.reshape(conv1, shape=[-1, 28 * 28 * 16])
        matmul = tf.matmul(conv1_reshaped, matmul_weights)
        reshape = tf.reshape(matmul, (1, 7, 7, 32))
        conv2_weights = tf.compat.v1.get_variable(
            "weight_conv2", [7, 7, 32, 1], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv2 = tf.nn.conv2d(reshape, conv2_weights, strides=[1, 2, 2, 1], padding="VALID")
        leaky_relu = tf.nn.leaky_relu(conv2, name="op_to_store")

        out_name = leaky_relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

        set_random_seed(9527)
        config = PostTrainingQuantConfig(
            quant_level=1,
            recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": 0.6}},
            calibration_sampling_size=[500],
        )

        from neural_compressor.data import Datasets

        dataset = Datasets("tensorflow")["dummy"](shape=(100, 56, 56, 16), label=True)
        dataloader = DataLoader(framework="tensorflow", dataset=dataset)
        from neural_compressor import Metric

        top1 = Metric(name="topk", k=1)
        output_graph = fit(
            model=output_graph_def,
            conf=config,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_metric=top1,
        )

        mul_count = 0
        for i in output_graph.graph_def.node:
            if i.op == "Mul":
                mul_count += 1

        self.assertEqual(mul_count, 3)


if __name__ == "__main__":
    unittest.main()
