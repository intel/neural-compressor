import math
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import graph_util

from neural_compressor.common import set_random_seed
from neural_compressor.tensorflow import SmoothQuantConfig, StaticQuantConfig, get_default_sq_config, quantize_model
from neural_compressor.tensorflow.utils import DummyDataset, disable_random, version1_gte_version2


def build_conv_graph():
    tf.compat.v1.disable_eager_execution()
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

    conv_weights2 = tf.compat.v1.get_variable(
        "weight2", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
    )
    conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
    normed2 = (
        tf.keras.layers.BatchNormalization()(conv2)
        if version1_gte_version2(tf.__version__, "2.16.1")
        else tf.compat.v1.layers.batch_normalization(conv2)
    )
    add = tf.raw_ops.Add(x=normed, y=normed2, name="addv2")

    relu = tf.nn.relu(add)
    relu6 = tf.nn.relu6(relu, name="op_to_store")

    out_name = relu6.name.split(":")[0]

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
        )
        return output_graph_def


class MyDataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = math.ceil(len(dataset) / self.batch_size)

    def __iter__(self):
        images_list = []
        labels_list = []
        for _, (images, labels) in enumerate(self.dataset):
            images = np.expand_dims(images, axis=0)
            labels = np.expand_dims(labels, axis=0)
            images_list.append(images[0])
            labels_list.append(labels[0])
            if self.batch_size == len(images_list):
                yield (images_list, labels_list)
                images_list = []
                labels_list = []

    def __len__(self):
        return self.length


class TestSmoothQuantTF3xNewApi(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.conv_graph = build_conv_graph()

    @classmethod
    def tearDownClass(self):
        pass

    def test_conv(self):
        set_random_seed(9527)
        quant_config = SmoothQuantConfig(alpha=0.5)
        dataset = DummyDataset(shape=(100, 56, 56, 16), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset, batch_size=1)
        q_model = quantize_model(self.conv_graph, quant_config, calib_dataloader, calib_iteration=500)

        mul_count = 0
        for i in q_model.graph_def.node:
            if i.op == "Mul":
                mul_count += 1

        self.assertEqual(mul_count, 2)

    def test_sq_from_class_beginner(self):
        set_random_seed(9527)
        quant_config = get_default_sq_config()
        dataset = DummyDataset(shape=(100, 56, 56, 16), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset, batch_size=1)
        q_model = quantize_model(self.conv_graph, quant_config, calib_dataloader, calib_iteration=500)

        mul_count = 0
        for i in q_model.graph_def.node:
            if i.op == "Mul":
                mul_count += 1

        self.assertEqual(mul_count, 2)

    def test_sq_from_dict_beginner(self):
        quant_config = {
            "smooth_quant": {
                "global": {
                    "alpha": 0.5,
                },
            }
        }
        dataset = DummyDataset(shape=(100, 56, 56, 16), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset, batch_size=1)
        q_model = quantize_model(self.conv_graph, quant_config, calib_dataloader, calib_iteration=500)

        mul_count = 0
        for i in q_model.graph_def.node:
            if i.op == "Mul":
                mul_count += 1

        self.assertEqual(mul_count, 2)

    def test_sq_completed_workflow(self):
        x_data = np.random.rand(1024, 1024).astype(np.float32)
        y_data = np.random.rand(1024, 1024).astype(np.float32)
        import tensorflow.compat.v1 as tf

        with tf.Session(graph=tf.Graph()) as sess:
            x = tf.placeholder(tf.float32, shape=[1024, 1024], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[1024, 1024])
            z = tf.matmul(x, y)
            bias = np.random.rand(1024).astype(np.float32)
            z = tf.nn.bias_add(z, bias)
            z = tf.nn.relu(z, name="op_to_store")
            sess.run(z, feed_dict={x: x_data, y: y_data})
            output_graph_def = sess.graph.as_graph_def()

        set_random_seed(9527)
        sq_config = SmoothQuantConfig(alpha=0.5)
        static_config = StaticQuantConfig()
        dataset = DummyDataset(shape=(1024, 1024), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset, batch_size=1024)
        q_model = quantize_model(output_graph_def, [sq_config, static_config], calib_dataloader, calib_iteration=500)

        mul_count = 0
        quantized = False
        for i in q_model.graph_def.node:
            if i.op == "Mul":
                mul_count += 1
            if "quantize" in i.op:
                quantized = True

        self.assertEqual(mul_count, 1)
        self.assertEqual(quantized, True)

    @disable_random()
    def test_matmul(self):
        x_data = np.random.rand(1024, 1024).astype(np.float32)
        y_data = np.random.rand(1024, 1024).astype(np.float32)
        import tensorflow.compat.v1 as tf

        with tf.Session(graph=tf.Graph()) as sess:
            x = tf.placeholder(tf.float32, shape=[1024, 1024], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[1024, 1024])
            z = tf.matmul(x, y)
            bias = np.random.rand(1024).astype(np.float32)
            z = tf.nn.bias_add(z, bias)
            z = tf.nn.relu(z, name="op_to_store")
            sess.run(z, feed_dict={x: x_data, y: y_data})
            output_graph_def = sess.graph.as_graph_def()

        set_random_seed(9527)
        quant_config = SmoothQuantConfig(alpha=0.5)
        dataset = DummyDataset(shape=(1024, 1024), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset, batch_size=1024)
        q_model = quantize_model(output_graph_def, quant_config, calib_dataloader, calib_iteration=1)

        mul_count = 0
        for i in q_model.graph_def.node:
            if i.op == "Mul":
                mul_count += 1

        self.assertEqual(mul_count, 1)

    @disable_random()
    def test_conv_matmul(self):
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
        quant_config = SmoothQuantConfig(alpha=0.6)
        dataset = DummyDataset(shape=(100, 56, 56, 16), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset, batch_size=1)
        q_model = quantize_model(output_graph_def, quant_config, calib_dataloader, calib_iteration=500)

        mul_count = 0
        for i in q_model.graph_def.node:
            if i.op == "Mul":
                mul_count += 1

        self.assertEqual(mul_count, 3)


if __name__ == "__main__":
    unittest.main()
