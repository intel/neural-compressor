#
#  -*- coding: utf-8 -*-
#
import os
import platform
import unittest

import tensorflow as tf

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.quantization.utils.utility import get_input_output_node_names, read_graph


def build_fake_model_1():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        dataset = tf.data.Dataset.range(10)
        ds_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        iter_tensors = ds_iterator.get_next()
        iter_tensors -= tf.compat.v1.constant([5], dtype=tf.int64)
        final_node = tf.nn.relu(iter_tensors, name="op_to_store")
        sess.run(tf.compat.v1.global_variables_initializer())
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=sess, input_graph_def=sess.graph_def, output_node_names=[final_node.name.split(":")[0]]
        )

        with tf.io.gfile.GFile("model_1.pb", mode="wb") as f:
            f.write(constant_graph.SerializeToString())


def build_fake_model_2():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        final_node = tf.no_op()
        sess.run(tf.compat.v1.global_variables_initializer())
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=sess, input_graph_def=sess.graph_def, output_node_names=[final_node.name.split(":")[0]]
        )

        with tf.io.gfile.GFile("model_2.pb", mode="wb") as f:
            f.write(constant_graph.SerializeToString())


def build_fake_model_3():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x = [1, 2, 3]
        final_node = tf.Assert(tf.less_equal(tf.reduce_max(x), 3), x)
        sess.run(tf.compat.v1.global_variables_initializer())
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=sess, input_graph_def=sess.graph_def, output_node_names=[final_node.name.split(":")[0]]
        )

        with tf.io.gfile.GFile("model_3.pb", mode="wb") as f:
            f.write(constant_graph.SerializeToString())


class TestGraphInputOutputDetection(unittest.TestCase):
    tf.compat.v1.disable_v2_behavior()
    mb_fp32_pb_url = (
        "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb"
    )
    pb_path = "/tmp/.neural_compressor/mobilenet_fp32.pb"
    platform = platform.system().lower()
    if platform == "windows":
        pb_path = "C:\\tmp\\.neural_compressor\\mobilenet_fp32.pb"
    inputs = ["input"]
    outputs = ["MobilenetV1/Predictions/Reshape_1"]

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path):
            if self.platform == "linux":
                os.system(
                    "mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(self.mb_fp32_pb_url, self.pb_path)
                )
            elif self.platform == "windows":
                os.system("md C:\\tmp\.neural_compressor && cd C:\\tmp\.neural_compressor")
                from urllib import request

                request.urlretrieve(self.mb_fp32_pb_url)
        self.input_graph = tf.compat.v1.GraphDef()
        with open(self.pb_path, "rb") as f:
            self.input_graph.ParseFromString(f.read())

        build_fake_model_1()
        build_fake_model_2()
        build_fake_model_3()

    @classmethod
    def tearDownClass(self):
        os.remove("model_1.pb")
        os.remove("model_2.pb")
        os.remove("model_3.pb")

    def test_identify_input_output(self):
        g = GraphAnalyzer()
        g.graph = self.input_graph
        g.parse_graph()
        inputs, outputs = g.get_graph_input_output()
        self.assertEqual(inputs, self.inputs)
        self.assertEqual(outputs, self.outputs)
        inputs, outputs = get_input_output_node_names(self.input_graph)
        self.assertEqual(inputs, self.inputs)
        self.assertEqual(outputs, self.outputs)

        input_graph = tf.compat.v1.GraphDef()
        with open("model_1.pb", "rb") as f:
            input_graph.ParseFromString(f.read())
        g = GraphAnalyzer()
        g.graph = input_graph
        g.parse_graph()
        inputs, outputs = g.get_graph_input_output()
        self.assertEqual(inputs, ["sub"])
        self.assertEqual(outputs, ["op_to_store"])
        inputs, outputs = get_input_output_node_names(input_graph)
        self.assertEqual(inputs, ["sub"])
        self.assertEqual(outputs, ["op_to_store"])

        input_graph = tf.compat.v1.GraphDef()
        with open("model_2.pb", "rb") as f:
            input_graph.ParseFromString(f.read())
        g = GraphAnalyzer()
        g.graph = input_graph
        g.parse_graph()
        inputs, outputs = g.get_graph_input_output()
        self.assertEqual(inputs, [])
        self.assertEqual(outputs, [])
        inputs, outputs = get_input_output_node_names(input_graph)
        self.assertEqual(inputs, [])
        self.assertEqual(outputs, [])

        input_graph = tf.compat.v1.GraphDef()
        with open("model_3.pb", "rb") as f:
            input_graph.ParseFromString(f.read())
        g = GraphAnalyzer()
        g.graph = input_graph
        g.parse_graph()
        inputs, outputs = g.get_graph_input_output()
        self.assertEqual(inputs, [])
        self.assertEqual(outputs, [])
        inputs, outputs = get_input_output_node_names(input_graph)
        self.assertEqual(inputs, [])
        self.assertEqual(outputs, [])

    def test_no_input_output_config(self):
        from neural_compressor.tensorflow import quantize_model
        from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

        g = GraphAnalyzer()
        g.graph = self.input_graph
        g.parse_graph()

        float_graph_def = g.dump_graph()
        fp32_graph_def = read_graph(self.pb_path)

        dataset = DummyDataset(shape=(20, 224, 224, 3), label=True)
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
        qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

        self.assertGreater(len(qmodel.graph_def.node), 0)

    def test_invalid_input_output_config(self):
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
        from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

        g = GraphAnalyzer()
        g.graph = self.input_graph
        g.parse_graph()
        float_graph_def = g.dump_graph()

        dataset = DummyDataset(shape=(20, 224, 224, 3), label=True)
        calib_dataloader = BaseDataLoader(dataset)
        quant_config = StaticQuantConfig()
        qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

        # will detect the right inputs/outputs
        self.assertNotEqual(qmodel.input_node_names, ["x"])
        self.assertNotEqual(qmodel.output_node_names, ["op_to_store"])


if __name__ == "__main__":
    unittest.main()
