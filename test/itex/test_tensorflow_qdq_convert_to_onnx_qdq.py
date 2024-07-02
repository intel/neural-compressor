import shutil
import unittest

import tensorflow as tf
from tensorflow.compat.v1 import graph_util

from neural_compressor.adaptor.tf_utils.util import disable_random, version1_gte_version2, version1_lt_version2


class TestConvertTensorflowQDQToOnnxQDQ(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        if version1_gte_version2(tf.version.VERSION, "2.8.0"):
            shutil.rmtree("workspace")

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, "2.8.0"), "Only supports tf greater 2.7.0")
    def test_convert_tf_fp32_to_onnx_fp32(self):
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

            from neural_compressor.config import TF2ONNXConfig
            from neural_compressor.model import Model

            inc_model = Model(output_graph_def)
            config = TF2ONNXConfig(dtype="fp32")
            inc_model.export("workspace/tf_fp32_to_onnx_fp32.onnx", config)

            import onnx

            onnx_model = onnx.load("workspace/tf_fp32_to_onnx_fp32.onnx")
            onnx.checker.check_model(onnx_model)

            import onnxruntime as ort

            from neural_compressor.data import DATALOADERS, Datasets

            ort_session = ort.InferenceSession("workspace/tf_fp32_to_onnx_fp32.onnx")
            dataset = Datasets("tensorflow")["dummy"]((100, 56, 56, 16))
            dataloader = DATALOADERS["tensorflow"](dataset)
            it = iter(dataloader)
            input = next(it)
            input_dict = {"input:0": input[0]}
            outputs = ort_session.run(None, input_dict)
            self.assertNotEqual(outputs, None)


if __name__ == "__main__":
    unittest.main()
