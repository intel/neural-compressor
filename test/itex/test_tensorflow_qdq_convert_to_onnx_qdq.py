#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import shutil
import yaml

from neural_compressor.adaptor.tf_utils.util import disable_random
from neural_compressor.experimental import Quantization, common, Benchmark

from neural_compressor.adaptor.tf_utils.util import version1_lt_version2, version1_gte_version2

import tensorflow as tf
from tensorflow.compat.v1 import graph_util

def build_fake_yaml(fake_yaml, save_path, **kwargs):
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open(file=save_path, mode=kwargs['mode'], encoding=kwargs['encoding']) as f:
        yaml.dump(y, f)

class TestConvertTensorflowQDQToOnnxQDQ(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        fake_yaml = '''
        model:
          name: fake_model_cpu
          framework: tensorflow_itex
          inputs: input
        device: cpu
        quantization:
          model_wise:
            weight:
                granularity: per_tensor
                scheme: sym
                dtype: int8
                algorithm: minmax
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: basic
            accuracy_criterion:
              relative: 0.1
            exit_policy:
              performance_only: True
            workspace:
              path: workspace
        '''
        build_fake_yaml(fake_yaml, 'fake_yaml.yaml', mode="w", encoding="utf-8")

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        if version1_gte_version2(tf.version.VERSION, '2.8.0'):
            shutil.rmtree('workspace')

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0'), "Only supports tf greater 2.7.0")
    def test_convert_tf_qdq_to_onnx_qdq(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        add = tf.raw_ops.Add(x=normed, y=conv2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)

            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            from neural_compressor.config import TF2ONNXConfig
            config = TF2ONNXConfig()
            output_graph.export("workspace/tf_qdq_to_onnx_qdq.onnx", config)

            import onnx
            onnx_model = onnx.load("workspace/tf_qdq_to_onnx_qdq.onnx")
            onnx.checker.check_model(onnx_model)

            import onnxruntime as ort
            from neural_compressor.data import Datasets, DATALOADERS
            ort_session = ort.InferenceSession("workspace/tf_qdq_to_onnx_qdq.onnx")
            dataset = Datasets("tensorflow")["dummy"]((100, 56, 56, 16))
            dataloader = DATALOADERS["tensorflow"](dataset)
            it = iter(dataloader)
            input = next(it)
            input_dict = {'input:0': input[0]}
            ort_session.run(None, input_dict)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0'), "Only supports tf greater 2.7.0")
    def test_convert_tf_fp32_to_onnx_fp32(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        add = tf.raw_ops.Add(x=normed, y=normed2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            from neural_compressor.model import Model
            from neural_compressor.config import TF2ONNXConfig
            inc_model = Model(output_graph_def)
            config = TF2ONNXConfig(dtype="fp32")
            inc_model.export("workspace/tf_fp32_to_onnx_fp32.onnx", config)

            import onnx
            onnx_model = onnx.load("workspace/tf_fp32_to_onnx_fp32.onnx")
            onnx.checker.check_model(onnx_model)

            import onnxruntime as ort
            from neural_compressor.data import Datasets, DATALOADERS
            ort_session = ort.InferenceSession("workspace/tf_fp32_to_onnx_fp32.onnx")
            dataset = Datasets("tensorflow")["dummy"]((100, 56, 56, 16))
            dataloader = DATALOADERS["tensorflow"](dataset)
            it = iter(dataloader)
            input = next(it)
            input_dict = {'input:0': input[0]}
            ort_session.run(None, input_dict)

if __name__ == '__main__':
    unittest.main()
