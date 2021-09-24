#
#  -*- coding: utf-8 -*-
#
import shutil
import unittest
import os
import yaml
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from neural_compressor.adaptor.tf_utils.util import disable_random

def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            accuracy_criterion:
              relative: 0.0001
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

def build_fake_yaml_2():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        graph_optimization:
          precisions: [bf16]
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            accuracy_criterion:
              relative: 0.0001
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_2.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

class TestTensorflowRecover(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('test.pb')
        shutil.rmtree('./saved', ignore_errors=True)

    @disable_random()
    def test_tensorflow_recover(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_weights_2 = tf.compat.v1.get_variable("weight_2", [3, 8, 16, 16],
                                                   initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        relu = tf.nn.relu(conv)

        max_pool = tf.nn.max_pool(relu, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        conv_bias = tf.compat.v1.get_variable("bias", [16],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv_1 = tf.nn.conv2d(max_pool, conv_weights_2, strides=[
                              1, 2, 2, 1], padding="VALID", name='conv1_3')
        conv_bias = tf.math.add(conv_1, conv_bias)
        relu6 = tf.nn.relu6(conv_bias, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            with gfile.GFile('./test.pb', "wb") as f:
                f.write(constant_graph.SerializeToString())

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization("./fake_yaml.yaml")
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = constant_graph
            q_model = quantizer()

            from neural_compressor.utils.utility import recover
            recover_model = recover('./test.pb', './saved/history.snapshot', 0)

            q_model_const_value = {}
            for node in q_model.graph_def.node:
                if node.op == "Const":
                    tensor_value = tensor_util.MakeNdarray(node.attr["value"].tensor)
                    if not tensor_value.shape:
                        q_model_const_value[node.name] = tensor_value
            for node in recover_model.graph_def.node:
                if node.op == "Const":
                    tensor_value = tensor_util.MakeNdarray(node.attr["value"].tensor)
                    if node.name in q_model_const_value:
                        self.assertEqual(tensor_value, q_model_const_value[node.name])

class TestTensorflowRecoverForceBF16(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.environ['FORCE_BF16'] = '1'
        build_fake_yaml_2()

    @classmethod
    def tearDownClass(self):
        del os.environ['FORCE_BF16']
        os.remove('fake_yaml_2.yaml')
        os.remove('test.pb')
        shutil.rmtree('./saved', ignore_errors=True)

    @disable_random()
    def test_tensorflow_recover_bf16(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_weights_2 = tf.compat.v1.get_variable("weight_2", [3, 8, 16, 16],
                                                   initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        relu = tf.nn.relu(conv)

        max_pool = tf.nn.max_pool(relu, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        conv_bias = tf.compat.v1.get_variable("bias", [16],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv_1 = tf.nn.conv2d(max_pool, conv_weights_2, strides=[
                              1, 2, 2, 1], padding="VALID", name='conv1_3')
        conv_bias = tf.math.add(conv_1, conv_bias)
        relu6 = tf.nn.relu6(conv_bias, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            with gfile.GFile('./test.pb', "wb") as f:
                f.write(constant_graph.SerializeToString())

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization("./fake_yaml_2.yaml")
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = constant_graph
            q_model = quantizer()
            found_cast_op = False

            from neural_compressor.utils.utility import recover
            recover_model = recover('./test.pb', './saved/history.snapshot', 0)

            q_model_const_value = {}
            for node in q_model.graph_def.node:
                if node.op == "Const":
                    tensor_value = tensor_util.MakeNdarray(node.attr["value"].tensor)
                    if not tensor_value.shape:
                        q_model_const_value[node.name] = tensor_value
            for node in recover_model.graph_def.node:
                if node.op == 'Cast':
                    found_cast_op = True
                    continue
                if node.op == "Const":
                    tensor_value = tensor_util.MakeNdarray(node.attr["value"].tensor)
                    if node.name in q_model_const_value:
                        self.assertEqual(tensor_value, q_model_const_value[node.name])

            self.assertEqual(found_cast_op, True)


if __name__ == "__main__":
    unittest.main()