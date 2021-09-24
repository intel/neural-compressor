#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import yaml
import tensorflow as tf

from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tf_utils.util import disable_random


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
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
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


class TestMetaPass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @disable_random()
    def test_tensorflow_graph_meta_pass_with_different_mode(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(top_relu, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        sq = tf.squeeze(relu, [0])
        reshape = tf.reshape(sq, [729, 16])
        conv_weights2 = tf.compat.v1.get_variable("weight2", [16, 729],
                                                  initializer=tf.compat.v1.random_normal_initializer())

        matmul = tf.matmul(reshape, conv_weights2)
        # normed2 = tf.compat.v1.layers.batch_normalization(matmul)
        bias = tf.compat.v1.get_variable("bias", [729],
                                         initializer=tf.compat.v1.random_normal_initializer())
        normed2 = tf.nn.bias_add(matmul, bias, name='bias_add')

        relu6 = tf.nn.relu6(normed2)
        reshape2 = tf.reshape(relu6, [1, 729, 729, 1], name='op_to_store')

        out_name = reshape2.name.split(':')[0]

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            found_reshape = False
            for i in output_graph.graph_def.node:
                if i.op == 'Reshape':
                    found_reshape = True
                    break

            self.assertEqual(found_reshape, True)

    @disable_random()
    def test_tensorflow_graph_meta_pass_with_same_mode(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(top_relu, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        sq = tf.squeeze(relu, [0])
        reshape = tf.reshape(sq, [1, 27, 27, 16])
        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(reshape, conv_weights2, strides=[1, 2, 2, 1], padding="VALID")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)

        relu6 = tf.nn.relu6(normed2, name='op_to_store')

        out_name = relu6.name.split(':')[0]

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()
            quantize_count = 0
            dequantize_count = 0

            for i in output_graph.graph_def.node:
                if i.op == 'QuantizeV2':
                    quantize_count += 1
                if i.op == 'Dequantize':
                    dequantize_count += 1

            self.assertEqual(quantize_count, 1)
            self.assertEqual(dequantize_count, 1)


if __name__ == '__main__':
    unittest.main()
