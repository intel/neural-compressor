#
#  -*- coding: utf-8 -*-
#
import os
import unittest
import tensorflow as tf

from tensorflow.python.framework import graph_util
from lpot.adaptor.tf_utils.util import disable_random


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          op_wise: {
                     \"conv1_[1-2]\": {
                       \"activation\":  {\"dtype\": [\"fp32\"]},
                     },
                   }
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: basic
            exit_policy:
              timeout: 0
            accuracy_criterion:
              relative: 0.05
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        '''
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        f.write(fake_yaml)
    f.close()


def build_fake_yaml_invalid_model_wise():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          op_wise: {
                     \"conv1_[1-2]\": {
                       \"activation\":  {\"dtype\": [\"fp32\"]},
                     },
                   }
          model_wise:
            weight:
                granularity: per_channel
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
              timeout: 0
            accuracy_criterion:
              relative: 0.05
            workspace:
              path: saved
        '''
    with open('fake_yaml_with_invalid_cfg.yaml', "w", encoding="utf-8") as f:
        f.write(fake_yaml)
    f.close()


class TestConfigRegex(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        build_fake_yaml_invalid_model_wise()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('fake_yaml_with_invalid_cfg.yaml')

    @disable_random()
    def test_config_regex(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_weights_2 = tf.compat.v1.get_variable("weight_2", [3, 8, 16, 16],
                                                   initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[
                            1, 2, 2, 1], padding="VALID", name='conv1_1')
        normed1 = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed1)
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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Quantization, common

            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            found_fp32_conv = False
            found_quantized_conv = False
            for i in output_graph.graph_def.node:
                if i.op == 'Conv2D' and i.name == 'conv1_1':
                    found_fp32_conv = True

                if i.op.find("QuantizedConv2D") != -1 and i.name == 'conv1_3_eightbit_requantize':
                    found_quantized_conv = True

            self.assertEqual(found_fp32_conv, True)
            self.assertEqual(found_quantized_conv, True)

    def test_config_regex_with_invalid_cfg(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(1)
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_weights_2 = tf.compat.v1.get_variable("weight_2", [3, 8, 16, 16],
                                                   initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[
                            1, 2, 2, 1], padding="VALID", name='conv1_1')
        normed1 = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed1)
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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Quantization, common

            quantizer = Quantization('fake_yaml_with_invalid_cfg.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            found_fp32_conv = False
            found_quantized_conv = False
            for i in output_graph.graph_def.node:
                if i.op == 'Conv2D' and i.name == 'conv1_1':
                    found_fp32_conv = True

                if i.op.find("QuantizedConv2D") != -1 and i.name == 'conv1_3_eightbit_requantize':
                    found_quantized_conv = True

            self.assertEqual(found_fp32_conv, True)
            self.assertEqual(found_quantized_conv, True)


if __name__ == '__main__':
    unittest.main()
