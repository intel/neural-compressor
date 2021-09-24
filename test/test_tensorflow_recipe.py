#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import yaml
import tensorflow as tf

from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tf_utils.util import disable_random


def build_fake_yaml_disable_first_quantization():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          recipes:
            first_conv_or_matmul_quantization: False
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
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_disable_first_quantization.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml_enable_first_quantization():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          recipes:
            first_conv_or_matmul_quantization: True
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
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_enable_first_quantization.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml_disable_scale_propagation():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          recipes:
            scale_propagation_max_pooling: False
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
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_disable_scale_propagation.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml_enable_scale_propagation():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          recipes:
            scale_propagation_max_pooling: True
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
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_enable_scale_propagation.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml_enable_scale_unification():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          recipes:
            scale_propagation_concat: True
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
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_enable_scale_unification.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml_disable_scale_unification():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          recipes:
            scale_propagation_concat: False
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
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_disable_scale_unification.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

class TestTensorflowInt8Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml_disable_first_quantization()
        build_fake_yaml_enable_first_quantization()
        build_fake_yaml_disable_scale_propagation()
        build_fake_yaml_enable_scale_propagation()
        build_fake_yaml_enable_scale_unification()
        build_fake_yaml_disable_scale_unification()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml_disable_first_quantization.yaml')
        os.remove('fake_yaml_enable_first_quantization.yaml')
        os.remove('fake_yaml_disable_scale_propagation.yaml')
        os.remove('fake_yaml_enable_scale_propagation.yaml')
        os.remove('fake_yaml_disable_scale_unification.yaml')
        os.remove('fake_yaml_enable_scale_unification.yaml')

    @disable_random()
    def test_disable_first_quantization(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)

        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization('fake_yaml_disable_first_quantization.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            found_fp32_conv = False

            for i in output_graph.graph_def.node:
              if i.op == 'Conv2D':
                found_fp32_conv = True
                break

            self.assertEqual(found_fp32_conv, True)

    @disable_random()
    def test_enable_first_quantization(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)

        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization('fake_yaml_enable_first_quantization.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            found_fp32_conv = False

            for i in output_graph.graph_def.node:
              if i.op == 'Conv2D':
                  found_fp32_conv = True
                  break

            self.assertEqual(found_fp32_conv, False)

    @disable_random()
    def test_enable_scale_propagation(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 30, 30, 1], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [2, 2, 1, 1],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [1],
                                              initializer=tf.compat.v1.random_normal_initializer())

        x = tf.nn.relu(x)
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        pool = tf.nn.avg_pool(relu, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        conv1 = tf.nn.conv2d(pool, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        x = tf.nn.relu(conv_bias)
        final_node = tf.nn.relu(x, name='op_to_store')

        out_name = final_node.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization('fake_yaml_enable_scale_propagation.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 30, 30, 1), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            max_freezed_out = []
            for i in output_graph.graph_def.node:
              if i.op == 'QuantizedConv2DWithBiasAndReluAndRequantize':
                max_freezed_out.append(i.input[-1])

            self.assertEqual(1, len(set(max_freezed_out)))

    @disable_random()
    def test_disable_scale_propagation(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 30, 30, 1], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [2, 2, 1, 1],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [1],
                                              initializer=tf.compat.v1.random_normal_initializer())

        x = tf.nn.relu(x)
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        pool = tf.nn.avg_pool(relu, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        conv1 = tf.nn.conv2d(pool, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        x = tf.nn.relu(conv_bias)
        final_node = tf.nn.relu(x, name='op_to_store')

        out_name = final_node.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization('fake_yaml_disable_scale_propagation.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 30, 30, 1), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            max_freezed_out = []
            for i in output_graph.graph_def.node:
              if i.op == 'QuantizedConv2DWithBiasAndReluAndRequantize':
                  max_freezed_out.append(i.input[-1])
            self.assertEqual(2, len(set(max_freezed_out)))

    @disable_random()
    def test_enable_scale_unification(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 128, 16], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [2, 2, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [16],
                                              initializer=tf.compat.v1.random_normal_initializer())

        x = tf.nn.relu(x)
        sqrt = tf.math.sqrt(x)
        relu_sqrt = tf.nn.relu(sqrt)
        conv = tf.nn.conv2d(relu_sqrt, conv_weights, strides=[
                            1, 2, 2, 1], padding="SAME", name='last')
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        relu1 = tf.nn.relu(conv_bias)
        concat = tf.concat([relu, relu1], 1)
        final_node = tf.nn.relu(concat, name='op_to_store')
        out_name = final_node.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization('fake_yaml_enable_scale_unification.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 128, 128, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            max_freezed_out = []
            for i in output_graph.graph_def.node:
              if i.op == 'QuantizedConv2DWithBiasAndReluAndRequantize':
                  max_freezed_out.append(i.input[-1])
            self.assertEqual(1, len(set(max_freezed_out)))

    @disable_random()
    def test_disable_scale_unification(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 30, 30, 1], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [2, 2, 1, 1],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [1],
                                              initializer=tf.compat.v1.random_normal_initializer())

        x = tf.nn.relu(x)
        sqrt = tf.math.sqrt(x)
        relu_sqrt = tf.nn.relu(sqrt)
        conv = tf.nn.conv2d(relu_sqrt, conv_weights, strides=[
                            1, 2, 2, 1], padding="SAME", name='last')
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        relu1 = tf.nn.relu(conv_bias)
        concat = tf.concat([relu, relu1], 1)
        final_node = tf.nn.relu(concat, name='op_to_store')
        out_name = final_node.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization('fake_yaml_disable_scale_unification.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 30, 30, 1), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            max_freezed_out = []
            for i in output_graph.graph_def.node:
              if i.op == 'QuantizedConv2DWithBiasAndReluAndRequantize':
                  max_freezed_out.append(i.input[-1])
            self.assertEqual(2, len(set(max_freezed_out)))

if __name__ == '__main__':
    unittest.main()
