#
#  -*- coding: utf-8 -*-
#
import unittest
import os
from lpot.experimental.common import postprocess
import yaml
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import graph_util
from lpot.adaptor.tf_utils.util import disable_random
from lpot.utils.utility import CpuInfo

def build_fake_yaml():
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
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml_2():
    fake_yaml_2 = '''
        model:
          name: fake_yaml_2
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        graph_optimization:
          precisions: [bf16]
        '''
    y = yaml.load(fake_yaml_2, Loader=yaml.SafeLoader)
    with open('fake_yaml_2.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml_3():
    fake_yaml_3 = '''
        model:
          name: fake_yaml_3
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        graph_optimization:
          precisions:
            - bf16
            - fp32
        '''
    y = yaml.load(fake_yaml_3, Loader=yaml.SafeLoader)
    with open('fake_yaml_3.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

class MyMetric(object):
    def __init__(self, *args):
        self.pred_list = []
        self.label_list = []
        self.samples = 0
    def update(self, predict, label):
        self.pred_list.extend(predict)
        self.label_list.extend(label)
        self.samples += len(label)
    def reset(self):
        self.pred_list = []
        self.label_list = []
        self.samples = 0
    def result(self):
        correct_num = np.sum(
                np.array(self.pred_list) == np.array(self.label_list))
        return correct_num / self.samples
        
class TestGraphOptimizationOnNonBF16Host(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @disable_random()
    def test_bf16_cfg_on_non_bf16_enabled_host(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 300, 300, 16], name="input")
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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization, common
            graph_optimizer = Graph_Optimization('fake_yaml.yaml')
            dataset = graph_optimizer.dataset('dummy', shape=(100, 300, 300, 16), label=True)
            graph_optimizer.eval_dataloader = common.DataLoader(dataset)
            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break

            if CpuInfo().bf16:
                self.assertEqual(found_cast_op, True)
            else:
                self.assertEqual(found_cast_op, False)

class TestGraphOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.environ['FORCE_BF16'] = '1'

        build_fake_yaml()
        build_fake_yaml_2()
        build_fake_yaml_3()

    @classmethod
    def tearDownClass(self):
        del os.environ['FORCE_BF16']
        os.remove('fake_yaml.yaml')
        os.remove('fake_yaml_2.yaml')
        os.remove('fake_yaml_3.yaml')

    @disable_random()
    def test_graph_optimization_with_evaluation(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 300, 300, 16], name="input")
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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization, common
            graph_optimizer = Graph_Optimization('fake_yaml.yaml')
            dataset = graph_optimizer.dataset('dummy', shape=(100, 300, 300, 16), label=True)
            graph_optimizer.eval_dataloader = common.DataLoader(dataset)
            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break

            self.assertEqual(found_cast_op, True)

    @disable_random()
    def test_graph_optimization_without_evaluation(self):

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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization, common
            graph_optimizer = Graph_Optimization('fake_yaml_2.yaml')
            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break

            self.assertEqual(found_cast_op, True)

    @disable_random()
    def test_graph_optimization_without_yaml(self):
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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization
            graph_optimizer = Graph_Optimization()
            graph_optimizer.precisions = 'fp32'
            graph_optimizer.input = 'input'
            graph_optimizer.output = 'op_to_store'

            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False
            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break
            input_name = graph_optimizer.input
            output_name = graph_optimizer.output
            self.assertEqual(found_cast_op, False)
            self.assertEqual(input_name, 'input')
            self.assertEqual(output_name, 'op_to_store')

    @disable_random()
    def test_graph_optimization_with_yaml(self):

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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization
            graph_optimizer = Graph_Optimization('fake_yaml_3.yaml')

            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break

            self.assertEqual(found_cast_op, True)

    @disable_random()
    def test_graph_optimization_with_custom_metric_without_postprocess(self):
        os.environ['FORCE_BF16'] = '1'

        x = tf.compat.v1.placeholder(tf.float32, [1, 300, 300, 16], name="input")
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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization, common
            graph_optimizer = Graph_Optimization('fake_yaml_3.yaml')
            graph_optimizer.metric = common.Metric(MyMetric)
            dataset = graph_optimizer.dataset('dummy', shape=(100, 300, 300, 16), label=True)
            graph_optimizer.precisions = ['fp32', 'bf16']
            graph_optimizer.eval_dataloader = common.DataLoader(dataset)
            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break
            metric = graph_optimizer.metric

            self.assertEqual(found_cast_op, True)
            self.assertIsNone(metric)

    @disable_random()
    def test_graph_optimization_without_custom_metric_with_postprocess(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 300, 300, 16], name="input")
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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization, common, data
            graph_optimizer = Graph_Optimization('fake_yaml.yaml')
            dataset = graph_optimizer.dataset('dummy', shape=(100, 300, 300, 16), label=True)
            graph_optimizer.eval_dataloader = common.DataLoader(dataset)
            graph_optimizer.postprocess = common.Postprocess(data.transforms.transform.TensorflowWrapFunction(np.array))
            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break
            
            self.assertEqual(found_cast_op, True)
    
    @disable_random()
    def test_graph_optimization_with_eval_func(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 300, 300, 16], name="input")
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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization, common
            graph_optimizer = Graph_Optimization('fake_yaml.yaml')
            
            dataset = graph_optimizer.dataset('dummy', shape=(100, 300, 300, 16), label=True)
            graph_optimizer.eval_dataloader = common.DataLoader(dataset)
            graph_optimizer.model = output_graph_def
            graph_optimizer.eval_func = None

            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break
            eval_func = graph_optimizer.eval_func
            self.assertEqual(found_cast_op, True)
            self.assertIsNone(eval_func, "eval_func is not None")

    @disable_random()
    def test_graph_optimization_with_force_bf16(self):
        os.environ['FORCE_BF16'] = '1'

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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization
            graph_optimizer = Graph_Optimization()
            graph_optimizer.input = 'input'
            graph_optimizer.output = 'op_to_store'

            graph_optimizer.precisions = 'bf16'
            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break

        self.assertEqual(found_cast_op, True)

class TestGraphOptmizationFP32(unittest.TestCase):

    @disable_random()
    def test_graph_optimization_without_yaml_without_precisions(self):
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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization
            graph_optimizer = Graph_Optimization()
            graph_optimizer.input = 'input'
            graph_optimizer.output = 'op_to_store'

            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break
            precision = graph_optimizer.precisions
            self.assertEqual(found_cast_op, False)
            self.assertEqual(precision, 'fp32')
            
    @disable_random()
    def test_graph_optimization_without_yaml_with_precisions(self):
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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization
            graph_optimizer = Graph_Optimization()
            graph_optimizer.precisions = 'fp32'

            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break

            self.assertEqual(found_cast_op, False)

    @disable_random()
    def test_graph_optimization_fp32_only_with_force_bf16(self):
        os.environ['FORCE_BF16'] = '1'

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
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Graph_Optimization
            graph_optimizer = Graph_Optimization()
            graph_optimizer.input = 'input'
            graph_optimizer.output = 'op_to_store'

            graph_optimizer.model = output_graph_def
            output_graph = graph_optimizer()
            found_cast_op = False

            for i in output_graph.graph_def.node:
                if i.op == 'Cast':
                    found_cast_op = True
                    break

            self.assertEqual(found_cast_op, False)

if __name__ == "__main__":
    unittest.main()
