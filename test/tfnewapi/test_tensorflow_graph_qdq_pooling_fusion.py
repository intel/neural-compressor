#
#  -*- coding: utf-8 -*-
#
import os
import unittest
import yaml
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import dtypes
from neural_compressor.adaptor.tensorflow import TensorflowQuery
from neural_compressor.adaptor.tf_utils.util import disable_random
from tensorflow.python.framework import graph_util

def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: inteltensorflow
          inputs: x
          outputs: op_to_store
        device: cpu
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: basic
            accuracy_criterion:
              relative: 0.01
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


class TestGraphQDQPoolingFusion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        self.op_wise_sequences = TensorflowQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "../../neural_compressor/adaptor/inteltensorflow.yaml")).get_eightbit_patterns(True)

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @disable_random()
    def test_qdq_maxpool_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 30, 30, 1], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [2, 2, 1, 1],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [1],
                                              initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.relu(x)
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        relu2 = tf.nn.relu(relu)
        pool = tf.nn.max_pool(relu2, ksize=1, strides=[1, 2, 2, 1], name='maxpool', padding="SAME")
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

            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 30, 30, 1), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            found_quantized_maxpool = False
            for i in output_graph.graph_def.node:
                if i.op == 'QuantizedMaxPool':
                    found_quantized_maxpool = True
                    break
            self.assertEqual(found_quantized_maxpool, True)
    
    @disable_random()
    def test_qdq_avgpool_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 30, 30, 1], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [2, 2, 1, 1],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [1],
                                              initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.relu(x)
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        relu2 = tf.nn.relu(relu)
        pool = tf.nn.avg_pool(relu2, ksize=1, strides=[1, 2, 2, 1], name='avgpool', padding="SAME")
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

            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 30, 30, 1), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            found_quantized_avgpool = False
            for i in output_graph.graph_def.node:
                if i.op == 'QuantizedAvgPool':
                    found_quantized_avgpool = True
                    break
            self.assertEqual(found_quantized_avgpool, True)

if __name__ == '__main__':
    unittest.main()
