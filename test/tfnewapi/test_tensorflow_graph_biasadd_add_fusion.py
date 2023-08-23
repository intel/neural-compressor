#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import yaml
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import graph_util
from neural_compressor.adaptor.tensorflow import TensorflowQuery
from neural_compressor.adaptor.tf_utils.util import disable_random


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
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
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


class TestConvBiasAddAddFusion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @disable_random()
    def test_conv_biasadd_add_relu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(top_relu, conv_weights, strides=[1, 2, 2, 1], padding="SAME")
        normed = tf.nn.bias_add(conv, tf.constant([3.0, 1.2,1,2,3,4,5,6,7,8,9,0,12,2,3,4]))
        add = normed + tf.constant([3.0])
        relu = tf.nn.relu6(add)
        mul1 = tf.math.multiply(relu, tf.constant([0.1]) )
        mul2 = tf.math.multiply(mul1, tf.constant([0.8]), name='op_to_store')

        out_name = mul2.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fuse_biasadd_add import FuseBiasAddAndAddOptimizer
            output_graph_def = FuseBiasAddAndAddOptimizer(output_graph_def).do_transformation()

            found_addv2 = False

            for i in output_graph_def.node:
                if i.op.find('AddV2') != -1:
                    found_addv2 = True
                    break

            self.assertEqual(found_addv2, False)

    def test_conv_biasadd_add_relu_no_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.nn.bias_add(conv2, tf.constant([3.0, 1.2,1,2,3,4,5,6,7,8,9,0,12,2,3,4]))
        add_y = tf.compat.v1.get_variable("add_y", [16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        add = normed2 + add_y
        relu = tf.nn.relu6(add)
        mul1 = tf.math.multiply(relu, tf.constant([0.1]) )
        mul2 = tf.math.multiply(mul1, tf.constant([0.8]), name='op_to_store')

        out_name = mul2.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fuse_biasadd_add import FuseBiasAddAndAddOptimizer
            output_graph_def = FuseBiasAddAndAddOptimizer(output_graph_def).do_transformation()

            found_addv2 = False

            for i in output_graph_def.node:
                if i.op.find('AddV2') != -1:
                    found_addv2 = True
                    break

            self.assertEqual(found_addv2, True)

if __name__ == '__main__':
    unittest.main()
