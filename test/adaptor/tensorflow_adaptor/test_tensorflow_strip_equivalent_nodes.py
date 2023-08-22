#
#  -*- coding: utf-8 -*-
#

import unittest
import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from neural_compressor.adaptor.tf_utils.util import disable_random
from neural_compressor.experimental import Quantization, common

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

class TestConvBiasAddAddReluFusion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @disable_random()
    def test_conv_relu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        conv2 = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        conv_add = tf.math.add(conv1, conv2)
        relu6 = tf.nn.relu6(conv_add)
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

            found_conv_num = 0
            for i in output_graph.graph_def.node:
                if "QuantizedConv2D" in i.op:
                    found_conv_num += 1
            self.assertEqual(found_conv_num, 1)


if __name__ == '__main__':
    unittest.main()
