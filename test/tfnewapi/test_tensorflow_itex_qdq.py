#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import shutil
import yaml
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tensorflow import TensorflowQuery
from neural_compressor.adaptor.tf_utils.util import disable_random
from neural_compressor.experimental import Quantization, common
from neural_compressor.utils.utility import CpuInfo

def build_fake_yaml(fake_yaml, save_path, **kwargs):
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open(file=save_path, mode=kwargs['mode'], encoding=kwargs['encoding']) as f:
        yaml.dump(y, f)

class TestItexEnabling(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        fake_yaml_1 = '''
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
              path: workspace_1
        '''

        build_fake_yaml(fake_yaml_1, 'fake_yaml_1.yaml', mode="w", encoding="utf-8")

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml_1.yaml')
        shutil.rmtree('workspace_1')

    @disable_random()
    def test_itex_qdq_basic(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        # relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        # relu2 = tf.nn.relu(normed2)
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

            quantizer = Quantization('fake_yaml_1.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            dequant_count = 0
            for i in output_graph.graph_def.node:
                if i.op == 'Dequantize':
                    dequant_count += 1

            bf16_enabled = False
            if CpuInfo().bf16 or os.getenv('FORCE_BF16') == '1':
                bf16_enabled = True
            if bf16_enabled:
                self.assertEqual(dequant_count, 4)
            else:
                self.assertEqual(dequant_count, 5)

if __name__ == '__main__':
    unittest.main()
